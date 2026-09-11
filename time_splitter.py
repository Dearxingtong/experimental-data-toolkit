from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from io import BytesIO
from pathlib import Path
from typing import Callable
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


TIMESTAMP_COLUMN = "TIMESTAMP"
POSITION_SUFFIXES = ("position1", "position2", "position3")


@dataclass
class TimeSeriesSheet:
    sheet_name: str
    timestamp_column: str
    parsed_timestamps: pd.Series

    @property
    def row_count(self) -> int:
        return len(self.parsed_timestamps)

    @property
    def first_timestamp(self) -> pd.Timestamp:
        return self.parsed_timestamps.iloc[0]

    @property
    def last_timestamp(self) -> pd.Timestamp:
        return self.parsed_timestamps.iloc[-1]


@dataclass
class WorkbookAnalysis:
    filename: str
    worksheets: dict[str, pd.DataFrame]
    time_series_sheets: list[TimeSeriesSheet]
    errors: list[str]

    @property
    def primary_sheet(self) -> TimeSeriesSheet | None:
        if not self.time_series_sheets:
            return None
        return self.time_series_sheets[0]

    @property
    def can_split(self) -> bool:
        return bool(self.time_series_sheets) and not self.errors


@dataclass
class PositionSummary:
    label: str
    rows: int
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None


@dataclass
class SheetSplitQC:
    sheet_name: str
    original_rows: int
    position_rows: tuple[int, ...]
    total_rows: int
    all_rows_accounted_for: bool
    no_duplicate_assignments: bool

    @property
    def passed(self) -> bool:
        return self.all_rows_accounted_for and self.no_duplicate_assignments


@dataclass
class SplitResult:
    filename: str
    boundary1: datetime
    boundary2: datetime
    position_summaries: tuple[PositionSummary, PositionSummary, PositionSummary]
    qc_results: list[SheetSplitQC]
    output_files: list[tuple[str, bytes]]


@dataclass
class PositionRange:
    label: str
    start_time: time
    end_time: time
    duration_minutes: int | None = None
    end_is_exclusive: bool = False
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    position_number: int | None = None
    split_number: int | None = None

    @property
    def output_date(self) -> date:
        if self.start_datetime is not None:
            return self.start_datetime.date()
        return datetime.today().date()

    @property
    def start_sort_value(self) -> datetime:
        if self.start_datetime is not None:
            return self.start_datetime
        return datetime.combine(datetime.today().date(), self.start_time)


@dataclass
class PositionRangeResult:
    filename: str
    position_summaries: tuple[PositionSummary, ...]
    qc_results: list[SheetSplitQC]
    output_files: list[tuple[str, bytes]]
    output_workbooks: list[dict[str, pd.DataFrame]]


def read_xlsx_workbook(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=None, engine="openpyxl")


def parse_hhmm_time(value: str) -> tuple[time | None, str | None]:
    stripped = value.strip()
    if len(stripped) != 5 or stripped[2] != ":":
        return None, "Please enter time in HH:MM format."

    hour_text, minute_text = stripped.split(":", 1)
    if not hour_text.isdigit() or not minute_text.isdigit():
        return None, "Please enter time in HH:MM format."

    hour = int(hour_text)
    minute = int(minute_text)
    if hour > 23 or minute > 59:
        return None, "Please enter time in HH:MM format."

    return time(hour, minute), None


def parse_time_component(
    value: str,
    position_label: str,
    component_label: str,
    maximum: int,
) -> tuple[int | None, str | None]:
    stripped = value.strip()
    if not stripped:
        return None, f"Please enter {position_label} {component_label}."

    if not stripped.isdigit():
        return None, f"{position_label} {component_label} must be between 00 and {maximum:02d}."

    parsed_value = int(stripped)
    if parsed_value > maximum:
        return None, f"{position_label} {component_label} must be between 00 and {maximum:02d}."

    return parsed_value, None


def build_time_range_from_parts(
    position_label: str,
    start_hour_text: str,
    start_minute_text: str,
    end_hour_text: str,
    end_minute_text: str,
) -> tuple[time | None, time | None, list[str]]:
    """Convert separate HH/MM text inputs into validated start and end times."""
    errors: list[str] = []
    start_hour, start_hour_error = parse_time_component(
        start_hour_text,
        position_label,
        "Start Hour",
        23,
    )
    start_minute, start_minute_error = parse_time_component(
        start_minute_text,
        position_label,
        "Start Minute",
        59,
    )
    end_hour, end_hour_error = parse_time_component(
        end_hour_text,
        position_label,
        "End Hour",
        23,
    )
    end_minute, end_minute_error = parse_time_component(
        end_minute_text,
        position_label,
        "End Minute",
        59,
    )

    for error in (
        start_hour_error,
        start_minute_error,
        end_hour_error,
        end_minute_error,
    ):
        if error:
            errors.append(error)

    if errors:
        return None, None, errors

    return time(start_hour, start_minute), time(end_hour, end_minute), []


def floor_to_minute(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.floor("min")


def generate_minute_time_options(
    first_timestamp: pd.Timestamp,
    last_timestamp: pd.Timestamp,
) -> list[time]:
    """Generate one-minute start choices from the detected experiment range."""
    start = floor_to_minute(first_timestamp)
    end = floor_to_minute(last_timestamp)
    if end < start:
        return []

    return [
        timestamp.time().replace(second=0, microsecond=0)
        for timestamp in pd.date_range(start, end, freq="min")
    ]


def available_dates_from_timestamps(timestamps: pd.Series) -> list[date]:
    return sorted(pd.Series(timestamps.dt.date).dropna().unique().tolist())


def generate_minute_time_options_for_date(
    timestamps: pd.Series,
    selected_date: date,
) -> list[time]:
    selected = timestamps[timestamps.dt.date == selected_date]
    if selected.empty:
        return []
    return generate_minute_time_options(selected.iloc[0], selected.iloc[-1])


def format_hhmm(value: time) -> str:
    return value.strftime("%H:%M")


def combine_date_and_time(selected_date: date, selected_time: time) -> datetime:
    return datetime.combine(selected_date, selected_time)


def format_calculated_end(start_datetime: datetime, end_datetime: datetime) -> str:
    if start_datetime.date() == end_datetime.date():
        return end_datetime.strftime("%H:%M")
    return end_datetime.strftime("%Y-%m-%d %H:%M")


def format_duration_option(duration_minutes: int) -> str:
    return f"{duration_minutes} min"


def duration_options() -> list[int]:
    return list(range(46))


def validate_duration_minutes(label: str, duration_minutes: int) -> str | None:
    if duration_minutes == 0:
        return f"Please select a duration greater than 0 minutes for {label}."
    if duration_minutes < 0 or duration_minutes > 45:
        return f"{label} duration must be between 1 and 45 minutes."
    return None


def add_minutes_to_time(start_time: time, duration_minutes: int) -> time:
    base = datetime.combine(datetime.today().date(), start_time)
    return (base + timedelta(minutes=duration_minutes)).time().replace(second=0, microsecond=0)


def build_position_range_from_duration(
    label: str,
    start_time: time,
    duration_minutes: int,
    selected_date: date | None = None,
    position_number: int | None = None,
    split_number: int | None = None,
) -> tuple[PositionRange | None, str | None]:
    duration_error = validate_duration_minutes(label, duration_minutes)
    if duration_error:
        return None, duration_error

    start_datetime = None
    end_datetime = None
    if selected_date is not None:
        start_datetime = combine_date_and_time(selected_date, start_time)
        end_datetime = start_datetime + timedelta(minutes=duration_minutes)
        calculated_end = end_datetime.time().replace(second=0, microsecond=0)
    else:
        calculated_end = add_minutes_to_time(start_time, duration_minutes)
    return (
        PositionRange(
            label=label,
            start_time=start_time,
            end_time=calculated_end,
            duration_minutes=duration_minutes,
            end_is_exclusive=True,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            position_number=position_number,
            split_number=split_number,
        ),
        None,
    )


def find_timestamp_column(dataframe: pd.DataFrame) -> str | None:
    for column in dataframe.columns:
        if str(column).strip() == TIMESTAMP_COLUMN:
            return column
    return None


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series)

    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(f"{bad_count} TIMESTAMP value(s) could not be parsed.")
    return parsed


def analyze_workbook_bytes(file_bytes: bytes, filename: str) -> WorkbookAnalysis:
    try:
        worksheets = read_xlsx_workbook(file_bytes)
    except Exception as exc:
        return WorkbookAnalysis(filename, {}, [], [f"{filename} could not be read: {exc}"])

    return analyze_workbook_dataframes(worksheets, filename)


def analyze_workbook_dataframes(
    worksheets: dict[str, pd.DataFrame],
    filename: str,
) -> WorkbookAnalysis:
    time_series_sheets: list[TimeSeriesSheet] = []
    errors: list[str] = []

    for sheet_name, dataframe in worksheets.items():
        timestamp_column = find_timestamp_column(dataframe)
        if timestamp_column is None:
            continue

        try:
            parsed_timestamps = parse_timestamp_series(dataframe[timestamp_column])
        except Exception as exc:
            errors.append(
                f"{filename} sheet {sheet_name} has unparseable TIMESTAMP values: {exc}"
            )
            continue

        if parsed_timestamps.empty:
            errors.append(f"{filename} sheet {sheet_name} has no timestamped rows.")
            continue

        if not parsed_timestamps.is_monotonic_increasing:
            errors.append(
                f"{filename} sheet {sheet_name} has TIMESTAMP values out of chronological order."
            )
            continue

        time_series_sheets.append(
            TimeSeriesSheet(sheet_name, timestamp_column, parsed_timestamps)
        )

    if not time_series_sheets and not errors:
        errors.append(f"{filename} does not contain a worksheet with a TIMESTAMP column.")

    return WorkbookAnalysis(filename, worksheets, time_series_sheets, errors)


def analyze_xlsx_upload(uploaded_file) -> WorkbookAnalysis:
    return analyze_workbook_bytes(uploaded_file.getvalue(), uploaded_file.name)


def build_boundaries(
    experiment_date,
    end_position1: time,
    end_position2: time,
) -> tuple[datetime, datetime]:
    boundary1 = datetime.combine(experiment_date, end_position1) + timedelta(minutes=1)
    boundary2 = datetime.combine(experiment_date, end_position2) + timedelta(minutes=1)
    return boundary1, boundary2


def validate_cut_times(
    analysis: WorkbookAnalysis,
    end_position1: time,
    end_position2: time,
) -> tuple[datetime | None, datetime | None, str | None]:
    primary = analysis.primary_sheet
    if primary is None:
        return None, None, f"{analysis.filename} cannot be split because no TIMESTAMP data was found."

    if end_position2 <= end_position1:
        return None, None, "End of Position 2 must be later than End of Position 1."

    experiment_date = primary.first_timestamp.date()
    cut1 = datetime.combine(experiment_date, end_position1)
    cut2 = datetime.combine(experiment_date, end_position2)
    start = primary.first_timestamp.to_pydatetime()
    end = primary.last_timestamp.to_pydatetime()

    boundary1, boundary2 = build_boundaries(experiment_date, end_position1, end_position2)

    if boundary1 <= start or boundary2 <= start or cut1 > end or cut2 > end:
        return None, None, "Selected cut time is outside the experimental time range."

    return boundary1, boundary2, None


def split_dataframe_by_boundaries(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    boundary1: datetime,
    boundary2: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed = parse_timestamp_series(dataframe[timestamp_column])
    mask1 = parsed < boundary1
    mask2 = (parsed >= boundary1) & (parsed < boundary2)
    mask3 = parsed >= boundary2
    return (
        dataframe.loc[mask1].copy(),
        dataframe.loc[mask2].copy(),
        dataframe.loc[mask3].copy(),
    )


def range_to_boundaries(experiment_date, position_range: PositionRange) -> tuple[datetime, datetime]:
    if position_range.start_datetime is not None and position_range.end_datetime is not None:
        return position_range.start_datetime, position_range.end_datetime
    start_boundary = datetime.combine(experiment_date, position_range.start_time)
    end_boundary = datetime.combine(experiment_date, position_range.end_time)
    if position_range.end_time < position_range.start_time:
        end_boundary += timedelta(days=1)
    if not position_range.end_is_exclusive:
        end_boundary += timedelta(minutes=1)
    return start_boundary, end_boundary


def split_dataframe_by_range(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    start_boundary: datetime,
    end_boundary: datetime,
) -> pd.DataFrame:
    parsed = parse_timestamp_series(dataframe[timestamp_column])
    mask = (parsed >= start_boundary) & (parsed < end_boundary)
    return dataframe.loc[mask].copy()


def summarize_position(label: str, dataframe: pd.DataFrame, timestamp_column: str | None) -> PositionSummary:
    if dataframe.empty or timestamp_column is None:
        return PositionSummary(label, len(dataframe), None, None)

    parsed = parse_timestamp_series(dataframe[timestamp_column])
    return PositionSummary(label, len(dataframe), parsed.iloc[0], parsed.iloc[-1])


def check_split_qc(
    sheet_name: str,
    original_dataframe: pd.DataFrame,
    split_dataframes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> SheetSplitQC:
    original_rows = len(original_dataframe)
    position_rows = tuple(len(frame) for frame in split_dataframes)
    total_rows = sum(position_rows)
    assigned_indices: list[int] = []
    for frame in split_dataframes:
        assigned_indices.extend(frame.index.tolist())

    return SheetSplitQC(
        sheet_name=sheet_name,
        original_rows=original_rows,
        position_rows=position_rows,
        total_rows=total_rows,
        all_rows_accounted_for=total_rows == original_rows,
        no_duplicate_assignments=len(assigned_indices) == len(set(assigned_indices)),
    )


def check_position_range_qc(
    sheet_name: str,
    split_dataframes: tuple[pd.DataFrame, ...],
) -> bool:
    # Overlapping split intervals are allowed for repeated measurements. Each
    # output must still contain each source observation at most once.
    for frame in split_dataframes:
        if len(frame.index.tolist()) != len(set(frame.index.tolist())):
            return False
    return True


def position_number_from_range(position_range: PositionRange) -> int:
    if position_range.position_number is not None:
        return position_range.position_number
    digits = "".join(character for character in position_range.label if character.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"{position_range.label} does not include a physical position number.")


def validate_position_ranges(
    analysis: WorkbookAnalysis,
    position_ranges: list[PositionRange],
) -> tuple[list[tuple[datetime, datetime]] | None, str | None]:
    primary = analysis.primary_sheet
    if primary is None:
        return None, f"{analysis.filename} cannot be split because no TIMESTAMP data was found."

    experiment_start = primary.first_timestamp.to_pydatetime()
    experiment_end = primary.last_timestamp.to_pydatetime()
    boundaries: list[tuple[datetime, datetime]] = []
    duplicate_keys: set[tuple[int, datetime]] = set()

    for position_range in position_ranges:
        if (
            not position_range.end_is_exclusive
            and position_range.start_time > position_range.end_time
        ):
            return None, f"{position_range.label}: Start Time cannot be later than End Time."

        start_boundary, end_boundary = range_to_boundaries(primary.first_timestamp.date(), position_range)
        duplicate_key = (position_number_from_range(position_range), start_boundary)
        if duplicate_key in duplicate_keys:
            return None, f"{position_range.label}: duplicate Position and Start Datetime."
        duplicate_keys.add(duplicate_key)

        if end_boundary <= start_boundary:
            return None, f"{position_range.label}: calculated end must be later than start."
        if end_boundary <= experiment_start or start_boundary > experiment_end:
            return None, f"{position_range.label}: requested range is outside the experimental time range."

        split_frame = split_dataframe_by_range(
            analysis.worksheets[primary.sheet_name],
            primary.timestamp_column,
            start_boundary,
            end_boundary,
        )
        if split_frame.empty:
            return None, f"{position_range.label}: requested range contains no observations."

        boundaries.append((start_boundary, end_boundary))

    return boundaries, None


def split_workbook_by_position_ranges(
    analysis: WorkbookAnalysis,
    position_ranges: list[PositionRange],
    output_filenames: list[str],
    workbook_transformer: Callable[[int, dict[str, pd.DataFrame]], dict[str, pd.DataFrame]] | None = None,
) -> tuple[PositionRangeResult | None, str | None]:
    if analysis.errors:
        return None, " ".join(analysis.errors)
    if len(position_ranges) != len(output_filenames):
        return None, "Position ranges and output filenames must have the same count."

    boundaries, validation_error = validate_position_ranges(analysis, position_ranges)
    if validation_error or boundaries is None:
        return None, validation_error

    timestamp_sheets = {
        sheet.sheet_name: sheet.timestamp_column for sheet in analysis.time_series_sheets
    }
    primary = analysis.primary_sheet
    if primary is None:
        return None, f"{analysis.filename} cannot be split because no TIMESTAMP data was found."

    output_workbooks: list[dict[str, pd.DataFrame]] = [
        {} for _position_range in position_ranges
    ]
    position_summaries: list[PositionSummary] = []
    qc_results: list[SheetSplitQC] = []

    for sheet_name, dataframe in analysis.worksheets.items():
        timestamp_column = timestamp_sheets.get(sheet_name)
        if timestamp_column is None:
            for output_workbook in output_workbooks:
                output_workbook[sheet_name] = dataframe.copy()
            continue

        sheet_splits = tuple(
            split_dataframe_by_range(dataframe, timestamp_column, start, end)
            for start, end in boundaries
        )

        if sheet_name == primary.sheet_name:
            position_summaries = [
                summarize_position(position_range.label, split_frame, timestamp_column)
                for position_range, split_frame in zip(position_ranges, sheet_splits)
            ]

        no_duplicate_assignments = check_position_range_qc(sheet_name, sheet_splits)
        qc_results.append(
            SheetSplitQC(
                sheet_name=sheet_name,
                original_rows=len(dataframe),
                position_rows=tuple(len(frame) for frame in sheet_splits),
                total_rows=sum(len(frame) for frame in sheet_splits),
                all_rows_accounted_for=True,
                no_duplicate_assignments=no_duplicate_assignments,
            )
        )
        if not no_duplicate_assignments:
            return None, f"QC checks failed for worksheet {sheet_name}: overlapping observations detected."

        for index, split_frame in enumerate(sheet_splits):
            output_workbooks[index][sheet_name] = split_frame.reset_index(drop=True)

    output_files = []
    final_output_workbooks = []
    for position_range, output_filename, output_workbook in zip(
        position_ranges,
        output_filenames,
        output_workbooks,
    ):
        if workbook_transformer is not None:
            output_workbook = workbook_transformer(position_number_from_range(position_range), output_workbook)
        final_output_workbooks.append(output_workbook)
        output_files.append((output_filename, write_xlsx_workbook(output_workbook)))

    return (
        PositionRangeResult(
            filename=analysis.filename,
            position_summaries=tuple(position_summaries),
            qc_results=qc_results,
            output_files=output_files,
            output_workbooks=final_output_workbooks,
        ),
        None,
    )


def validate_no_empty_positions(
    position_summaries: tuple[PositionSummary, PositionSummary, PositionSummary],
) -> str | None:
    for summary in position_summaries:
        if summary.rows == 0:
            return f"{summary.label} would be empty with the selected cut times."
    return None


def split_workbook(
    analysis: WorkbookAnalysis,
    end_position1: time,
    end_position2: time,
) -> tuple[SplitResult | None, str | None]:
    if analysis.errors:
        return None, " ".join(analysis.errors)

    boundary1, boundary2, validation_error = validate_cut_times(
        analysis,
        end_position1,
        end_position2,
    )
    if validation_error or boundary1 is None or boundary2 is None:
        return None, validation_error

    primary = analysis.primary_sheet
    if primary is None:
        return None, f"{analysis.filename} cannot be split because no TIMESTAMP data was found."

    timestamp_sheets = {
        sheet.sheet_name: sheet.timestamp_column for sheet in analysis.time_series_sheets
    }
    split_workbooks: list[dict[str, pd.DataFrame]] = [{}, {}, {}]
    qc_results: list[SheetSplitQC] = []
    primary_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None

    for sheet_name, dataframe in analysis.worksheets.items():
        timestamp_column = timestamp_sheets.get(sheet_name)
        if timestamp_column is None:
            for workbook in split_workbooks:
                workbook[sheet_name] = dataframe.copy()
            continue

        split_dataframes = split_dataframe_by_boundaries(
            dataframe,
            timestamp_column,
            boundary1,
            boundary2,
        )
        if sheet_name == primary.sheet_name:
            primary_splits = split_dataframes

        qc_results.append(check_split_qc(sheet_name, dataframe, split_dataframes))
        for index, split_dataframe in enumerate(split_dataframes):
            split_workbooks[index][sheet_name] = split_dataframe.reset_index(drop=True)

    if primary_splits is None:
        return None, f"{analysis.filename} cannot be split because no TIMESTAMP data was found."

    position_summaries = (
        summarize_position("Position 1", primary_splits[0], primary.timestamp_column),
        summarize_position("Position 2", primary_splits[1], primary.timestamp_column),
        summarize_position("Position 3", primary_splits[2], primary.timestamp_column),
    )
    empty_error = validate_no_empty_positions(position_summaries)
    if empty_error:
        return None, empty_error

    failed_qc = [qc for qc in qc_results if not qc.passed]
    if failed_qc:
        failed_sheets = ", ".join(qc.sheet_name for qc in failed_qc)
        return None, f"QC checks failed for worksheet(s): {failed_sheets}."

    output_files = [
        (position_output_filename(analysis.filename, index + 1), write_xlsx_workbook(workbook))
        for index, workbook in enumerate(split_workbooks)
    ]

    return (
        SplitResult(
            filename=analysis.filename,
            boundary1=boundary1,
            boundary2=boundary2,
            position_summaries=position_summaries,
            qc_results=qc_results,
            output_files=output_files,
        ),
        None,
    )


def position_output_filename(filename: str, position_number: int) -> str:
    stem = Path(filename).stem
    return f"{stem}_position{position_number}.xlsx"


def write_xlsx_workbook(worksheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, dataframe in worksheets.items():
            safe_name = sheet_name[:31] or "Sheet1"
            dataframe.to_excel(writer, index=False, sheet_name=safe_name)
    return output.getvalue()


def build_zip(files: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for filename, file_bytes in files:
            archive.writestr(filename, file_bytes)
    return output.getvalue()
