from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from io import BytesIO
from pathlib import Path
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
    position_rows: tuple[int, int, int]
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


def read_xlsx_workbook(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=None, engine="openpyxl")


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

        dates = pd.Series(parsed_timestamps.dt.date).dropna().unique()
        if len(dates) != 1:
            errors.append(
                "This version of Split by Time supports single-day experimental files only."
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
