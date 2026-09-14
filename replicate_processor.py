from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


TIMESTAMP_COLUMN = "TIMESTAMP"
DEFAULT_REPLICATE_COUNT = 3
DEFAULT_REPLICATE_COLUMNS = (
    "U1",
    "V1",
    "W1",
    "U2",
    "V2",
    "W2",
    "U3",
    "V3",
    "W3",
    "Vx1",
    "Vy1",
    "Vz1",
    "Vx2",
    "Vy2",
    "Vz2",
    "Vx3",
    "Vy3",
    "Vz3",
)


@dataclass
class ReplicateDataset:
    filename: str
    dataframe: pd.DataFrame


@dataclass
class AlignmentReport:
    method: str
    matched_rows: int
    unmatched_rows: dict[str, int]
    warnings: list[str]
    input_rows: dict[str, int] | None = None


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if column != TIMESTAMP_COLUMN
        and not pd.to_numeric(dataframe[column], errors="coerce").dropna().empty
    ]


def matching_numeric_columns(
    datasets: list[ReplicateDataset],
    include_optional_numeric: bool = True,
) -> tuple[list[str], list[str]]:
    if not datasets:
        return [], ["Upload at least one replicate file."]

    column_sets = [set(numeric_columns(dataset.dataframe)) for dataset in datasets]
    common_columns = set.intersection(*column_sets) if column_sets else set()
    missing_errors: list[str] = []

    candidate_columns = [
        column
        for column in DEFAULT_REPLICATE_COLUMNS
        if any(column in columns for columns in column_sets)
    ]
    for replicate_index, (dataset, columns) in enumerate(zip(datasets, column_sets), start=1):
        missing = [column for column in candidate_columns if column not in columns]
        if missing:
            missing_errors.append(
                f"Replicate {replicate_index} is missing column: {', '.join(missing)}."
            )
    if missing_errors:
        return [], missing_errors

    first_columns = list(datasets[0].dataframe.columns)
    if include_optional_numeric:
        selected = [column for column in first_columns if column in common_columns and column != TIMESTAMP_COLUMN]
    else:
        selected = [column for column in DEFAULT_REPLICATE_COLUMNS if column in common_columns]

    if not selected:
        return [], ["No matching numeric columns were found across the replicate files."]
    return selected, []


def prepare_timestamp_frame(dataset: ReplicateDataset, columns: list[str]) -> pd.DataFrame:
    dataframe = dataset.dataframe[[TIMESTAMP_COLUMN] + columns].copy()
    dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(dataframe[TIMESTAMP_COLUMN], format="mixed")
    dataframe = dataframe.dropna(subset=[TIMESTAMP_COLUMN])
    if dataframe[TIMESTAMP_COLUMN].duplicated().any():
        dataframe = dataframe.groupby(TIMESTAMP_COLUMN, as_index=False)[columns].mean(numeric_only=True)
    return dataframe


def align_replicates_by_timestamp(
    datasets: list[ReplicateDataset],
    columns: list[str],
) -> tuple[list[pd.DataFrame], AlignmentReport, list[str]]:
    errors: list[str] = []
    for dataset in datasets:
        if TIMESTAMP_COLUMN not in dataset.dataframe.columns:
            errors.append(f"{dataset.filename} does not contain TIMESTAMP for timestamp alignment.")
    if errors:
        return [], AlignmentReport("timestamp", 0, {}, [], {}), errors

    prepared = [prepare_timestamp_frame(dataset, columns) for dataset in datasets]
    common_timestamps = set(prepared[0][TIMESTAMP_COLUMN])
    for dataframe in prepared[1:]:
        common_timestamps &= set(dataframe[TIMESTAMP_COLUMN])

    common_index = pd.Index(sorted(common_timestamps), name=TIMESTAMP_COLUMN)
    aligned = [
        dataframe.set_index(TIMESTAMP_COLUMN).loc[common_index, columns].reset_index()
        for dataframe in prepared
    ]
    unmatched_rows = {
        dataset.filename: int(len(dataframe) - len(common_index))
        for dataset, dataframe in zip(datasets, prepared)
    }
    warnings = []
    if any(count > 0 for count in unmatched_rows.values()):
        warnings.append("Some rows were not present at common exact TIMESTAMP values and were excluded.")
    return aligned, AlignmentReport(
        "timestamp",
        len(common_index),
        unmatched_rows,
        warnings,
        {dataset.filename: len(dataframe) for dataset, dataframe in zip(datasets, prepared)},
    ), []


def align_replicates_by_row(
    datasets: list[ReplicateDataset],
    columns: list[str],
) -> tuple[list[pd.DataFrame], AlignmentReport, list[str]]:
    min_rows = min(len(dataset.dataframe) for dataset in datasets)
    aligned = [
        dataset.dataframe[columns].iloc[:min_rows].reset_index(drop=True)
        for dataset in datasets
    ]
    unmatched_rows = {
        dataset.filename: int(len(dataset.dataframe) - min_rows)
        for dataset in datasets
    }
    warnings = ["Measurement-sequence alignment was used. Confirm that row order follows the same protocol."]
    return aligned, AlignmentReport(
        "measurement_sequence",
        min_rows,
        unmatched_rows,
        warnings,
        {dataset.filename: len(dataset.dataframe) for dataset in datasets},
    ), []


def rowwise_replicate_statistics(
    datasets: list[ReplicateDataset],
    include_optional_numeric: bool = True,
    alignment_mode: str = "row",
) -> tuple[pd.DataFrame, AlignmentReport, list[str]]:
    columns, errors = matching_numeric_columns(datasets, include_optional_numeric)
    if errors:
        return pd.DataFrame(), AlignmentReport(alignment_mode, 0, {}, [], {}), errors

    if alignment_mode == "timestamp":
        aligned, report, errors = align_replicates_by_timestamp(datasets, columns)
    elif alignment_mode in ("row", "measurement_sequence"):
        aligned, report, errors = align_replicates_by_row(datasets, columns)
    else:
        return pd.DataFrame(), AlignmentReport(alignment_mode, 0, {}, [], {}), ["Alignment mode must be timestamp or row."]
    if errors:
        return pd.DataFrame(), report, errors

    result = pd.DataFrame(index=range(report.matched_rows))
    first_frame = datasets[0].dataframe.iloc[: report.matched_rows].reset_index(drop=True)
    if alignment_mode == "timestamp" and aligned:
        result[TIMESTAMP_COLUMN] = aligned[0][TIMESTAMP_COLUMN]
    elif TIMESTAMP_COLUMN in first_frame.columns:
        result[TIMESTAMP_COLUMN] = first_frame[TIMESTAMP_COLUMN]

    for column in first_frame.columns:
        if column == TIMESTAMP_COLUMN:
            continue
        if column in columns:
            values = pd.concat(
                [pd.to_numeric(dataframe[column], errors="coerce") for dataframe in aligned],
                axis=1,
            )
            result[column] = values.mean(axis=1)
            result[f"{column}_std"] = values.std(axis=1, ddof=1)
        elif column not in result.columns:
            result[column] = first_frame[column]
    return result, report, []


def summary_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in dataframe.columns:
        if column == TIMESTAMP_COLUMN:
            continue
        series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        rows.append(
            {
                "Variable": column,
                "Mean": float(series.mean()) if not series.empty else pd.NA,
                "Standard Deviation": float(series.std(ddof=1)) if len(series) > 1 else pd.NA,
                "Sample Count": int(series.count()),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_xlsx_bytes(dataframe: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return output.getvalue()


def build_replicate_zip(files: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for filename, file_bytes in files:
            archive.writestr(filename, file_bytes)
    return output.getvalue()
