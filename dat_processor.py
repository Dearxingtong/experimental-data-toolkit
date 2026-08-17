from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


DAT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

COLUMN_MAPPINGS: tuple[tuple[str, str], ...] = (
    ("TIMESTAMP", "TIMESTAMP"),
    ("A_U_ms", "U1"),
    ("A_V_ms", "V1"),
    ("A_W_ms", "W1"),
    ("A_SonicTemp_C", "Temp1"),
    ("B_U_ms", "U2"),
    ("B_V_ms", "V2"),
    ("B_W_ms", "W2"),
    ("B_SonicTemp_C", "Temp2"),
    ("C_U_ms", "U3"),
    ("C_V_ms", "V3"),
    ("C_W_ms", "W3"),
    ("C_SonicTemp_C", "Temp3"),
)

REQUIRED_COLUMNS = tuple(original for original, _clean in COLUMN_MAPPINGS)
FINAL_COLUMNS = tuple(clean for _original, clean in COLUMN_MAPPINGS)
REMOVED_COLUMNS = (
    "RECORD",
    "A_SensorStatus",
    "B_SensorStatus",
    "C_SensorStatus",
)


@dataclass
class ProcessedDatFile:
    filename: str
    raw_dataframe: pd.DataFrame
    cleaned_dataframe: pd.DataFrame
    encoding: str
    renamed_columns: list[tuple[str, str]]
    removed_columns: list[str]

    @property
    def data_row_count(self) -> int:
        return len(self.raw_dataframe)

    @property
    def original_column_count(self) -> int:
        return len(self.raw_dataframe.columns)


def dat_output_filename(filename: str) -> str:
    return f"{Path(filename).stem}.xlsx"


def read_dat_text(file_bytes: bytes) -> tuple[str | None, str | None]:
    for encoding in DAT_ENCODINGS:
        try:
            return file_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    return None, None


def parse_dat_table(text: str) -> pd.DataFrame:
    """Use row 2 as headers and rows 5+ as observations."""
    dataframe = pd.read_csv(
        StringIO(text),
        header=0,
        skiprows=[0, 2, 3],
        sep=",",
    )
    dataframe.columns = normalize_column_names(dataframe.columns)
    return dataframe


def normalize_column_name(column_name: object) -> str:
    """Remove extra DAT quote formatting from a source header."""
    normalized = str(column_name).strip().lstrip("\ufeff").strip()
    normalized = normalized.strip('"').strip()

    while normalized.startswith('"') or normalized.endswith('"'):
        normalized = normalized.strip('"').strip()

    return normalized


def normalize_column_names(column_names) -> list[str]:
    return [normalize_column_name(column_name) for column_name in column_names]


def validate_required_columns(dataframe: pd.DataFrame, filename: str) -> str | None:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if not missing_columns:
        return None

    return (
        f"{filename} cannot be processed because the following required columns "
        f"are missing: {', '.join(missing_columns)}"
    )


def clean_dat_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    selected = dataframe.loc[:, list(REQUIRED_COLUMNS)].copy()
    cleaned = selected.rename(columns=dict(COLUMN_MAPPINGS)).loc[:, list(FINAL_COLUMNS)]
    cleaned["TIMESTAMP"] = parse_timestamp_column(cleaned["TIMESTAMP"])
    return cleaned


def parse_timestamp_column(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series)


def detect_removed_columns(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in REMOVED_COLUMNS if column in dataframe.columns]


def process_dat_bytes(file_bytes: bytes, filename: str) -> tuple[ProcessedDatFile | None, str | None]:
    text, encoding = read_dat_text(file_bytes)
    if text is None or encoding is None:
        return None, (
            f"{filename} could not be read with supported encodings: "
            f"{', '.join(DAT_ENCODINGS)}."
        )

    try:
        raw_dataframe = parse_dat_table(text)
    except pd.errors.EmptyDataError:
        return None, f"{filename} is empty or does not contain DAT tabular data."
    except pd.errors.ParserError as exc:
        return None, f"{filename} could not be parsed as DAT data: {exc}"
    except Exception as exc:
        return None, f"{filename} could not be read: {exc}"

    validation_error = validate_required_columns(raw_dataframe, filename)
    if validation_error:
        return None, validation_error

    cleaned_dataframe = clean_dat_dataframe(raw_dataframe)
    processed_file = ProcessedDatFile(
        filename=filename,
        raw_dataframe=raw_dataframe,
        cleaned_dataframe=cleaned_dataframe,
        encoding=encoding,
        renamed_columns=list(COLUMN_MAPPINGS),
        removed_columns=detect_removed_columns(raw_dataframe),
    )
    return processed_file, None


def process_dat_upload(uploaded_file) -> tuple[ProcessedDatFile | None, str | None]:
    return process_dat_bytes(uploaded_file.getvalue(), uploaded_file.name)


def process_dat_uploads(uploaded_files) -> tuple[list[ProcessedDatFile], list[str]]:
    processed_files: list[ProcessedDatFile] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files or []:
        processed_file, error = process_dat_upload(uploaded_file)
        if error:
            errors.append(error)
        elif processed_file:
            processed_files.append(processed_file)

    return processed_files, errors


def dat_to_xlsx_bytes(processed_file: ProcessedDatFile) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        processed_file.cleaned_dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned_Data",
        )
        processed_file.raw_dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Raw_Data",
        )
    return output.getvalue()


def build_xlsx_zip(files: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for filename, file_bytes in files:
            archive.writestr(filename, file_bytes)
    return output.getvalue()
