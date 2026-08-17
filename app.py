from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO

import pandas as pd
import streamlit as st

from dat_processor import (
    FINAL_COLUMNS,
    build_xlsx_zip,
    dat_output_filename,
    dat_to_xlsx_bytes,
    process_dat_uploads,
)


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
PREVIEW_ROWS = 20


@dataclass
class UploadedDataset:
    filename: str
    dataframe: pd.DataFrame
    encoding: str


def read_csv_upload(uploaded_file) -> tuple[UploadedDataset | None, str | None]:
    """Read a Streamlit upload without mutating the original uploaded object."""
    file_bytes = uploaded_file.getvalue()

    for encoding in CSV_ENCODINGS:
        try:
            text = file_bytes.decode(encoding)
            dataframe = pd.read_csv(StringIO(text))
            return UploadedDataset(uploaded_file.name, dataframe, encoding), None
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, f"{uploaded_file.name} is empty or does not contain CSV data."
        except pd.errors.ParserError as exc:
            return None, f"{uploaded_file.name} could not be parsed as CSV: {exc}"
        except Exception as exc:  # Streamlit should show a useful error, not a traceback.
            return None, f"{uploaded_file.name} could not be read: {exc}"

    return None, (
        f"{uploaded_file.name} could not be read with supported encodings: "
        f"{', '.join(CSV_ENCODINGS)}."
    )


def load_uploaded_csvs(uploaded_files) -> tuple[list[UploadedDataset], list[str]]:
    datasets: list[UploadedDataset] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files or []:
        dataset, error = read_csv_upload(uploaded_file)
        if error:
            errors.append(error)
        elif dataset:
            datasets.append(dataset)

    return datasets, errors


def dataframe_to_xlsx_bytes(dataframe: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Data")
    return output.getvalue()


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def show_dataset_preview(dataset: UploadedDataset) -> None:
    st.markdown(f"**{dataset.filename}**")
    st.caption(
        f"{len(dataset.dataframe):,} rows | "
        f"{len(dataset.dataframe.columns):,} columns | "
        f"encoding: {dataset.encoding}"
    )
    st.dataframe(dataset.dataframe.head(PREVIEW_ROWS), use_container_width=True)


def show_upload_errors(errors: list[str]) -> None:
    for error in errors:
        st.error(error)


def render_dat_to_xlsx_tool() -> None:
    st.subheader("DAT → XLSX")
    uploaded_files = st.file_uploader(
        "Upload one or more DAT files",
        type=["dat"],
        accept_multiple_files=True,
        key="dat_uploads",
    )

    processed_files, errors = process_dat_uploads(uploaded_files)
    show_upload_errors(errors)

    if not processed_files:
        st.info("Upload DAT files to preview, clean, and convert them.")
        return

    st.markdown("### Uploaded files")
    converted_files: list[tuple[str, bytes]] = []

    for processed_file in processed_files:
        st.markdown(f"**{processed_file.filename}**")
        st.caption(
            f"{processed_file.data_row_count:,} data rows | "
            f"{processed_file.original_column_count:,} original columns | "
            f"encoding: {processed_file.encoding}"
        )

        with st.expander("Processing summary", expanded=True):
            st.markdown("Detected and renamed columns")
            mapping_lines = [
                f"`{original}` → `{clean}`"
                for original, clean in processed_file.renamed_columns
            ]
            st.write(", ".join(mapping_lines))

            if processed_file.removed_columns:
                st.markdown("Removed columns")
                st.write(", ".join(f"`{column}`" for column in processed_file.removed_columns))
            else:
                st.markdown("Removed columns")
                st.write("None of the standard removable columns were present.")

        st.dataframe(
            processed_file.cleaned_dataframe.loc[:, list(FINAL_COLUMNS)].head(PREVIEW_ROWS),
            use_container_width=True,
        )

        output_name = dat_output_filename(processed_file.filename)
        output_bytes = dat_to_xlsx_bytes(processed_file)
        converted_files.append((output_name, output_bytes))
        st.download_button(
            label=f"Download {output_name}",
            data=output_bytes,
            file_name=output_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if len(converted_files) > 1:
        zip_bytes = build_xlsx_zip(converted_files)
        st.download_button(
            label="Download processed DAT files as ZIP",
            data=zip_bytes,
            file_name="processed_dat_files.zip",
            mime="application/zip",
        )


def validate_identical_columns(datasets: list[UploadedDataset]) -> str | None:
    if not datasets:
        return None

    expected_columns = list(datasets[0].dataframe.columns)
    expected_file = datasets[0].filename

    for dataset in datasets[1:]:
        current_columns = list(dataset.dataframe.columns)
        if current_columns != expected_columns:
            return (
                f"Column mismatch in {dataset.filename}. "
                f"Expected the same columns and order as {expected_file}: "
                f"{expected_columns}. Found: {current_columns}."
            )

    return None


def merge_datasets(
    datasets: list[UploadedDataset],
    require_identical_columns: bool,
    add_source_file: bool,
) -> tuple[pd.DataFrame | None, str | None]:
    if len(datasets) < 2:
        return None, "Upload at least two readable CSV files to merge."

    if require_identical_columns:
        error = validate_identical_columns(datasets)
        if error:
            return None, error

    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        frame = dataset.dataframe.copy()
        if add_source_file:
            frame["_source_file"] = dataset.filename
        frames.append(frame)

    return pd.concat(frames, axis=0, ignore_index=True, sort=False), None


def render_merge_csv_tool() -> None:
    st.subheader("Merge CSV Files")
    uploaded_files = st.file_uploader(
        "Upload two or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="merge_uploads",
    )

    datasets, errors = load_uploaded_csvs(uploaded_files)
    show_upload_errors(errors)

    if not datasets:
        st.info("Upload CSV files to preview and merge them.")
        return

    st.markdown("### Uploaded files")
    for dataset in datasets:
        show_dataset_preview(dataset)

    merge_mode = st.radio(
        "Merge mode",
        options=(
            "Require identical columns",
            "Keep all columns",
        ),
        help=(
            "Identical-column mode requires matching column names and order. "
            "Keep-all-columns mode preserves the union of columns and leaves "
            "missing values blank/NaN."
        ),
    )
    add_source_file = st.checkbox("Add source filename column")

    require_identical = merge_mode == "Require identical columns"
    merged_dataframe, merge_error = merge_datasets(
        datasets,
        require_identical_columns=require_identical,
        add_source_file=add_source_file,
    )

    if merge_error:
        st.error(merge_error)
        return

    if merged_dataframe is None:
        return

    st.markdown("### Merged result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Files", f"{len(datasets):,}")
    col2.metric("Rows", f"{len(merged_dataframe):,}")
    col3.metric("Columns", f"{len(merged_dataframe.columns):,}")
    st.dataframe(merged_dataframe.head(PREVIEW_ROWS), use_container_width=True)

    csv_bytes = dataframe_to_csv_bytes(merged_dataframe)
    xlsx_bytes = dataframe_to_xlsx_bytes(merged_dataframe)

    col_csv, col_xlsx = st.columns(2)
    col_csv.download_button(
        label="Download merged CSV",
        data=csv_bytes,
        file_name="merged_data.csv",
        mime="text/csv",
    )
    col_xlsx.download_button(
        label="Download merged XLSX",
        data=xlsx_bytes,
        file_name="merged_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    st.set_page_config(
        page_title="Experimental Data Toolkit",
        layout="wide",
    )

    st.title("Experimental Data Toolkit")
    st.caption("A lightweight web tool for routine experimental data processing.")

    dat_tab, merge_tab = st.tabs(["DAT → XLSX", "Merge CSV"])
    with dat_tab:
        render_dat_to_xlsx_tool()
    with merge_tab:
        render_merge_csv_tool()


if __name__ == "__main__":
    main()
