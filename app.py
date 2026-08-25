from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from io import BytesIO, StringIO
import re

import pandas as pd
import streamlit as st

from dat_processor import (
    FINAL_COLUMNS,
    build_xlsx_zip,
    dat_output_filename,
    dat_to_xlsx_bytes,
    process_dat_uploads,
)
from time_splitter import (
    WorkbookAnalysis,
    analyze_workbook_dataframes,
    build_zip as build_split_zip,
    analyze_xlsx_upload,
    split_workbook,
)


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
PREVIEW_ROWS = 20
MAX_DAT_UPLOADS = 10
HHMM_PATTERN = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


@dataclass
class UploadedDataset:
    filename: str
    dataframe: pd.DataFrame
    encoding: str


@dataclass
class DatBatchOutput:
    filename: str
    files: list[tuple[str, bytes]]
    split_result: object


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


def format_timestamp(value) -> str:
    if value is None or pd.isna(value):
        return "No rows"
    return str(value)


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-size: 18px;
        }
        .stApp {
            background: #ffffff;
            color: #111827;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }
        h1 {
            font-size: 40px !important;
            line-height: 1.15 !important;
            color: #111827 !important;
        }
        h2, .stMarkdown h2 {
            font-size: 30px !important;
            line-height: 1.2 !important;
            color: #111827 !important;
        }
        h3, .stMarkdown h3 {
            font-size: 24px !important;
            line-height: 1.25 !important;
            color: #111827 !important;
        }
        p, li, label, .stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {
            font-size: 18px !important;
            color: #111827;
        }
        small, .stCaption {
            font-size: 15px !important;
        }
        .stButton > button, .stDownloadButton > button {
            font-size: 17px !important;
            font-weight: 650;
            border-radius: 6px;
            border: 1px solid #2563eb;
            background: #2563eb;
            color: #ffffff;
        }
        .stTextInput input, .stTimeInput input, .stSelectbox div[data-baseweb="select"] {
            font-size: 18px !important;
            color: #111827 !important;
            background: #ffffff !important;
        }
        [data-testid="stFileUploader"] {
            background: #f8fafc;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 1rem;
        }
        [data-testid="stDataFrame"] {
            font-size: 15px !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background: #ffffff;
        }
        div[data-testid="stAlert"] {
            font-size: 17px !important;
        }
        code {
            color: #064e3b;
            background: #ecfdf5;
            font-size: 0.92em;
        }
        .edt-card {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            margin: 1.2rem 0;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }
        .edt-card-title {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 0.35rem;
            color: #111827;
        }
        .edt-muted {
            color: #4b5563;
            font-size: 16px;
        }
        .edt-result {
            border-top: 1px solid #e5e7eb;
            padding-top: 1rem;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_hhmm_time(value: str) -> tuple[time | None, str | None]:
    stripped = value.strip()
    match = HHMM_PATTERN.match(stripped)
    if not match:
        return None, "Please enter time in HH:MM format."
    return time(int(match.group(1)), int(match.group(2))), None


def dat_split_analysis(processed_file):
    return analyze_workbook_dataframes(
        {
            "Cleaned_Data": processed_file.cleaned_dataframe,
            "Raw_Data": processed_file.raw_dataframe,
        },
        dat_output_filename(processed_file.filename),
    )


def build_dat_batch_output(processed_file, end_position1: time, end_position2: time):
    output_name = dat_output_filename(processed_file.filename)
    full_file = (output_name, dat_to_xlsx_bytes(processed_file))
    analysis = dat_split_analysis(processed_file)
    result, split_error = split_workbook(analysis, end_position1, end_position2)
    if split_error:
        return None, split_error
    if result is None:
        return None, f"{processed_file.filename} could not be split."
    return DatBatchOutput(processed_file.filename, [full_file] + result.output_files, result), None


def render_dat_to_xlsx_tool() -> None:
    st.subheader("DAT → XLSX & Time Split")
    st.write("Upload up to 10 DAT files. Each file will be converted and split into three positions.")
    uploaded_files = st.file_uploader(
        "Upload DAT files",
        type=["dat"],
        accept_multiple_files=True,
        key="dat_uploads",
    )

    if uploaded_files and len(uploaded_files) > MAX_DAT_UPLOADS:
        st.error("A maximum of 10 DAT files can be processed at one time.")
        return

    processed_files, errors = process_dat_uploads(uploaded_files)
    show_upload_errors(errors)

    if not processed_files:
        st.info("Upload 1 to 10 DAT files to configure cut times and process the batch.")
        return

    st.markdown("## Files to Process")
    st.write("Enter cut times in 24-hour `HH:MM` format for each file.")
    configured_files = []
    validation_errors: list[str] = []
    batch_signature_parts = []

    for index, processed_file in enumerate(processed_files):
        dat_key = f"{index}_{processed_file.filename}_{processed_file.data_row_count}"
        st.markdown('<div class="edt-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="edt-card-title">{processed_file.filename}</div>',
            unsafe_allow_html=True,
        )

        range_text = "Data range unavailable"
        if "TIMESTAMP" in processed_file.cleaned_dataframe.columns:
            timestamp_values = pd.to_datetime(
                processed_file.cleaned_dataframe["TIMESTAMP"],
                format="mixed",
            )
            range_text = (
                f"{format_timestamp(timestamp_values.iloc[0])} → "
                f"{format_timestamp(timestamp_values.iloc[-1])}"
            )

        st.markdown(
            f"""
            <div class="edt-muted">
            {processed_file.data_row_count:,} observations ·
            {processed_file.original_column_count:,} original columns ·
            encoding: {processed_file.encoding}<br>
            Data range: {range_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        input_col1, input_col2 = st.columns(2)
        end_position1_text = input_col1.text_input(
            "End of Position 1",
            value="",
            placeholder="HH:MM",
            key=f"dat_end_position1_text_{dat_key}",
        )
        input_col1.caption("HH:MM")
        end_position2_text = input_col2.text_input(
            "End of Position 2",
            value="",
            placeholder="HH:MM",
            key=f"dat_end_position2_text_{dat_key}",
        )
        input_col2.caption("HH:MM")

        end_position1 = None
        end_position2 = None
        batch_signature_parts.append(
            (
                processed_file.filename,
                processed_file.data_row_count,
                end_position1_text.strip(),
                end_position2_text.strip(),
            )
        )
        if end_position1_text or end_position2_text:
            end_position1, error1 = parse_hhmm_time(end_position1_text)
            end_position2, error2 = parse_hhmm_time(end_position2_text)
            if error1:
                validation_errors.append(f"{processed_file.filename}: End of Position 1 - {error1}")
                st.error("End of Position 1: Please enter time in HH:MM format.")
            if error2:
                validation_errors.append(f"{processed_file.filename}: End of Position 2 - {error2}")
                st.error("End of Position 2: Please enter time in HH:MM format.")
        else:
            validation_errors.append(f"{processed_file.filename}: enter both cut times.")
            st.warning("Enter both cut times before batch processing.")

        if end_position1 is not None and end_position2 is not None:
            configured_files.append((processed_file, end_position1, end_position2))

        with st.expander("Processing details", expanded=False):
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

        with st.expander("Preview cleaned data", expanded=False):
            st.dataframe(
                processed_file.cleaned_dataframe.loc[:, list(FINAL_COLUMNS)].head(PREVIEW_ROWS),
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Batch Processing")
    if validation_errors:
        st.info("Enter valid HH:MM cut times for every uploaded DAT file before processing.")

    batch_signature = tuple(batch_signature_parts)
    process_clicked = st.button(
        "Process All Files",
        type="primary",
        disabled=bool(validation_errors) or len(configured_files) != len(processed_files),
    )

    if process_clicked:
        batch_outputs: list[DatBatchOutput] = []
        batch_errors: list[str] = []
        for processed_file, end_position1, end_position2 in configured_files:
            batch_output, batch_error = build_dat_batch_output(
                processed_file,
                end_position1,
                end_position2,
            )
            if batch_error:
                batch_errors.append(f"{processed_file.filename}: {batch_error}")
            elif batch_output:
                batch_outputs.append(batch_output)

        st.session_state["dat_batch_outputs"] = batch_outputs
        st.session_state["dat_batch_errors"] = batch_errors
        st.session_state["dat_batch_signature"] = batch_signature

    batch_errors = st.session_state.get("dat_batch_errors", [])
    batch_outputs = st.session_state.get("dat_batch_outputs", [])
    previous_signature = st.session_state.get("dat_batch_signature")
    show_upload_errors(batch_errors)

    if batch_outputs and previous_signature != batch_signature:
        st.info("Click Process All Files to regenerate outputs for the current files and cut times.")
    elif batch_outputs:
        st.markdown("## Results")
        all_generated_files = []
        for batch_output in batch_outputs:
            st.markdown(f"### {batch_output.filename}")
            render_split_result(batch_output.split_result)
            all_generated_files.extend(batch_output.files)

            button_cols = st.columns(4)
            labels = ["Full XLSX", "Position 1", "Position 2", "Position 3"]
            for button_col, label, output_file in zip(button_cols, labels, batch_output.files):
                output_name, output_bytes = output_file
                button_col.download_button(
                    label=label,
                    data=output_bytes,
                    file_name=output_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dat_result_download_{batch_output.filename}_{output_name}",
                )

        st.download_button(
            label="Download All",
            data=build_xlsx_zip(all_generated_files),
            file_name="processed_dat_files.zip",
            mime="application/zip",
            key="dat_download_all_batch_outputs",
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


def render_workbook_analysis(analysis: WorkbookAnalysis) -> None:
    primary_sheet = analysis.primary_sheet
    if primary_sheet is None:
        st.caption("TIMESTAMP parsing did not succeed.")
        return

    st.caption(
        f"{primary_sheet.row_count:,} rows | "
        f"TIMESTAMP column: {primary_sheet.timestamp_column} | "
        f"date: {primary_sheet.first_timestamp.date()}"
    )
    st.write(
        "Data range: "
        f"`{format_timestamp(primary_sheet.first_timestamp)}` → "
        f"`{format_timestamp(primary_sheet.last_timestamp)}`"
    )
    st.write("TIMESTAMP parsing succeeded.")


def render_split_result(result) -> None:
    st.markdown(
        f"Boundaries: `{result.boundary1.strftime('%H:%M:%S')}` and "
        f"`{result.boundary2.strftime('%H:%M:%S')}`"
    )

    for summary in result.position_summaries:
        st.write(
            f"**{summary.label}**: "
            f"{summary.rows:,} rows | "
            f"`{format_timestamp(summary.first_timestamp)}` → "
            f"`{format_timestamp(summary.last_timestamp)}`"
        )

    st.markdown("QC")
    for qc in result.qc_results:
        st.write(
            f"**{qc.sheet_name}**: original `{qc.original_rows:,}`, "
            f"position 1 `{qc.position_rows[0]:,}`, "
            f"position 2 `{qc.position_rows[1]:,}`, "
            f"position 3 `{qc.position_rows[2]:,}`, "
            f"total `{qc.total_rows:,}`"
        )
        if not qc.all_rows_accounted_for:
            st.error("Rows were lost during splitting.")
        if not qc.no_duplicate_assignments:
            st.error("Duplicated observations detected between positions.")
        if qc.passed:
            st.write("✓ All rows accounted for")
            st.write("✓ No duplicated observations")


def render_split_downloads(result, single_file_upload: bool) -> None:
    if single_file_upload:
        for output_name, output_bytes in result.output_files:
            st.download_button(
                label=f"Download {output_name}",
                data=output_bytes,
                file_name=output_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{result.filename}_{output_name}",
            )

        st.download_button(
            label="Download All Positions",
            data=build_split_zip(result.output_files),
            file_name=f"{result.filename.rsplit('.', 1)[0]}_positions.zip",
            mime="application/zip",
            key=f"download_all_{result.filename}",
        )


def render_split_by_time_tool() -> None:
    st.subheader("Split by Time")
    uploaded_files = st.file_uploader(
        "Upload one or more XLSX files",
        type=["xlsx"],
        accept_multiple_files=True,
        key="split_uploads",
    )

    if not uploaded_files:
        st.info("Upload XLSX time-series workbooks to split them into three positions.")
        return

    st.write(
        "The selected time represents the final minute included in the preceding "
        "position. For example, if End of Position 1 is 13:38, Position 1 includes "
        "observations through 13:38:59.xxx, and Position 2 begins at or after 13:39:00."
    )

    split_results = []
    valid_analysis_count = 0

    for index, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.getvalue()
        file_key = f"{index}_{uploaded_file.name}_{len(file_bytes)}"
        analysis = analyze_xlsx_upload(uploaded_file)

        st.markdown(f"### {uploaded_file.name}")
        render_workbook_analysis(analysis)
        show_upload_errors(analysis.errors)

        if not analysis.can_split or analysis.primary_sheet is None:
            continue

        valid_analysis_count += 1
        primary = analysis.primary_sheet
        default_position1 = primary.first_timestamp.time().replace(
            second=0,
            microsecond=0,
        )
        default_position2 = primary.last_timestamp.time().replace(
            second=0,
            microsecond=0,
        )

        input_col1, input_col2 = st.columns(2)
        end_position1 = input_col1.time_input(
            "End of Position 1",
            value=default_position1,
            step=timedelta(minutes=1),
            key=f"end_position1_{file_key}",
        )
        end_position2 = input_col2.time_input(
            "End of Position 2",
            value=default_position2,
            step=timedelta(minutes=1),
            key=f"end_position2_{file_key}",
        )

        result_key = f"split_result_{file_key}"
        error_key = f"split_error_{file_key}"
        settings_key = f"split_settings_{file_key}"
        selected_settings = (end_position1, end_position2)
        if st.button("Split File", key=f"split_button_{file_key}"):
            result, error = split_workbook(analysis, end_position1, end_position2)
            st.session_state[result_key] = result
            st.session_state[error_key] = error
            st.session_state[settings_key] = selected_settings

        split_error = st.session_state.get(error_key)
        split_result = st.session_state.get(result_key)
        split_settings = st.session_state.get(settings_key)

        if split_settings is not None and split_settings != selected_settings:
            st.info("Click Split File to run the split with the currently selected times.")
        elif split_error:
            st.error(split_error)
        elif split_result:
            render_split_result(split_result)
            render_split_downloads(split_result, single_file_upload=len(uploaded_files) == 1)
            split_results.append(split_result)

    if len(uploaded_files) > 1 and valid_analysis_count > 0 and len(split_results) == valid_analysis_count:
        all_output_files = [
            output_file
            for result in split_results
            for output_file in result.output_files
        ]
        st.download_button(
            label="Download all split files as ZIP",
            data=build_split_zip(all_output_files),
            file_name="split_files.zip",
            mime="application/zip",
            key="download_all_split_files",
        )


def main() -> None:
    st.set_page_config(
        page_title="Experimental Data Toolkit",
        layout="wide",
    )

    st.title("Experimental Data Toolkit")
    st.caption("A lightweight web tool for routine experimental data processing.")

    dat_tab, merge_tab, split_tab = st.tabs(["DAT → XLSX", "Merge CSV", "Split by Time"])
    with dat_tab:
        render_dat_to_xlsx_tool()
    with merge_tab:
        render_merge_csv_tool()
    with split_tab:
        render_split_by_time_tool()


if __name__ == "__main__":
    main()
