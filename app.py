from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from io import BytesIO, StringIO

import pandas as pd
import streamlit as st

from dat_batch import (
    all_workbook_filename,
    normalize_case_number,
    position_workbook_filename,
    format_case_date,
)
from dat_processor import (
    add_transformed_velocity_columns,
    build_xlsx_zip,
    dat_to_xlsx_bytes,
    process_dat_upload,
    velocity_transform_description,
)
from time_splitter import (
    PositionRange,
    WorkbookAnalysis,
    analyze_workbook_dataframes,
    build_zip as build_split_zip,
    analyze_xlsx_upload,
    build_time_range_from_parts,
    parse_hhmm_time,
    split_workbook_by_position_ranges,
    split_workbook,
)


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
PREVIEW_ROWS = 20
MIN_DAT_POSITIONS = 2
MAX_DAT_POSITIONS = 6


@dataclass
class UploadedDataset:
    filename: str
    dataframe: pd.DataFrame
    encoding: str


@dataclass
class DatasetDownload:
    label: str
    rows: int
    xlsx_filename: str
    xlsx_bytes: bytes
    csv_filename: str
    csv_bytes: bytes


@dataclass
class CompletedDatCase:
    original_filename: str
    case_number: str
    case_date: date
    split_enabled: bool
    position_count: int
    position_ranges: list[tuple[str, str, str]]
    all_rows: int
    raw_rows: int
    files: list[tuple[str, bytes]]
    position_summaries: tuple
    qc_results: list
    velocity_transform_notes: tuple[str, ...] = tuple()
    datasets: list[DatasetDownload] = field(default_factory=list)


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


def xlsx_filename_to_csv(filename: str) -> str:
    if filename.lower().endswith(".xlsx"):
        return f"{filename[:-5]}.csv"
    return f"{filename}.csv"


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
        .edt-next-step {
            margin-top: 1.75rem;
            padding-top: 1.25rem;
            border-top: 2px solid #dbeafe;
            font-size: 28px;
            font-weight: 800;
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def dat_split_analysis(processed_file):
    return analyze_workbook_dataframes(
        {
            "Cleaned_Data": processed_file.cleaned_dataframe,
            "Raw_Data": processed_file.raw_dataframe,
        },
        processed_file.filename,
    )


def transform_dat_position_workbook(
    position_number: int,
    workbook: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    transformed_workbook = {
        sheet_name: dataframe.copy()
        for sheet_name, dataframe in workbook.items()
    }
    if "Cleaned_Data" in transformed_workbook:
        transformed_workbook["Cleaned_Data"] = add_transformed_velocity_columns(
            transformed_workbook["Cleaned_Data"],
            position_number,
        )
    return transformed_workbook


def dat_velocity_transform_notes(position_count: int) -> tuple[str, ...]:
    notes: list[str] = []
    if position_count >= 1:
        first_group_end = min(3, position_count)
        notes.append(
            f"P01-P{first_group_end:02d}: {velocity_transform_description(1)}"
        )
    if position_count >= 4:
        notes.append(
            f"P04-P{position_count:02d}: {velocity_transform_description(4)}"
        )
    return tuple(notes)


def build_completed_dat_case(
    processed_file,
    case_number: str,
    case_date: date,
    split_enabled: bool,
    position_ranges: list[PositionRange],
) -> tuple[CompletedDatCase | None, str | None]:
    all_filename = all_workbook_filename(case_number, case_date)
    all_xlsx_bytes = dat_to_xlsx_bytes(processed_file)
    all_csv_filename = xlsx_filename_to_csv(all_filename)
    all_csv_bytes = dataframe_to_csv_bytes(processed_file.cleaned_dataframe)
    files = [
        (all_filename, all_xlsx_bytes),
        (all_csv_filename, all_csv_bytes),
    ]
    datasets = [
        DatasetDownload(
            label="Full Dataset",
            rows=processed_file.data_row_count,
            xlsx_filename=all_filename,
            xlsx_bytes=all_xlsx_bytes,
            csv_filename=all_csv_filename,
            csv_bytes=all_csv_bytes,
        )
    ]
    position_summaries = tuple()
    qc_results = []

    if split_enabled:
        analysis = analyze_workbook_dataframes(
            {
                "Cleaned_Data": processed_file.cleaned_dataframe,
                "Raw_Data": processed_file.raw_dataframe,
            },
            all_filename,
        )
        position_filenames = [
            position_workbook_filename(case_number, index + 1, case_date)
            for index in range(len(position_ranges))
        ]
        result, split_error = split_workbook_by_position_ranges(
            analysis,
            position_ranges,
            position_filenames,
            workbook_transformer=transform_dat_position_workbook,
        )
        if split_error:
            return None, split_error
        if result is None:
            return None, f"{processed_file.filename} could not be split."
        for summary, (xlsx_filename, xlsx_bytes), workbook in zip(
            result.position_summaries,
            result.output_files,
            result.output_workbooks,
        ):
            csv_filename = xlsx_filename_to_csv(xlsx_filename)
            cleaned_data = workbook["Cleaned_Data"]
            csv_bytes = dataframe_to_csv_bytes(cleaned_data)
            files.extend(
                [
                    (xlsx_filename, xlsx_bytes),
                    (csv_filename, csv_bytes),
                ]
            )
            datasets.append(
                DatasetDownload(
                    label=summary.label,
                    rows=summary.rows,
                    xlsx_filename=xlsx_filename,
                    xlsx_bytes=xlsx_bytes,
                    csv_filename=csv_filename,
                    csv_bytes=csv_bytes,
                )
            )
        position_summaries = result.position_summaries
        qc_results = result.qc_results

    all_preserved = (
        len(processed_file.cleaned_dataframe) == processed_file.data_row_count
        and len(processed_file.raw_dataframe) == processed_file.data_row_count
    )
    if not all_preserved:
        return None, "Full Dataset preservation check failed."

    return (
        CompletedDatCase(
            original_filename=processed_file.filename,
            case_number=case_number,
            case_date=case_date,
            split_enabled=split_enabled,
            position_count=len(position_ranges) if split_enabled else 0,
            position_ranges=[
                (position_range.label, position_range.start_time.strftime("%H:%M"), position_range.end_time.strftime("%H:%M"))
                for position_range in position_ranges
            ],
            all_rows=processed_file.data_row_count,
            raw_rows=len(processed_file.raw_dataframe),
            files=files,
            position_summaries=position_summaries,
            qc_results=qc_results,
            velocity_transform_notes=(
                dat_velocity_transform_notes(len(position_ranges))
                if split_enabled
                else tuple()
            ),
            datasets=datasets,
        ),
        None,
    )


def init_dat_batch_state() -> None:
    st.session_state.setdefault("dat_completed_cases", [])
    st.session_state.setdefault("dat_current_result", None)
    st.session_state.setdefault("dat_batch_finished", False)
    st.session_state.setdefault("dat_uploader_version", 0)


def reset_current_dat_form() -> None:
    st.session_state["dat_current_result"] = None
    st.session_state["dat_uploader_version"] = st.session_state.get("dat_uploader_version", 0) + 1


def start_new_dat_batch() -> None:
    st.session_state["dat_completed_cases"] = []
    st.session_state["dat_current_result"] = None
    st.session_state["dat_batch_finished"] = False
    st.session_state["dat_uploader_version"] = st.session_state.get("dat_uploader_version", 0) + 1


def case_key(case_number: str, case_date: date) -> tuple[str, str]:
    return case_number, format_case_date(case_date)


def case_already_used(case_number: str, case_date: date) -> bool:
    key = case_key(case_number, case_date)
    for completed_case in st.session_state.get("dat_completed_cases", []):
        if case_key(completed_case.case_number, completed_case.case_date) == key:
            return True
    return False


def detected_case_date(processed_file) -> date:
    timestamps = pd.to_datetime(processed_file.cleaned_dataframe["TIMESTAMP"], format="mixed")
    return timestamps.iloc[0].date()


def render_completed_case_downloads(completed_case: CompletedDatCase, prefix: str) -> None:
    st.markdown("### Downloads")
    header_cols = st.columns([2.1, 1, 1.4, 1.4])
    header_cols[0].markdown("**Dataset**")
    header_cols[1].markdown("**Rows**")
    header_cols[2].markdown("**XLSX**")
    header_cols[3].markdown("**CSV**")

    for dataset in completed_case.datasets:
        xlsx_label = (
            "Download Full XLSX"
            if dataset.label == "Full Dataset"
            else f"Download {dataset.label} XLSX"
        )
        csv_label = (
            "Download Full CSV"
            if dataset.label == "Full Dataset"
            else f"Download {dataset.label} CSV"
        )
        row_cols = st.columns([2.1, 1, 1.4, 1.4])
        row_cols[0].write(dataset.label)
        row_cols[1].write(f"{dataset.rows:,}")
        row_cols[2].download_button(
            label=xlsx_label,
            data=dataset.xlsx_bytes,
            file_name=dataset.xlsx_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{prefix}_{completed_case.case_number}_{format_case_date(completed_case.case_date)}_{dataset.xlsx_filename}",
        )
        row_cols[3].download_button(
            label=csv_label,
            data=dataset.csv_bytes,
            file_name=dataset.csv_filename,
            mime="text/csv",
            key=f"{prefix}_{completed_case.case_number}_{format_case_date(completed_case.case_date)}_{dataset.csv_filename}",
        )

    st.download_button(
        label="Download All Files (ZIP)",
        data=build_xlsx_zip(completed_case.files),
        file_name=f"C{completed_case.case_number}_{format_case_date(completed_case.case_date)}_files.zip",
        mime="application/zip",
        key=f"{prefix}_{completed_case.case_number}_{format_case_date(completed_case.case_date)}_case_zip",
    )


def render_completed_case_summary(completed_case: CompletedDatCase) -> None:
    st.markdown(
        f"## C{completed_case.case_number} — {format_case_date(completed_case.case_date)}"
    )
    st.write("✓ Processing completed successfully.")
    st.write(f"Full Dataset: `{completed_case.all_rows:,}` rows")

    for summary in completed_case.position_summaries:
        st.write(f"{summary.label}: `{summary.rows:,}` rows")

    st.write("✓ Full Dataset preserved")
    if completed_case.split_enabled:
        if all(qc.no_duplicate_assignments for qc in completed_case.qc_results):
            st.write("✓ All Position ranges processed")
            st.write("✓ No overlapping observations")
        else:
            st.error("Position QC failed.")
        for note in completed_case.velocity_transform_notes:
            st.write(note)

    render_completed_case_downloads(completed_case, "current_download")


def render_completed_files_section() -> None:
    completed_cases = st.session_state.get("dat_completed_cases", [])
    if not completed_cases:
        return

    with st.expander(f"Completed Files ({len(completed_cases)})", expanded=False):
        summary_rows = []
        for completed_case in completed_cases:
            outputs = "Full"
            if completed_case.split_enabled:
                outputs = f"Full + P01-P{completed_case.position_count:02d}"
            summary_rows.append(
                {
                    "Case": f"C{completed_case.case_number}",
                    "Date": format_case_date(completed_case.case_date),
                    "Split": (
                        f"{completed_case.position_count} Positions"
                        if completed_case.split_enabled
                        else "No"
                    ),
                    "Outputs": outputs,
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        for completed_case in completed_cases:
            st.markdown(
                f"**C{completed_case.case_number} — {format_case_date(completed_case.case_date)}**"
            )
            render_completed_case_downloads(completed_case, "completed_download")


def render_batch_complete() -> None:
    completed_cases = st.session_state.get("dat_completed_cases", [])
    st.markdown("# Batch Complete")
    st.write(f"{len(completed_cases)} DAT files processed")
    render_completed_files_section()
    all_files = [
        output_file
        for completed_case in completed_cases
        for output_file in completed_case.files
    ]
    if all_files:
        st.download_button(
            label="Download All Files (ZIP)",
            data=build_xlsx_zip(all_files),
            file_name="processed_dat_files.zip",
            mime="application/zip",
            key="finished_batch_zip",
        )
    if st.button("Start New Batch", type="primary"):
        start_new_dat_batch()
        st.rerun()


def collect_position_ranges(position_count: int, key_prefix: str) -> tuple[list[PositionRange], list[str]]:
    position_ranges: list[PositionRange] = []
    errors: list[str] = []
    st.markdown("### Position Time Ranges")
    header_cols = st.columns([1.1, 1, 0.15, 1, 0.35, 1, 0.15, 1])
    header_cols[0].markdown("**Position**")
    header_cols[1].markdown("**Start HH**")
    header_cols[2].markdown("&nbsp;")
    header_cols[3].markdown("**Start MM**")
    header_cols[5].markdown("**End HH**")
    header_cols[6].markdown("&nbsp;")
    header_cols[7].markdown("**End MM**")

    for position_number in range(1, position_count + 1):
        label = f"P{position_number:02d}"
        row_cols = st.columns([1.1, 1, 0.15, 1, 0.35, 1, 0.15, 1])
        row_cols[0].write(label)
        start_hour_text = row_cols[1].text_input(
            f"{label} Start Hour",
            value="",
            placeholder="HH",
            label_visibility="collapsed",
            key=f"{key_prefix}_start_hour_{position_number}",
        )
        row_cols[2].write(":")
        start_minute_text = row_cols[3].text_input(
            f"{label} Start Minute",
            value="",
            placeholder="MM",
            label_visibility="collapsed",
            key=f"{key_prefix}_start_minute_{position_number}",
        )
        end_hour_text = row_cols[5].text_input(
            f"{label} End Hour",
            value="",
            placeholder="HH",
            label_visibility="collapsed",
            key=f"{key_prefix}_end_hour_{position_number}",
        )
        row_cols[6].write(":")
        end_minute_text = row_cols[7].text_input(
            f"{label} End Minute",
            value="",
            placeholder="MM",
            label_visibility="collapsed",
            key=f"{key_prefix}_end_minute_{position_number}",
        )

        start_time, end_time, range_errors = build_time_range_from_parts(
            label,
            start_hour_text,
            start_minute_text,
            end_hour_text,
            end_minute_text,
        )
        errors.extend(range_errors)
        if start_time is not None and end_time is not None:
            position_ranges.append(PositionRange(label, start_time, end_time))

    return position_ranges, errors


def validate_case_inputs(
    case_no_input: str,
    case_date_input,
    split_enabled: bool,
    position_count: int,
    position_ranges: list[PositionRange],
    position_errors: list[str],
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    case_number, case_error = normalize_case_number(case_no_input)
    if case_error:
        errors.append(case_error)

    if case_date_input is None:
        errors.append("Please select a Case Date.")

    if split_enabled:
        if position_count < MIN_DAT_POSITIONS or position_count > MAX_DAT_POSITIONS:
            errors.append(
                f"Split Position count must be between {MIN_DAT_POSITIONS} and {MAX_DAT_POSITIONS}."
            )
        if position_errors:
            errors.extend(position_errors)
        if len(position_ranges) != position_count:
            errors.append("Please enter Start Time and End Time for all Positions.")

    if case_number and case_date_input is not None and case_already_used(case_number, case_date_input):
        errors.append("This Case No and Case Date combination has already been used in the current batch.")

    return case_number, errors


def render_dat_to_xlsx_tool() -> None:
    init_dat_batch_state()
    completed_count = len(st.session_state["dat_completed_cases"])

    st.subheader("DAT → XLSX")
    st.write(f"Files processed in current batch: {completed_count}")

    if st.session_state.get("dat_batch_finished"):
        render_batch_complete()
        return

    current_result = st.session_state.get("dat_current_result")
    if current_result:
        render_completed_case_summary(current_result)
        st.markdown('<div class="edt-next-step">Process another file?</div>', unsafe_allow_html=True)
        next_col, finish_col = st.columns(2)
        if next_col.button("Process Next File"):
            reset_current_dat_form()
            st.rerun()
        if finish_col.button("Finish Batch", type="primary"):
            st.session_state["dat_batch_finished"] = True
            st.rerun()
        return

    render_completed_files_section()

    st.markdown("## Current File")
    uploaded_files = st.file_uploader(
        "Upload one DAT file",
        type=["dat"],
        accept_multiple_files=False,
        key=f"dat_upload_{st.session_state['dat_uploader_version']}",
    )

    if uploaded_files is None:
        st.info("Upload one DAT file to begin. Completed files remain stored in this batch session.")
        return

    processed_file, parse_error = process_dat_upload(uploaded_files)
    if parse_error:
        st.error(parse_error)
        return

    timestamps = pd.to_datetime(processed_file.cleaned_dataframe["TIMESTAMP"], format="mixed")
    with st.container(border=True):
        st.markdown("### Current File")
        st.write(f"Original filename: `{processed_file.filename}`")
        st.write(f"Observations: `{processed_file.data_row_count:,}`")
        st.write(
            "Detected time range: "
            f"`{format_timestamp(timestamps.iloc[0])}` → `{format_timestamp(timestamps.iloc[-1])}`"
        )

        st.markdown("### Case Information")
        case_col, date_col, split_col = st.columns([1, 1, 1])
        case_no_input = case_col.text_input(
            "Case No",
            value="",
            placeholder="3",
            key=f"dat_case_no_{st.session_state['dat_uploader_version']}",
        )
        case_date_input = date_col.date_input(
            "Case Date",
            value=detected_case_date(processed_file),
            key=f"dat_case_date_{st.session_state['dat_uploader_version']}",
        )
        split_choice = split_col.selectbox(
            "Split this file?",
            options=("No", "Yes"),
            index=None,
            placeholder="Choose",
            key=f"dat_split_choice_{st.session_state['dat_uploader_version']}",
        )

        position_ranges: list[PositionRange] = []
        position_errors: list[str] = []
        split_enabled = split_choice == "Yes"
        if split_enabled:
            position_count = st.number_input(
                "Number of Positions",
                min_value=MIN_DAT_POSITIONS,
                max_value=MAX_DAT_POSITIONS,
                value=3,
                step=1,
                key=f"dat_position_count_{st.session_state['dat_uploader_version']}",
            )
            position_ranges, position_errors = collect_position_ranges(
                int(position_count),
                f"dat_position_{st.session_state['dat_uploader_version']}",
            )
        else:
            position_count = 0

        if st.button("Process File", type="primary"):
            if split_choice is None:
                form_errors = ["Please choose whether this file should be split."]
                case_number = None
            else:
                case_number, form_errors = validate_case_inputs(
                    case_no_input,
                    case_date_input,
                    split_enabled,
                    int(position_count),
                    position_ranges,
                    position_errors,
                )

            for form_error in form_errors:
                st.error(form_error)

            if not form_errors and case_number is not None:
                completed_case = None
                process_error = None
                with st.status(f"Processing C{case_number}...", expanded=True) as status:
                    try:
                        status.write("Reading DAT file...")
                        status.write("Creating cleaned dataset...")
                        status.write("Generating Full Dataset...")
                        if split_enabled:
                            status.write("Generating Position datasets...")
                            status.write("Applying velocity transformations...")
                        status.write("Creating XLSX and CSV outputs...")
                        status.write("Running QC...")
                        completed_case, process_error = build_completed_dat_case(
                            processed_file,
                            case_number,
                            case_date_input,
                            split_enabled,
                            position_ranges,
                        )
                    except Exception as exc:
                        process_error = str(exc)

                    if process_error:
                        status.update(
                            label=f"Processing failed: {process_error}",
                            state="error",
                            expanded=True,
                        )
                    elif completed_case:
                        status.write("Preparing downloads...")
                        status.update(
                            label="✓ Processing completed successfully.",
                            state="complete",
                            expanded=False,
                        )

                if process_error:
                    st.error(process_error)
                elif completed_case:
                    st.session_state["dat_completed_cases"].append(completed_case)
                    st.session_state["dat_current_result"] = completed_case
                    st.rerun()


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
    inject_global_styles()

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
