from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from io import BytesIO, StringIO

import pandas as pd
import streamlit as st

from analysis_processor import (
    VIEW_AZIMUTH,
    VIEW_ELEVATION,
    VIEW_ROLL,
    VectorRecord,
    build_vector_records,
    close_figure,
    compute_mean_vectors,
    duplicate_position_height_errors,
    export_figure_eps,
    export_figure_png,
    plot_3d_air_vectors,
    plot_interactive_air_vectors,
    validate_analysis_csv,
    validate_heights,
    vector_records_to_dataframe,
)
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
from replicate_processor import (
    ReplicateDataset,
    build_replicate_zip,
    dataframe_to_csv_bytes as replicate_dataframe_to_csv_bytes,
    dataframe_to_xlsx_bytes as replicate_dataframe_to_xlsx_bytes,
    rowwise_replicate_statistics,
    summary_statistics as replicate_summary_statistics,
)
from project_io import (
    ProjectLoadError,
    build_project_state,
    decode_project_state,
    load_project_archive,
    project_filename,
    save_project_archive,
)
from time_splitter import (
    PositionRange,
    WorkbookAnalysis,
    analyze_workbook_dataframes,
    available_dates_from_timestamps,
    build_zip as build_split_zip,
    analyze_xlsx_upload,
    build_position_range_from_duration,
    duration_options,
    format_calculated_end,
    format_duration_option,
    format_hhmm,
    generate_minute_time_options_for_date,
    parse_hhmm_time,
    position_number_from_range,
    split_workbook_by_position_ranges,
    split_workbook,
)
from vertical_profile_processor import (
    DATASET_TYPES,
    DEFAULT_ROOM_HEIGHT_FT,
    build_vertical_profile_records_from_sources,
    build_replicate_vertical_profile_records,
    build_vertical_profile_records,
    build_vertical_profile_zip,
    default_temperature_columns,
    export_vertical_profile_eps,
    export_vertical_profile_png,
    numeric_columns,
    plot_vertical_profiles,
    records_have_plot_standard_deviation,
    vertical_profile_records_to_dataframe,
    vertical_profile_records_to_unified_dataframe,
    vertical_profile_summary_csv,
    vertical_profile_summary_xlsx,
)


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
PREVIEW_ROWS = 20
MIN_DAT_SPLITS = 1
MAX_DAT_PHYSICAL_POSITIONS = 6
DEFAULT_DAT_SPLITS = 3
VERTICAL_PROFILE_TEMPERATURE_LABELS = (
    "Height 1 Temperature Column",
    "Height 2 Temperature Column",
    "Height 3 Temperature Column",
)
VERTICAL_PROFILE_CONCENTRATION_LABELS = (
    "Height 1 Concentration Column",
    "Height 2 Concentration Column",
    "Height 3 Concentration Column",
)
ANALYSIS_CAMERA_DEFAULTS = {
    "azimuth": float(VIEW_AZIMUTH),
    "elevation": float(VIEW_ELEVATION),
    "roll": float(VIEW_ROLL),
}
ANALYSIS_CAMERA_RANGES = {
    "azimuth": (-180.0, 180.0),
    "elevation": (-90.0, 90.0),
    "roll": (-180.0, 180.0),
}


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
    position_ranges: list[tuple[str, str, int | None, str]]
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


def dat_velocity_transform_notes(position_numbers: list[int]) -> tuple[str, ...]:
    notes: list[str] = []
    if any(position_number <= 3 for position_number in position_numbers):
        notes.append(f"P01-P03: {velocity_transform_description(1)}")
    if any(position_number >= 4 for position_number in position_numbers):
        notes.append(f"P04-P06: {velocity_transform_description(4)}")
    return tuple(notes)


def split_range_output_date(position_range: PositionRange, fallback_date: date) -> date:
    if position_range.start_datetime is not None:
        return position_range.start_datetime.date()
    return fallback_date


def split_output_requires_start_time(
    position_ranges: list[PositionRange],
    position_range: PositionRange,
    fallback_date: date,
) -> bool:
    current_position = position_number_from_range(position_range)
    current_date = split_range_output_date(position_range, fallback_date)
    matches = [
        candidate
        for candidate in position_ranges
        if position_number_from_range(candidate) == current_position
        and split_range_output_date(candidate, fallback_date) == current_date
    ]
    return len(matches) > 1


def split_position_workbook_filename(
    case_number: str,
    fallback_date: date,
    position_ranges: list[PositionRange],
    position_range: PositionRange,
) -> str:
    start_datetime = position_range.start_datetime
    return position_workbook_filename(
        case_number,
        position_number_from_range(position_range),
        split_range_output_date(position_range, fallback_date),
        start_datetime=start_datetime,
        include_start_time=split_output_requires_start_time(position_ranges, position_range, fallback_date),
    )


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
            split_position_workbook_filename(case_number, case_date, position_ranges, position_range)
            for position_range in position_ranges
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
                (
                    position_range.label,
                    position_range.start_datetime.strftime("%Y-%m-%d %H:%M")
                    if position_range.start_datetime
                    else position_range.start_time.strftime("%H:%M"),
                    position_range.duration_minutes,
                    position_range.end_datetime.strftime("%Y-%m-%d %H:%M")
                    if position_range.end_datetime
                    else position_range.end_time.strftime("%H:%M"),
                )
                for position_range in position_ranges
            ],
            all_rows=processed_file.data_row_count,
            raw_rows=len(processed_file.raw_dataframe),
            files=files,
            position_summaries=position_summaries,
            qc_results=qc_results,
            velocity_transform_notes=(
                dat_velocity_transform_notes([position_number_from_range(position_range) for position_range in position_ranges])
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
            st.write("✓ Each split output contains unique observations")
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
                outputs = f"Full + {completed_case.position_count} split outputs"
            summary_rows.append(
                {
                    "Case": f"C{completed_case.case_number}",
                    "Date": format_case_date(completed_case.case_date),
                    "Split": (
                        f"{completed_case.position_count} Splits"
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


def collect_position_ranges(
    split_count: int,
    key_prefix: str,
    timestamps: pd.Series,
) -> tuple[list[PositionRange], list[str]]:
    position_ranges: list[PositionRange] = []
    errors: list[str] = []
    st.markdown("### Split Time Ranges")
    date_options = available_dates_from_timestamps(timestamps)
    duration_values = duration_options()
    if not date_options:
        return [], ["No Date options could be generated from the detected data range."]

    header_cols = st.columns([0.7, 1.1, 1.4, 1.4, 1.2, 1.4])
    header_cols[0].markdown("**Split**")
    header_cols[1].markdown("**Position**")
    header_cols[2].markdown("**Date**")
    header_cols[3].markdown("**Start Time**")
    header_cols[4].markdown("**Duration (min)**")
    header_cols[5].markdown("**Calculated End**")

    for split_number in range(1, split_count + 1):
        split_label = f"Split {split_number}"
        row_cols = st.columns([0.7, 1.1, 1.4, 1.4, 1.2, 1.4])
        row_cols[0].write(split_number)
        selected_position = row_cols[1].selectbox(
            f"{split_label} Position",
            options=list(range(1, MAX_DAT_PHYSICAL_POSITIONS + 1)),
            format_func=lambda value: f"P{value}",
            label_visibility="collapsed",
            key=f"{key_prefix}_position_{split_number}",
        )
        selected_date = row_cols[2].selectbox(
            f"{split_label} Date",
            options=date_options,
            label_visibility="collapsed",
            key=f"{key_prefix}_date_{split_number}",
        )
        start_options = generate_minute_time_options_for_date(timestamps, selected_date)
        if not start_options:
            errors.append(f"{split_label}: selected date is not represented in the uploaded data.")
            row_cols[3].write("No times")
            continue
        selected_start = row_cols[3].selectbox(
            f"{split_label} Start Time",
            options=start_options,
            format_func=format_hhmm,
            label_visibility="collapsed",
            key=f"{key_prefix}_start_time_{split_number}",
        )
        selected_duration = row_cols[4].selectbox(
            f"{split_label} Duration (min)",
            options=duration_values,
            format_func=format_duration_option,
            index=0,
            label_visibility="collapsed",
            key=f"{key_prefix}_duration_{split_number}",
        )
        label = f"P{selected_position:02d}"
        position_range, range_error = build_position_range_from_duration(
            label,
            selected_start,
            selected_duration,
            selected_date=selected_date,
            position_number=selected_position,
            split_number=split_number,
        )
        calculated_end = (
            format_calculated_end(position_range.start_datetime, position_range.end_datetime)
            if position_range is not None
            else "Select duration"
        )
        row_cols[5].write(calculated_end)
        if range_error:
            errors.append(range_error)
        elif position_range is not None:
            position_ranges.append(position_range)

    return position_ranges, errors


def validate_case_inputs(
    case_no_input: str,
    case_date_input,
    split_enabled: bool,
    split_count: int,
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
        if split_count < MIN_DAT_SPLITS:
            errors.append(f"Number of Splits must be at least {MIN_DAT_SPLITS}.")
        if position_errors:
            errors.extend(position_errors)
        if len(position_ranges) != split_count:
            errors.append("Please select Position, Date, Start Time, and Duration for all Splits.")

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
        case_col, split_col, date_col = st.columns([1, 1, 1])
        case_no_input = case_col.text_input(
            "Case No",
            value="",
            placeholder="3",
            key=f"dat_case_no_{st.session_state['dat_uploader_version']}",
        )
        split_choice = split_col.selectbox(
            "Split this file?",
            options=("No", "Yes"),
            index=None,
            placeholder="Choose",
            key=f"dat_split_choice_{st.session_state['dat_uploader_version']}",
        )
        split_enabled = split_choice == "Yes"
        case_date_input = date_col.date_input(
            "Case Date",
            value=detected_case_date(processed_file),
            disabled=split_enabled,
            help=(
                "Used for the Full Dataset output. Split outputs use each row's selected Date."
                if split_enabled
                else None
            ),
            key=f"dat_case_date_{st.session_state['dat_uploader_version']}",
        )
        if split_enabled:
            st.caption("Split output filenames use each split row's selected Date. The global Case Date remains for the Full Dataset.")

        position_ranges: list[PositionRange] = []
        position_errors: list[str] = []
        if split_enabled:
            split_count = st.number_input(
                "Number of Splits",
                min_value=MIN_DAT_SPLITS,
                value=DEFAULT_DAT_SPLITS,
                step=1,
                key=f"dat_split_count_{st.session_state['dat_uploader_version']}",
            )
            position_ranges, position_errors = collect_position_ranges(
                int(split_count),
                f"dat_position_{st.session_state['dat_uploader_version']}",
                timestamps,
            )
        else:
            split_count = 0

        if st.button("Process File", type="primary"):
            if split_choice is None:
                form_errors = ["Please choose whether this file should be split."]
                case_number = None
            else:
                case_number, form_errors = validate_case_inputs(
                    case_no_input,
                    case_date_input,
                    split_enabled,
                    int(split_count),
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
    processing_mode = st.radio(
        "Processing mode",
        options=("Standard Merge", "Replicate Mean & SD"),
        horizontal=True,
        key="merge_processing_mode",
    )
    if processing_mode == "Replicate Mean & SD":
        render_replicate_mean_sd_tool(show_back_button=False)
        return

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


def init_analysis_state() -> None:
    st.session_state.setdefault("analysis_vector_records", [])
    st.session_state.setdefault("analysis_awaiting_continue", False)
    st.session_state.setdefault("analysis_finished", False)
    st.session_state.setdefault("analysis_uploader_version", 0)
    st.session_state.setdefault("analysis_camera_pending", None)
    for parameter, default_value in ANALYSIS_CAMERA_DEFAULTS.items():
        st.session_state.setdefault(f"analysis_camera_{parameter}", default_value)
        st.session_state.setdefault(f"analysis_camera_{parameter}_slider", default_value)


def reset_analysis_form() -> None:
    st.session_state["analysis_awaiting_continue"] = False
    st.session_state["analysis_uploader_version"] = st.session_state.get(
        "analysis_uploader_version",
        0,
    ) + 1


def reset_analysis() -> None:
    st.session_state["analysis_vector_records"] = []
    st.session_state["analysis_awaiting_continue"] = False
    st.session_state["analysis_finished"] = False
    st.session_state["analysis_camera_pending"] = None
    st.session_state["analysis_uploader_version"] = st.session_state.get(
        "analysis_uploader_version",
        0,
    ) + 1


def clear_data_analysis_project_state() -> None:
    reset_analysis()
    reset_vertical_profile_analysis()
    st.session_state["analysis_method"] = None
    st.session_state["replicate_analysis_state"] = {}
    st.session_state["project_metadata"] = {}
    st.session_state["loaded_project_summary"] = None


def current_project_case_metadata() -> tuple[str | None, str | None]:
    metadata = st.session_state.get("project_metadata") or {}
    case_number = metadata.get("case_number")
    case_date = metadata.get("case_date")
    replicate_state = st.session_state.get("replicate_analysis_state") or {}
    if not case_number:
        case_number = replicate_state.get("case_number")
    if not case_date:
        case_date = replicate_state.get("case_date")
    return case_number, case_date


def build_current_project_state() -> dict:
    case_number, case_date = current_project_case_metadata()
    vector_settings = {
        "camera": {
            "azimuth": st.session_state.get("analysis_camera_azimuth", ANALYSIS_CAMERA_DEFAULTS["azimuth"]),
            "elevation": st.session_state.get("analysis_camera_elevation", ANALYSIS_CAMERA_DEFAULTS["elevation"]),
            "roll": st.session_state.get("analysis_camera_roll", ANALYSIS_CAMERA_DEFAULTS["roll"]),
        },
        "unit_outputs": ["si", "imperial"],
    }
    vertical_settings = {
        "selected_variables": st.session_state.get("vertical_profile_selected_variables", []),
        "profile_mode": "Raw Profiles",
        "show_error_bars": st.session_state.get("vertical_profile_show_error_bars", True),
        "contaminant_name": st.session_state.get("vertical_profile_contaminant_name", "Contaminant"),
        "concentration_unit": st.session_state.get("vertical_profile_concentration_unit", ""),
    }
    return build_project_state(
        case_number=case_number,
        case_date=case_date,
        vector_records=st.session_state.get("analysis_vector_records", []),
        vector_settings=vector_settings,
        vertical_records=st.session_state.get("vertical_profile_records", []),
        vertical_settings=vertical_settings,
        replicate_analysis=st.session_state.get("replicate_analysis_state", {}),
        ui_preferences={"analysis_method": st.session_state.get("analysis_method")},
    )


def restore_project_to_session(project_state: dict) -> dict:
    decoded = decode_project_state(project_state)
    init_analysis_state()
    init_vertical_profile_state()

    st.session_state["project_metadata"] = decoded["metadata"]
    st.session_state["analysis_vector_records"] = decoded["vector_records"]
    st.session_state["analysis_finished"] = bool(decoded["vector_records"])
    st.session_state["analysis_awaiting_continue"] = False
    camera = decoded["vector_settings"].get("camera", {})
    set_analysis_camera(
        camera.get("azimuth", ANALYSIS_CAMERA_DEFAULTS["azimuth"]),
        camera.get("elevation", ANALYSIS_CAMERA_DEFAULTS["elevation"]),
        camera.get("roll", ANALYSIS_CAMERA_DEFAULTS["roll"]),
    )
    for parameter in ("azimuth", "elevation", "roll"):
        st.session_state[f"analysis_camera_{parameter}_slider"] = st.session_state[f"analysis_camera_{parameter}"]

    st.session_state["vertical_profile_records"] = decoded["vertical_records"]
    st.session_state["vertical_profile_finished"] = bool(decoded["vertical_records"])
    st.session_state["vertical_profile_awaiting_continue"] = False
    vertical_settings = decoded["vertical_settings"]
    if vertical_settings:
        st.session_state["vertical_profile_selected_variables"] = vertical_settings.get("selected_variables", [])
        st.session_state["vertical_profile_profile_mode"] = "Raw Profiles"
        st.session_state["vertical_profile_show_error_bars"] = vertical_settings.get("show_error_bars", True)
        st.session_state["vertical_profile_contaminant_name"] = vertical_settings.get("contaminant_name", "Contaminant")
        st.session_state["vertical_profile_concentration_unit"] = vertical_settings.get("concentration_unit", "")

    st.session_state["replicate_analysis_state"] = decoded["replicate_analysis"]
    preferred_method = decoded["ui_preferences"].get("analysis_method")
    if preferred_method in {"3d_vector_plot", "vertical_profile_plot"}:
        st.session_state["analysis_method"] = preferred_method
    elif decoded["vector_records"]:
        st.session_state["analysis_method"] = "3d_vector_plot"
    elif decoded["vertical_records"]:
        st.session_state["analysis_method"] = "vertical_profile_plot"
    else:
        st.session_state["analysis_method"] = None

    analyses = decoded["metadata"].get("analyses", [])
    st.session_state["loaded_project_summary"] = {
        "project": decoded["metadata"].get("case_number") or "Untitled",
        "date": decoded["metadata"].get("case_date"),
        "analyses": analyses,
        "format_version": decoded["metadata"].get("format_version"),
    }
    return st.session_state["loaded_project_summary"]


def clamp_analysis_camera_value(parameter: str, value: float) -> float:
    lower_bound, upper_bound = ANALYSIS_CAMERA_RANGES[parameter]
    return min(max(float(value), lower_bound), upper_bound)


def set_analysis_camera(azimuth: float, elevation: float, roll: float) -> None:
    for parameter, value in {
        "azimuth": azimuth,
        "elevation": elevation,
        "roll": roll,
    }.items():
        camera_value = clamp_analysis_camera_value(parameter, value)
        st.session_state[f"analysis_camera_{parameter}"] = camera_value


def queue_analysis_camera_update(azimuth: float, elevation: float, roll: float) -> None:
    st.session_state["analysis_camera_pending"] = {
        "azimuth": clamp_analysis_camera_value("azimuth", azimuth),
        "elevation": clamp_analysis_camera_value("elevation", elevation),
        "roll": clamp_analysis_camera_value("roll", roll),
    }


def apply_pending_analysis_camera_update() -> None:
    pending_camera = st.session_state.pop("analysis_camera_pending", None)
    if pending_camera is None:
        return
    set_analysis_camera(
        pending_camera["azimuth"],
        pending_camera["elevation"],
        pending_camera["roll"],
    )
    for parameter in ANALYSIS_CAMERA_DEFAULTS:
        st.session_state[f"analysis_camera_{parameter}_slider"] = st.session_state[
            f"analysis_camera_{parameter}"
        ]


def sync_analysis_camera_from_widget(parameter: str, widget_key: str) -> None:
    camera_value = clamp_analysis_camera_value(parameter, st.session_state[widget_key])
    st.session_state[f"analysis_camera_{parameter}"] = camera_value


def render_camera_parameter_control(parameter: str, label: str, help_text: str) -> float:
    slider_key = f"analysis_camera_{parameter}_slider"
    lower_bound, upper_bound = ANALYSIS_CAMERA_RANGES[parameter]
    st.slider(
        label,
        min_value=lower_bound,
        max_value=upper_bound,
        step=1.0,
        help=help_text,
        key=slider_key,
        on_change=sync_analysis_camera_from_widget,
        args=(parameter, slider_key),
    )
    return float(st.session_state[f"analysis_camera_{parameter}"])


def render_analysis_camera_controls() -> tuple[float, float, float]:
    apply_pending_analysis_camera_update()
    st.markdown("### Camera Controls")
    st.caption(
        "Azimuth = horizontal rotation. Elevation = vertical viewing angle. "
        "Roll = rotation around the viewing axis."
    )
    azimuth = render_camera_parameter_control("azimuth", "Azimuth (deg)", "Horizontal rotation")
    elevation = render_camera_parameter_control("elevation", "Elevation (deg)", "Vertical viewing angle")
    roll = render_camera_parameter_control("roll", "Roll (deg)", "Rotation around the viewing axis")

    preset_cols = st.columns(6)
    if preset_cols[0].button("Apply View"):
        set_analysis_camera(azimuth, elevation, roll)
        st.rerun()
    if preset_cols[1].button("Reset View"):
        queue_analysis_camera_update(
            ANALYSIS_CAMERA_DEFAULTS["azimuth"],
            ANALYSIS_CAMERA_DEFAULTS["elevation"],
            ANALYSIS_CAMERA_DEFAULTS["roll"],
        )
        st.rerun()
    if preset_cols[2].button("Perspective"):
        queue_analysis_camera_update(
            ANALYSIS_CAMERA_DEFAULTS["azimuth"],
            ANALYSIS_CAMERA_DEFAULTS["elevation"],
            ANALYSIS_CAMERA_DEFAULTS["roll"],
        )
        st.rerun()
    if preset_cols[3].button("Front"):
        queue_analysis_camera_update(0.0, 0.0, 0.0)
        st.rerun()
    if preset_cols[4].button("Side"):
        queue_analysis_camera_update(90.0, 0.0, 0.0)
        st.rerun()
    if preset_cols[5].button("Top"):
        queue_analysis_camera_update(-90.0, 90.0, 0.0)
        st.rerun()

    azimuth = float(st.session_state["analysis_camera_azimuth"])
    elevation = float(st.session_state["analysis_camera_elevation"])
    roll = float(st.session_state["analysis_camera_roll"])
    st.markdown(
        f"**Current camera:** Azimuth `{azimuth:g}°`, "
        f"Elevation `{elevation:g}°`, Roll `{roll:g}°`"
    )
    return azimuth, elevation, roll


def render_analysis_summary(vector_records: list[VectorRecord]) -> None:
    if not vector_records:
        return
    st.markdown("### Processed airflow vectors")
    st.dataframe(
        vector_records_to_dataframe(vector_records, "si"),
        use_container_width=True,
        hide_index=True,
    )


def render_analysis_outputs(vector_records: list[VectorRecord]) -> None:
    if not vector_records:
        st.info("Add at least one position data file before generating the final figure.")
        return

    st.markdown("### Combined 3D vector plots")
    st.caption(
        "Arrow direction represents mean airflow direction. Arrow length is proportional to "
        "velocity magnitude and scaled for visualization. Colors indicate velocity magnitude."
    )
    azimuth = float(st.session_state["analysis_camera_azimuth"])
    elevation = float(st.session_state["analysis_camera_elevation"])
    roll = float(st.session_state["analysis_camera_roll"])

    st.markdown("### Interactive 3D View")
    st.caption(
        "Drag or zoom this Plotly preview for exploration. Exported images use the manual "
        "Azimuth, Elevation, and Roll settings below."
    )
    try:
        interactive_fig = plot_interactive_air_vectors(vector_records, "si", azimuth, elevation, roll)
        st.plotly_chart(
            interactive_fig,
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
        )
    except ImportError:
        st.info("Install plotly to enable the interactive 3D preview. Static exports still use matplotlib.")

    azimuth, elevation, roll = render_analysis_camera_controls()

    try:
        si_fig = plot_3d_air_vectors(vector_records, "si", azimuth, elevation, roll)
        imperial_fig = plot_3d_air_vectors(vector_records, "imperial", azimuth, elevation, roll)
    except ImportError:
        st.error("matplotlib is required for Data Analysis plots. Run pip install -r requirements.txt.")
        return

    si_summary_dataframe = vector_records_to_dataframe(vector_records, "si")
    imperial_summary_dataframe = vector_records_to_dataframe(vector_records, "imperial")
    si_eps_bytes = export_figure_eps(si_fig)
    si_png_bytes = export_figure_png(si_fig)
    imperial_eps_bytes = export_figure_eps(imperial_fig)
    imperial_png_bytes = export_figure_png(imperial_fig)
    si_summary_csv_bytes = dataframe_to_csv_bytes(si_summary_dataframe)
    si_summary_xlsx_bytes = dataframe_to_xlsx_bytes(si_summary_dataframe)
    imperial_summary_csv_bytes = dataframe_to_csv_bytes(imperial_summary_dataframe)
    imperial_summary_xlsx_bytes = dataframe_to_xlsx_bytes(imperial_summary_dataframe)
    analysis_zip_bytes = build_split_zip(
        [
            ("3D_Air_Velocity_SI.eps", si_eps_bytes),
            ("3D_Air_Velocity_SI.png", si_png_bytes),
            ("3D_Air_Velocity_Imperial.eps", imperial_eps_bytes),
            ("3D_Air_Velocity_Imperial.png", imperial_png_bytes),
            ("Air_Velocity_Summary_SI.csv", si_summary_csv_bytes),
            ("Air_Velocity_Summary_SI.xlsx", si_summary_xlsx_bytes),
            ("Air_Velocity_Summary_Imperial.csv", imperial_summary_csv_bytes),
            ("Air_Velocity_Summary_Imperial.xlsx", imperial_summary_xlsx_bytes),
        ]
    )

    st.markdown("#### SI 3D Plot")
    st.pyplot(si_fig, clear_figure=False)
    si_eps_col, si_png_col, si_csv_col, si_xlsx_col = st.columns(4)
    si_eps_col.download_button(
        label="Download SI EPS",
        data=si_eps_bytes,
        file_name="3D_Air_Velocity_SI.eps",
        mime="application/postscript",
        key="analysis_download_si_eps",
    )
    si_png_col.download_button(
        label="Download SI PNG",
        data=si_png_bytes,
        file_name="3D_Air_Velocity_SI.png",
        mime="image/png",
        key="analysis_download_si_png",
    )
    si_csv_col.download_button(
        label="Download SI Summary CSV",
        data=si_summary_csv_bytes,
        file_name="Air_Velocity_Summary_SI.csv",
        mime="text/csv",
        key="analysis_download_si_summary_csv",
    )
    si_xlsx_col.download_button(
        label="Download SI Summary XLSX",
        data=si_summary_xlsx_bytes,
        file_name="Air_Velocity_Summary_SI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="analysis_download_si_summary_xlsx",
    )
    st.markdown("##### SI Summary")
    st.dataframe(si_summary_dataframe, use_container_width=True, hide_index=True)

    st.markdown("#### Imperial 3D Plot")
    st.pyplot(imperial_fig, clear_figure=False)
    imperial_eps_col, imperial_png_col, imperial_csv_col, imperial_xlsx_col = st.columns(4)
    imperial_eps_col.download_button(
        label="Download Imperial EPS",
        data=imperial_eps_bytes,
        file_name="3D_Air_Velocity_Imperial.eps",
        mime="application/postscript",
        key="analysis_download_imperial_eps",
    )
    imperial_png_col.download_button(
        label="Download Imperial PNG",
        data=imperial_png_bytes,
        file_name="3D_Air_Velocity_Imperial.png",
        mime="image/png",
        key="analysis_download_imperial_png",
    )
    imperial_csv_col.download_button(
        label="Download Imperial Summary CSV",
        data=imperial_summary_csv_bytes,
        file_name="Air_Velocity_Summary_Imperial.csv",
        mime="text/csv",
        key="analysis_download_imperial_summary_csv",
    )
    imperial_xlsx_col.download_button(
        label="Download Imperial Summary XLSX",
        data=imperial_summary_xlsx_bytes,
        file_name="Air_Velocity_Summary_Imperial.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="analysis_download_imperial_summary_xlsx",
    )

    st.markdown("##### Imperial Summary")
    st.dataframe(imperial_summary_dataframe, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download All Analysis Files (ZIP)",
        data=analysis_zip_bytes,
        file_name="Air_Velocity_Analysis_Files.zip",
        mime="application/zip",
        key="analysis_download_all_outputs",
    )
    close_figure(si_fig)
    close_figure(imperial_fig)


def render_3d_vector_plot_tool() -> None:
    init_analysis_state()
    vector_records: list[VectorRecord] = st.session_state["analysis_vector_records"]

    if st.button("Back to Data Analysis", key="analysis_back_from_3d"):
        st.session_state["analysis_method"] = None
        st.rerun()

    st.subheader("3D Vector Plot")
    st.write("Build a combined 3D airflow vector map from position CSV files.")
    st.caption("Room dimensions: x = 8 ft, y = 10 ft, z = 8.75 ft.")

    if st.button("Reset analysis"):
        reset_analysis()
        st.rerun()

    render_analysis_summary(vector_records)

    if st.session_state.get("analysis_finished"):
        render_analysis_outputs(vector_records)
        return

    if st.session_state.get("analysis_awaiting_continue"):
        st.markdown("### Do you want to add another position data file?")
        yes_col, no_col = st.columns(2)
        if yes_col.button("Yes"):
            reset_analysis_form()
            st.rerun()
        if no_col.button("No", type="primary"):
            st.session_state["analysis_finished"] = True
            st.rerun()
        return

    st.markdown("### Add one position dataset")
    input_col, height_col, upload_col = st.columns([1, 1.5, 2])
    position_number = input_col.selectbox(
        "Position Number",
        options=list(range(1, 7)),
        format_func=lambda value: f"Position {value}",
        key=f"analysis_position_{st.session_state['analysis_uploader_version']}",
    )
    height1 = height_col.number_input(
        "Height 1 (ft)",
        min_value=0.0,
        max_value=8.75,
        value=2.0,
        step=0.25,
        key=f"analysis_height1_{st.session_state['analysis_uploader_version']}",
    )
    height2 = height_col.number_input(
        "Height 2 (ft)",
        min_value=0.0,
        max_value=8.75,
        value=4.0,
        step=0.25,
        key=f"analysis_height2_{st.session_state['analysis_uploader_version']}",
    )
    height3 = height_col.number_input(
        "Height 3 (ft)",
        min_value=0.0,
        max_value=8.75,
        value=6.0,
        step=0.25,
        key=f"analysis_height3_{st.session_state['analysis_uploader_version']}",
    )
    uploaded_file = upload_col.file_uploader(
        "Upload airflow CSV file",
        type=["csv"],
        accept_multiple_files=False,
        key=f"analysis_upload_{st.session_state['analysis_uploader_version']}",
    )

    if st.button("Process Position File", type="primary"):
        if uploaded_file is None:
            st.error("Please upload one airflow CSV file.")
            return

        dataset, read_error = read_csv_upload(uploaded_file)
        if read_error:
            st.error(read_error)
            return
        if dataset is None:
            st.error(f"{uploaded_file.name} could not be read.")
            return

        heights = [height1, height2, height3]
        height_error = validate_heights(heights)
        csv_error = validate_analysis_csv(dataset.dataframe, dataset.filename)
        for error in (height_error, csv_error):
            if error:
                st.error(error)
        if height_error or csv_error:
            return

        mean_vectors = compute_mean_vectors(dataset.dataframe)
        new_records = build_vector_records(
            position_number,
            heights,
            mean_vectors,
            dataset.filename,
        )
        duplicate_errors = duplicate_position_height_errors(vector_records, new_records)
        if duplicate_errors:
            for error in duplicate_errors:
                st.error(error)
            return

        st.session_state["analysis_vector_records"].extend(new_records)
        st.session_state["analysis_awaiting_continue"] = True
        st.success(f"Processed {dataset.filename} for Position {position_number}.")
        st.rerun()


def init_vertical_profile_state() -> None:
    st.session_state.setdefault("vertical_profile_records", [])
    st.session_state.setdefault("vertical_profile_awaiting_continue", False)
    st.session_state.setdefault("vertical_profile_finished", False)
    st.session_state.setdefault("vertical_profile_uploader_version", 0)


def reset_vertical_profile_analysis() -> None:
    st.session_state["vertical_profile_records"] = []
    st.session_state["vertical_profile_awaiting_continue"] = False
    st.session_state["vertical_profile_finished"] = False
    st.session_state["vertical_profile_uploader_version"] = st.session_state.get(
        "vertical_profile_uploader_version",
        0,
    ) + 1


def reset_vertical_profile_form() -> None:
    st.session_state["vertical_profile_awaiting_continue"] = False
    st.session_state["vertical_profile_uploader_version"] = st.session_state.get(
        "vertical_profile_uploader_version",
        0,
    ) + 1


def selected_vertical_profile_variables() -> list[str]:
    variables = []
    if st.checkbox("Air Velocity", value=True, key="vertical_profile_velocity_selected"):
        variables.append("Air Velocity")
    if st.checkbox("Temperature", value=True, key="vertical_profile_temperature_selected"):
        variables.append("Temperature")
    if st.checkbox(
        "Contaminant Concentration",
        value=False,
        key="vertical_profile_contaminant_selected",
    ):
        variables.append("Contaminant Concentration")
    return variables


def render_normalization_methods() -> None:
    return None


def render_vertical_profile_outputs(
    records,
    selected_variables: list[str],
    profile_mode: str,
    contaminant_name: str,
    concentration_unit: str,
    show_error_bars: bool = True,
) -> None:
    if not records:
        st.info("Add at least one vertical-profile dataset before generating the figure.")
        return

    st.markdown("### Vertical Profile Figure")
    try:
        si_fig = plot_vertical_profiles(
            records,
            selected_variables,
            profile_mode,
            contaminant_name,
            concentration_unit,
            show_error_bars,
            unit_system="si",
        )
        ip_fig = plot_vertical_profiles(
            records,
            selected_variables,
            profile_mode,
            contaminant_name,
            concentration_unit,
            show_error_bars,
            unit_system="ip",
        )
    except ImportError:
        st.error("matplotlib is required for Vertical Profile plots. Run pip install -r requirements.txt.")
        return

    if show_error_bars and not records_have_plot_standard_deviation(records, profile_mode):
        st.info("Standard deviation columns were not found, so the profile was plotted without error bars.")

    st.markdown("#### SI Units")
    st.pyplot(si_fig, clear_figure=False)
    st.markdown("#### IP Units")
    st.pyplot(ip_fig, clear_figure=False)
    si_eps_bytes = export_vertical_profile_eps(si_fig)
    si_png_bytes = export_vertical_profile_png(si_fig)
    ip_eps_bytes = export_vertical_profile_eps(ip_fig)
    ip_png_bytes = export_vertical_profile_png(ip_fig)
    summary_csv = vertical_profile_summary_csv(records, selected_variables)
    summary_xlsx = vertical_profile_summary_xlsx(records, selected_variables)
    zip_bytes = build_vertical_profile_zip(
        [
            ("Vertical_Profiles_SI.eps", si_eps_bytes),
            ("Vertical_Profiles_SI.png", si_png_bytes),
            ("Vertical_Profiles_IP.eps", ip_eps_bytes),
            ("Vertical_Profiles_IP.png", ip_png_bytes),
            ("Vertical_Profile_Summary.csv", summary_csv),
            ("Vertical_Profile_Summary.xlsx", summary_xlsx),
        ]
    )
    close_figure(si_fig)
    close_figure(ip_fig)

    si_eps_col, si_png_col, ip_eps_col, ip_png_col = st.columns(4)
    si_eps_col.download_button(
        "Download SI EPS",
        data=si_eps_bytes,
        file_name="Vertical_Profiles_SI.eps",
        mime="application/postscript",
        key="vertical_profile_download_si_eps",
    )
    si_png_col.download_button(
        "Download SI PNG",
        data=si_png_bytes,
        file_name="Vertical_Profiles_SI.png",
        mime="image/png",
        key="vertical_profile_download_si_png",
    )
    ip_eps_col.download_button(
        "Download IP EPS",
        data=ip_eps_bytes,
        file_name="Vertical_Profiles_IP.eps",
        mime="application/postscript",
        key="vertical_profile_download_ip_eps",
    )
    ip_png_col.download_button(
        "Download IP PNG",
        data=ip_png_bytes,
        file_name="Vertical_Profiles_IP.png",
        mime="image/png",
        key="vertical_profile_download_ip_png",
    )
    csv_col, xlsx_col = st.columns(2)
    csv_col.download_button(
        "Summary CSV",
        data=summary_csv,
        file_name="Vertical_Profile_Summary.csv",
        mime="text/csv",
        key="vertical_profile_download_csv",
    )
    xlsx_col.download_button(
        "Summary XLSX",
        data=summary_xlsx,
        file_name="Vertical_Profile_Summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="vertical_profile_download_xlsx",
    )
    st.download_button(
        "Download All Vertical Profile Files",
        data=zip_bytes,
        file_name="Vertical_Profile_Files.zip",
        mime="application/zip",
        key="vertical_profile_download_zip",
    )
    st.markdown("### Vertical Profile Summary")
    st.dataframe(
        vertical_profile_records_to_unified_dataframe(records, selected_variables),
        use_container_width=True,
        hide_index=True,
    )


def render_vertical_profile_tool() -> None:
    init_vertical_profile_state()
    records = st.session_state["vertical_profile_records"]

    if st.button("Back to Data Analysis", key="analysis_back_from_vertical_profile"):
        st.session_state["analysis_method"] = None
        st.rerun()

    st.subheader("Vertical Profile Plot")
    st.write("Plot vertical distributions of air velocity, temperature, and contaminant concentration.")

    if st.button("Reset Vertical Profile Analysis"):
        reset_vertical_profile_analysis()
        st.rerun()

    if records:
        st.markdown("### Processed vertical-profile points")
        processed_variables = st.session_state.get("vertical_profile_selected_variables")
        st.dataframe(
            vertical_profile_records_to_unified_dataframe(records, processed_variables),
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.get("vertical_profile_finished"):
        selected_variables = st.session_state.get("vertical_profile_selected_variables", ["Air Velocity"])
        profile_mode = "Raw Profiles"
        contaminant_name = st.session_state.get("vertical_profile_contaminant_name", "Contaminant")
        concentration_unit = st.session_state.get("vertical_profile_concentration_unit", "")
        render_vertical_profile_outputs(
            records,
            selected_variables,
            profile_mode,
            contaminant_name,
            concentration_unit,
            st.session_state.get("vertical_profile_show_error_bars", True),
        )
        return

    if st.session_state.get("vertical_profile_awaiting_continue"):
        selected_variables = st.session_state.get("vertical_profile_selected_variables", ["Air Velocity"])
        profile_mode = "Raw Profiles"
        contaminant_name = st.session_state.get("vertical_profile_contaminant_name", "Contaminant")
        concentration_unit = st.session_state.get("vertical_profile_concentration_unit", "")
        render_vertical_profile_outputs(
            records,
            selected_variables,
            profile_mode,
            contaminant_name,
            concentration_unit,
            st.session_state.get("vertical_profile_show_error_bars", True),
        )
        st.markdown("### Continue this case or start a new case?")
        yes_col, no_col = st.columns(2)
        if yes_col.button("Continue this case", key="vertical_profile_continue_case"):
            reset_vertical_profile_form()
            st.rerun()
        if no_col.button("Start new case analysis", type="primary", key="vertical_profile_start_new_case"):
            reset_vertical_profile_analysis()
            st.rerun()
        return

    dataset_type = st.radio("Dataset Type", DATASET_TYPES, horizontal=True)
    if dataset_type != "Experimental":
        st.warning("CFD input support is reserved for a future update.")

    profile_mode = "Raw Profiles"
    selected_variables = selected_vertical_profile_variables()
    show_error_bars = st.checkbox(
        "Show replicate SD error bars",
        value=True,
        key="vertical_profile_show_error_bars_input",
    )
    room_height_ft = DEFAULT_ROOM_HEIGHT_FT
    supply_velocity = None
    tin = tout = None
    cin = cout = None
    contaminant_name = st.session_state.get("vertical_profile_contaminant_name", "Contaminant")
    concentration_unit = st.session_state.get("vertical_profile_concentration_unit", "")
    if "Contaminant Concentration" in selected_variables:
        contaminant_name = st.text_input(
            "Contaminant Name",
            value=contaminant_name,
            placeholder="CO2, Tracer gas, PM2.5",
        )
        concentration_unit = st.text_input(
            "Concentration Unit",
            value=concentration_unit,
            placeholder="ppm, ppb, ug/m3, mg/m3",
        )

    st.markdown("### Add one position dataset")
    input_col, height_col = st.columns([1, 1.4])
    position_number = input_col.selectbox(
        "Position Number",
        options=list(range(1, 7)),
        format_func=lambda value: f"P{value}",
        key=f"vertical_profile_position_{st.session_state['vertical_profile_uploader_version']}",
    )
    heights = [
        height_col.number_input("Height 1 (ft)", min_value=0.0, value=2.0, step=0.25, key=f"vp_h1_{st.session_state['vertical_profile_uploader_version']}"),
        height_col.number_input("Height 2 (ft)", min_value=0.0, value=4.0, step=0.25, key=f"vp_h2_{st.session_state['vertical_profile_uploader_version']}"),
        height_col.number_input("Height 3 (ft)", min_value=0.0, value=6.0, step=0.25, key=f"vp_h3_{st.session_state['vertical_profile_uploader_version']}"),
    ]
    st.caption(
        "Each selected variable may use its own CSV file. Timestamp alignment between source files is not required."
    )
    use_replicate_profile = st.checkbox(
        "Use replicate files for this position",
        value=False,
        key=f"vertical_profile_use_replicates_{st.session_state['vertical_profile_uploader_version']}",
    )
    uploaded_files_by_variable = {}
    datasets_by_variable = {}
    dataframes_by_variable = {}
    source_files_by_variable = {}
    replicate_datasets_by_variable = {}
    replicate_dataframes_by_variable = {}
    replicate_source_files_by_variable = {}
    velocity_upload = None
    if use_replicate_profile:
        velocity_replicate_uploads = []
        if "Air Velocity" in selected_variables:
            with st.expander("Air Velocity Replicates", expanded=True):
                velocity_replicate_uploads = [
                    st.file_uploader(
                        f"Velocity Replicate {index} file",
                        type=["csv"],
                        accept_multiple_files=False,
                        key=f"vertical_profile_velocity_rep_{index}_{st.session_state['vertical_profile_uploader_version']}",
                    )
                    for index in (1, 2, 3)
                ]
                uploaded_files_by_variable["Air Velocity"] = velocity_replicate_uploads
        if "Temperature" in selected_variables:
            with st.expander("Temperature Replicates", expanded=False):
                reuse_velocity_replicates = bool(velocity_replicate_uploads) and st.checkbox(
                    "Reuse velocity replicate files",
                    key=f"vertical_profile_reuse_velocity_temp_reps_{st.session_state['vertical_profile_uploader_version']}",
                )
                temperature_replicate_uploads = (
                    velocity_replicate_uploads
                    if reuse_velocity_replicates
                    else [
                        st.file_uploader(
                            f"Temperature Replicate {index} file",
                            type=["csv"],
                            accept_multiple_files=False,
                            key=f"vertical_profile_temperature_rep_{index}_{st.session_state['vertical_profile_uploader_version']}",
                        )
                        for index in (1, 2, 3)
                    ]
                )
                uploaded_files_by_variable["Temperature"] = temperature_replicate_uploads
        if "Contaminant Concentration" in selected_variables:
            with st.expander("Contaminant Replicates", expanded=False):
                reuse_velocity_replicates = bool(velocity_replicate_uploads) and st.checkbox(
                    "Reuse velocity replicate files for contaminant",
                    key=f"vertical_profile_reuse_velocity_cont_reps_{st.session_state['vertical_profile_uploader_version']}",
                )
                contaminant_replicate_uploads = (
                    velocity_replicate_uploads
                    if reuse_velocity_replicates
                    else [
                        st.file_uploader(
                            f"Contaminant Replicate {index} file",
                            type=["csv"],
                            accept_multiple_files=False,
                            key=f"vertical_profile_contaminant_rep_{index}_{st.session_state['vertical_profile_uploader_version']}",
                        )
                        for index in (1, 2, 3)
                    ]
                )
                uploaded_files_by_variable["Contaminant Concentration"] = contaminant_replicate_uploads
        for variable, uploaded_files in uploaded_files_by_variable.items():
            replicate_datasets = []
            replicate_dataframes = []
            replicate_source_files = []
            for uploaded_file in uploaded_files:
                if uploaded_file is None:
                    continue
                dataset, read_error = read_csv_upload(uploaded_file)
                if read_error:
                    st.error(f"{variable}: {read_error}")
                    continue
                if dataset is not None:
                    replicate_datasets.append(dataset)
                    replicate_dataframes.append(dataset.dataframe)
                    replicate_source_files.append(dataset.filename)
            if replicate_datasets:
                replicate_datasets_by_variable[variable] = replicate_datasets
                replicate_dataframes_by_variable[variable] = replicate_dataframes
                replicate_source_files_by_variable[variable] = replicate_source_files
                datasets_by_variable[variable] = replicate_datasets[0]
    elif "Air Velocity" in selected_variables:
        velocity_upload = st.file_uploader(
            "Velocity CSV upload",
            type=["csv"],
            accept_multiple_files=False,
            key=f"vertical_profile_velocity_upload_{st.session_state['vertical_profile_uploader_version']}",
        )
        uploaded_files_by_variable["Air Velocity"] = velocity_upload

    if not use_replicate_profile and "Temperature" in selected_variables:
        use_velocity_for_temperature = False
        if velocity_upload is not None:
            use_velocity_for_temperature = st.checkbox(
                "Use Velocity CSV for Temperature",
                key=f"vertical_profile_reuse_velocity_for_temperature_{st.session_state['vertical_profile_uploader_version']}",
            )
        temperature_upload = (
            velocity_upload
            if use_velocity_for_temperature
            else st.file_uploader(
                "Temperature CSV upload",
                type=["csv"],
                accept_multiple_files=False,
                key=f"vertical_profile_temperature_upload_{st.session_state['vertical_profile_uploader_version']}",
            )
        )
        uploaded_files_by_variable["Temperature"] = temperature_upload

    if not use_replicate_profile and "Contaminant Concentration" in selected_variables:
        use_velocity_for_contaminant = False
        if velocity_upload is not None:
            use_velocity_for_contaminant = st.checkbox(
                "Use Velocity CSV for Contaminant Concentration",
                key=f"vertical_profile_reuse_velocity_for_contaminant_{st.session_state['vertical_profile_uploader_version']}",
            )
        contaminant_upload = (
            velocity_upload
            if use_velocity_for_contaminant
            else st.file_uploader(
                "Contaminant CSV upload",
                type=["csv"],
                accept_multiple_files=False,
                key=f"vertical_profile_contaminant_upload_{st.session_state['vertical_profile_uploader_version']}",
            )
        )
        uploaded_files_by_variable["Contaminant Concentration"] = contaminant_upload

    if not use_replicate_profile:
        for variable, uploaded_file in uploaded_files_by_variable.items():
            if uploaded_file is None:
                continue
            dataset, read_error = read_csv_upload(uploaded_file)
            if read_error:
                st.error(f"{variable}: {read_error}")
                continue
            if dataset is not None:
                datasets_by_variable[variable] = dataset
                dataframes_by_variable[variable] = dataset.dataframe
                source_files_by_variable[variable] = dataset.filename

    temperature_columns = None
    contaminant_columns = None
    if "Temperature" in selected_variables and "Temperature" in datasets_by_variable:
        temperature_dataset = datasets_by_variable["Temperature"]
        available_numeric_columns = [
            column for column in numeric_columns(temperature_dataset.dataframe)
            if not str(column).endswith("_std")
        ]
        if not available_numeric_columns:
            st.error("The Temperature CSV does not contain numeric columns for temperature mapping.")
        else:
            detected = default_temperature_columns(temperature_dataset.dataframe)
            st.markdown("### Temperature Columns")
            temperature_columns = [
                st.selectbox(
                    VERTICAL_PROFILE_TEMPERATURE_LABELS[index - 1],
                    options=available_numeric_columns,
                    index=available_numeric_columns.index(detected_column)
                    if detected_column in available_numeric_columns
                    else 0,
                    key=f"vertical_profile_temp_{index}_{st.session_state['vertical_profile_uploader_version']}",
                )
                for index, detected_column in enumerate(detected, start=1)
            ]
    if "Contaminant Concentration" in selected_variables and "Contaminant Concentration" in datasets_by_variable:
        contaminant_dataset = datasets_by_variable["Contaminant Concentration"]
        available_numeric_columns = [
            column for column in numeric_columns(contaminant_dataset.dataframe)
            if not str(column).endswith("_std")
        ]
        if not available_numeric_columns:
            st.error("The Contaminant CSV does not contain numeric columns for concentration mapping.")
        else:
            st.markdown("### Concentration Columns")
            contaminant_columns = [
                st.selectbox(
                    VERTICAL_PROFILE_CONCENTRATION_LABELS[index - 1],
                    options=available_numeric_columns,
                    index=min(index - 1, max(len(available_numeric_columns) - 1, 0)),
                    key=f"vertical_profile_conc_{index}_{st.session_state['vertical_profile_uploader_version']}",
                )
                for index in (1, 2, 3)
            ]

    if st.button("Process Vertical Profile Dataset", type="primary"):
        if not selected_variables:
            st.error("Select at least one vertical-profile variable.")
            return
        if use_replicate_profile:
            has_uploads = any(replicate_dataframes_by_variable.values())
        else:
            has_uploads = any(uploaded_files_by_variable.values())
        if not has_uploads:
            st.error("Please upload at least one source CSV for the selected variables.")
            return

        if use_replicate_profile:
            new_records, errors = build_replicate_vertical_profile_records(
                replicate_dataframes_by_variable,
                replicate_source_files_by_variable,
                dataset_type,
                position_number,
                heights,
                selected_variables,
                profile_mode,
                room_height_ft,
                supply_velocity=supply_velocity,
                temperature_columns=temperature_columns,
                tin=tin,
                tout=tout,
                contaminant_columns=contaminant_columns,
                contaminant_name=contaminant_name,
                concentration_unit=concentration_unit,
                cin=cin,
                cout=cout,
            )
        else:
            new_records, errors = build_vertical_profile_records_from_sources(
                dataframes_by_variable,
                source_files_by_variable,
                dataset_type,
                position_number,
                heights,
                selected_variables,
                profile_mode,
                room_height_ft,
                supply_velocity=supply_velocity,
                temperature_columns=temperature_columns,
                tin=tin,
                tout=tout,
                contaminant_columns=contaminant_columns,
                contaminant_name=contaminant_name,
                concentration_unit=concentration_unit,
                cin=cin,
                cout=cout,
            )
        if errors:
            for error in errors:
                st.error(error)
            if not new_records:
                return
        st.session_state["vertical_profile_records"].extend(new_records)
        st.session_state["vertical_profile_selected_variables"] = selected_variables
        st.session_state["vertical_profile_profile_mode"] = profile_mode
        st.session_state["vertical_profile_show_error_bars"] = show_error_bars
        st.session_state["vertical_profile_contaminant_name"] = contaminant_name
        st.session_state["vertical_profile_concentration_unit"] = concentration_unit
        st.session_state["vertical_profile_awaiting_continue"] = True
        if use_replicate_profile:
            processed_sources = ", ".join(
                sorted(
                    {
                        filename
                        for filenames in replicate_source_files_by_variable.values()
                        for filename in filenames
                    }
                )
            )
        else:
            processed_sources = ", ".join(sorted(set(source_files_by_variable.values())))
        st.success(f"Processed P{position_number} source data: {processed_sources}.")
        st.rerun()


def validate_required_replicate_uploads(replicate_uploads: list[object | None]) -> tuple[list[object], str | None]:
    required_count = 3
    if len(replicate_uploads) != required_count or any(uploaded_file is None for uploaded_file in replicate_uploads):
        return [], "Please upload all 3 replicate CSV files before processing."
    return list(replicate_uploads), None


def replicate_case_key(case_number: str | None, position_number: int, case_date: date | str | None) -> tuple[str, int, str]:
    clean_case_number = (case_number or "").strip()
    if isinstance(case_date, date):
        clean_date = case_date.isoformat()
    else:
        clean_date = str(case_date or "")
    return clean_case_number, int(position_number), clean_date


def replicate_case_label(case_number: str | None, position_number: int, case_date: date | str | None = None) -> str:
    case_text = f"C{str(case_number).strip()}" if case_number else "No case number"
    date_text = case_date.isoformat() if isinstance(case_date, date) else str(case_date or "")
    if date_text:
        return f"{case_text} - P{position_number} - {date_text}"
    return f"{case_text} - P{position_number}"


def completed_replicate_cases() -> list[dict]:
    state = st.session_state.setdefault("replicate_analysis_state", {})
    return state.setdefault("completed_cases", [])


def is_duplicate_replicate_case(case_number: str | None, position_number: int, case_date: date | str | None) -> bool:
    current_key = replicate_case_key(case_number, position_number, case_date)
    return any(completed_case.get("case_key") == current_key for completed_case in completed_replicate_cases())


def reset_current_replicate_inputs() -> None:
    st.session_state["replicate_input_version"] = st.session_state.get("replicate_input_version", 0) + 1
    state = st.session_state.setdefault("replicate_analysis_state", {})
    state["awaiting_next_case_choice"] = False
    state["processed_dataframe"] = None
    state["summary_dataframe"] = None
    state["base_name"] = None


def replicate_case_download_files(completed_case: dict) -> list[tuple[str, bytes]]:
    base_name = completed_case.get("base_name", "ReplicateMeanSD")
    summary_name = completed_case.get("summary_base_name", f"{base_name}_Summary")
    processed_dataframe = completed_case.get("processed_dataframe")
    summary_dataframe = completed_case.get("summary_dataframe")
    files: list[tuple[str, bytes]] = []
    if processed_dataframe is not None:
        files.append((f"{base_name}.csv", replicate_dataframe_to_csv_bytes(processed_dataframe)))
        files.append((f"{base_name}.xlsx", replicate_dataframe_to_xlsx_bytes(processed_dataframe, "ReplicateMeanSD")))
    if summary_dataframe is not None:
        files.append((f"{summary_name}.csv", replicate_dataframe_to_csv_bytes(summary_dataframe)))
        files.append((f"{summary_name}.xlsx", replicate_dataframe_to_xlsx_bytes(summary_dataframe, "ReplicateSummary")))
    return files


def build_all_replicate_cases_zip(completed_cases: list[dict]) -> bytes:
    files: list[tuple[str, bytes]] = []
    for completed_case in completed_cases:
        files.extend(replicate_case_download_files(completed_case))
    return build_replicate_zip(files)


def render_processed_replicate_cases_summary(completed_cases: list[dict]) -> None:
    if not completed_cases:
        return
    st.markdown("### Processed cases")
    summary_rows = [
        {
            "Case": completed_case.get("case_number") or "",
            "Position": f"P{completed_case.get('position_number')}",
            "Date": completed_case.get("case_date") or "",
            "Aligned Observations": completed_case.get("aligned_observations"),
            "Output": f"{completed_case.get('base_name')}.csv",
        }
        for completed_case in completed_cases
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    for index, completed_case in enumerate(completed_cases, start=1):
        label = replicate_case_label(
            completed_case.get("case_number"),
            int(completed_case.get("position_number") or 0),
            completed_case.get("case_date"),
        )
        with st.expander(label, expanded=index == len(completed_cases)):
            summary_dataframe = completed_case.get("summary_dataframe")
            if summary_dataframe is not None:
                st.dataframe(summary_dataframe, use_container_width=True, hide_index=True)
            files = replicate_case_download_files(completed_case)
            if files:
                col1, col2 = st.columns(2)
                base_name = completed_case.get("base_name", f"ReplicateCase{index}")
                processed_csv = dict(files).get(f"{base_name}.csv")
                processed_xlsx = dict(files).get(f"{base_name}.xlsx")
                if processed_csv:
                    col1.download_button(
                        "Download CSV",
                        processed_csv,
                        f"{base_name}.csv",
                        "text/csv",
                        key=f"replicate_case_{index}_csv",
                    )
                if processed_xlsx:
                    col2.download_button(
                        "Download XLSX",
                        processed_xlsx,
                        f"{base_name}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"replicate_case_{index}_xlsx",
                    )
                st.download_button(
                    "Download Case ZIP",
                    build_replicate_zip(files),
                    f"{base_name}_Files.zip",
                    "application/zip",
                    key=f"replicate_case_{index}_zip",
                )
    if len(completed_cases) > 1:
        st.download_button(
            "Download All Processed Replicate Cases",
            build_all_replicate_cases_zip(completed_cases),
            "All_Replicate_Cases.zip",
            "application/zip",
            key="replicate_all_cases_zip",
        )


def render_replicate_mean_sd_tool(show_back_button: bool = True) -> None:
    if show_back_button and st.button("Back to Data Analysis", key="analysis_back_from_replicate"):
        st.session_state["analysis_method"] = None
        st.rerun()

    st.subheader("Replicate Mean & SD")
    st.write("Combine replicate measurements and calculate mean and sample standard deviation for matching numeric variables.")

    existing_state = st.session_state.setdefault("replicate_analysis_state", {})
    input_version = st.session_state.setdefault("replicate_input_version", 0)
    if existing_state.get("processed_dataframe") is not None and existing_state.get("summary_dataframe") is not None:
        st.markdown("### Current Replicate Analysis")
        st.dataframe(existing_state["summary_dataframe"], use_container_width=True, hide_index=True)
        base_name = existing_state.get("base_name", "ReplicateMeanSD")
        processed_csv = replicate_dataframe_to_csv_bytes(existing_state["processed_dataframe"])
        processed_xlsx = replicate_dataframe_to_xlsx_bytes(existing_state["processed_dataframe"], "ReplicateMeanSD")
        summary_csv = replicate_dataframe_to_csv_bytes(existing_state["summary_dataframe"])
        summary_xlsx = replicate_dataframe_to_xlsx_bytes(existing_state["summary_dataframe"], "ReplicateSummary")
        zip_bytes = build_replicate_zip(
            [
                (f"{base_name}.csv", processed_csv),
                (f"{base_name}.xlsx", processed_xlsx),
                (f"{base_name}_Summary.csv", summary_csv),
                (f"{base_name}_Summary.xlsx", summary_xlsx),
            ]
        )
        col1, col2, col3 = st.columns(3)
        col1.download_button("Download Current Processed CSV", processed_csv, f"{base_name}.csv", "text/csv", key="replicate_current_csv")
        col2.download_button("Download Current Summary XLSX", summary_xlsx, f"{base_name}_Summary.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="replicate_current_summary_xlsx")
        col3.download_button("Download Current ZIP", zip_bytes, f"{base_name}_Files.zip", "application/zip", key="replicate_current_zip")

    completed_cases = completed_replicate_cases()
    render_processed_replicate_cases_summary(completed_cases)

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    position_number = meta_col1.selectbox(
        "Position Number",
        options=list(range(1, 7)),
        format_func=lambda value: f"P{value}",
        key=f"replicate_position_number_{input_version}",
    )
    case_number = meta_col2.text_input(
        "Case number if applicable",
        value="",
        placeholder="03",
        key=f"replicate_case_number_{input_version}",
    )
    replicate_date = meta_col3.date_input(
        "Date if applicable",
        value=date.today(),
        key=f"replicate_date_{input_version}",
    )

    alignment_mode = st.radio(
        "Alignment mode",
        options=("row", "timestamp"),
        format_func=lambda value: "Measurement sequence" if value == "row" else "Exact timestamp",
        horizontal=True,
        key=f"replicate_alignment_mode_{input_version}",
    )
    if alignment_mode == "row":
        st.caption("Output TIMESTAMP is taken from Replicate 1 when measurement-sequence alignment is used.")
    include_optional_numeric = st.checkbox(
        "Include all matching numeric columns",
        value=True,
        key=f"replicate_include_optional_numeric_{input_version}",
    )
    duplicate_current_case = is_duplicate_replicate_case(case_number.strip() or None, position_number, replicate_date)
    allow_duplicate_reprocess = False
    if duplicate_current_case:
        st.warning(
            f"{replicate_case_label(case_number.strip() or None, position_number, replicate_date)} "
            "has already been processed in this session."
        )
        allow_duplicate_reprocess = st.checkbox(
            "I want to intentionally reprocess this case",
            value=False,
            key=f"replicate_allow_duplicate_{input_version}",
        )

    st.markdown("### Replicate CSV files")
    replicate_uploads = [
        st.file_uploader(
            "Replicate 1 CSV",
            type=["csv"],
            accept_multiple_files=False,
            key=f"replicate_file_1_{input_version}",
        ),
        st.file_uploader(
            "Replicate 2 CSV",
            type=["csv"],
            accept_multiple_files=False,
            key=f"replicate_file_2_{input_version}",
        ),
        st.file_uploader(
            "Replicate 3 CSV",
            type=["csv"],
            accept_multiple_files=False,
            key=f"replicate_file_3_{input_version}",
        ),
    ]
    ready_uploads, upload_error = validate_required_replicate_uploads(replicate_uploads)
    if not upload_error:
        st.markdown("#### Uploaded replicate group")
        for replicate_index, uploaded_file in enumerate(ready_uploads, start=1):
            st.write(f"Replicate {replicate_index}: `{uploaded_file.name}`")

    if st.button("Process Replicates", type="primary"):
        if upload_error:
            st.error(upload_error)
            return
        if duplicate_current_case and not allow_duplicate_reprocess:
            st.warning("This case already exists. Confirm intentional reprocessing before processing again.")
            return

        with st.spinner("Processing replicate files..."):
            datasets: list[ReplicateDataset] = []
            for uploaded_file in ready_uploads:
                dataset, read_error = read_csv_upload(uploaded_file)
                if read_error:
                    st.error(read_error)
                    return
                if dataset is not None:
                    datasets.append(ReplicateDataset(dataset.filename, dataset.dataframe))

            if len(datasets) != 3:
                st.error("Please upload all 3 replicate CSV files before processing.")
                return

            processed_dataframe, alignment_report, errors = rowwise_replicate_statistics(
                datasets,
                include_optional_numeric=include_optional_numeric,
                alignment_mode=alignment_mode,
            )
        if errors:
            for error in errors:
                st.error(error)
            return

        summary_dataframe = replicate_summary_statistics(processed_dataframe)
        case_prefix = f"C{case_number.strip()}_" if case_number.strip() else ""
        base_name = f"{case_prefix}P{position_number:02d}_ReplicateMeanSD"
        processed_csv = replicate_dataframe_to_csv_bytes(processed_dataframe)
        processed_xlsx = replicate_dataframe_to_xlsx_bytes(processed_dataframe, "ReplicateMeanSD")
        summary_csv = replicate_dataframe_to_csv_bytes(summary_dataframe)
        summary_xlsx = replicate_dataframe_to_xlsx_bytes(summary_dataframe, "ReplicateSummary")
        zip_bytes = build_replicate_zip(
            [
                (f"{base_name}.csv", processed_csv),
                (f"{base_name}.xlsx", processed_xlsx),
                (f"{case_prefix}P{position_number:02d}_ReplicateSummary.csv", summary_csv),
                (f"{case_prefix}P{position_number:02d}_ReplicateSummary.xlsx", summary_xlsx),
            ]
        )
        completed_case = {
            "case_key": replicate_case_key(case_number.strip() or None, position_number, replicate_date),
            "position_number": position_number,
            "case_number": case_number.strip() or None,
            "case_date": replicate_date.isoformat() if replicate_date else None,
            "replicate_count": len(datasets),
            "source_files": [dataset.filename for dataset in datasets],
            "alignment": {
                "method": alignment_report.method,
                "matched_rows": alignment_report.matched_rows,
                "unmatched_rows": alignment_report.unmatched_rows,
                "warnings": alignment_report.warnings,
            },
            "aligned_observations": alignment_report.matched_rows,
            "base_name": base_name,
            "summary_base_name": f"{case_prefix}P{position_number:02d}_ReplicateSummary",
            "processed_dataframe": processed_dataframe,
            "summary_dataframe": summary_dataframe,
        }
        completed_cases.append(completed_case)
        st.session_state["replicate_analysis_state"] = {
            **existing_state,
            "completed_cases": completed_cases,
            "position_number": position_number,
            "case_number": case_number.strip() or None,
            "case_date": replicate_date.isoformat() if replicate_date else None,
            "replicate_count": len(datasets),
            "source_files": [dataset.filename for dataset in datasets],
            "alignment": {
                "method": alignment_report.method,
                "matched_rows": alignment_report.matched_rows,
                "unmatched_rows": alignment_report.unmatched_rows,
                "warnings": alignment_report.warnings,
            },
            "base_name": base_name,
            "processed_dataframe": processed_dataframe,
            "summary_dataframe": summary_dataframe,
            "awaiting_next_case_choice": True,
        }

        st.success("Replicate processing completed successfully.")
        st.success(f"Processed {len(datasets)} replicate files for P{position_number}.")
        st.write(
            "Alignment: `Measurement sequence`"
            if alignment_report.method == "measurement_sequence"
            else "Alignment: `Exact timestamp`"
        )
        st.write(f"Replicate files: `{len(datasets):,}`")
        if alignment_report.input_rows:
            st.write("Rows:")
            for filename, row_count in alignment_report.input_rows.items():
                st.write(f"- {filename}: `{row_count:,}`")
        st.write(f"Aligned observations: `{alignment_report.matched_rows:,}`")
        if any(alignment_report.unmatched_rows.values()):
            label = "Trailing rows excluded" if alignment_report.method == "measurement_sequence" else "Rows excluded"
            st.warning(f"{label}: {alignment_report.unmatched_rows}")
        for warning in alignment_report.warnings:
            st.info(warning)
        st.dataframe(summary_dataframe, use_container_width=True, hide_index=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.download_button("Download Processed CSV", processed_csv, f"{base_name}.csv", "text/csv")
        col2.download_button(
            "Download Processed XLSX",
            processed_xlsx,
            f"{base_name}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        col3.download_button("Download Summary CSV", summary_csv, f"{case_prefix}P{position_number:02d}_ReplicateSummary.csv", "text/csv")
        col4.download_button(
            "Download Summary XLSX",
            summary_xlsx,
            f"{case_prefix}P{position_number:02d}_ReplicateSummary.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download All Replicate Analysis Files",
            zip_bytes,
            f"{base_name}_Files.zip",
            "application/zip",
        )

    if st.session_state.get("replicate_analysis_state", {}).get("awaiting_next_case_choice"):
        st.markdown("## Process another case?")
        next_col, finish_col = st.columns(2)
        if next_col.button("Yes, process another case", key="replicate_process_another_case"):
            reset_current_replicate_inputs()
            st.rerun()
        if finish_col.button("No, finish", key="replicate_finish_batch"):
            st.session_state["replicate_analysis_state"]["awaiting_next_case_choice"] = False
            st.success("Finished replicate processing. Completed case results remain available above.")


def render_project_controls() -> None:
    st.markdown("### Project")
    load_col, save_col, new_col = st.columns([1.4, 1.2, 1])
    uploaded_project = load_col.file_uploader(
        "Load Existing Project",
        type=["edtproj"],
        accept_multiple_files=False,
        key="project_load_uploader",
    )
    if uploaded_project is not None and load_col.button("Load Project", type="primary"):
        try:
            project_state = load_project_archive(uploaded_project.getvalue())
            summary = restore_project_to_session(project_state)
            st.success("Project loaded successfully.")
            st.session_state["loaded_project_summary"] = summary
            st.rerun()
        except ProjectLoadError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Project could not be loaded: {exc}")

    project_state = build_current_project_state()
    project_bytes = save_project_archive(project_state)
    case_number, case_date = current_project_case_metadata()
    save_col.download_button(
        "Save Project",
        data=project_bytes,
        file_name=project_filename(case_number, case_date),
        mime="application/octet-stream",
        key="project_save_download",
    )

    new_col.caption("Starting a new analysis clears only Data Analysis project state.")
    if new_col.button("Start New Analysis"):
        clear_data_analysis_project_state()
        st.success("Started a new Data Analysis project.")
        st.rerun()

    summary = st.session_state.get("loaded_project_summary")
    if summary:
        analyses = summary.get("analyses") or []
        st.info(
            "Project: "
            f"{summary.get('project')}\n\n"
            f"Date: {summary.get('date') or 'Not set'}\n\n"
            "Available analyses: "
            f"{', '.join(analyses) if analyses else 'None'}\n\n"
            f"Project format version: {summary.get('format_version')}"
        )


def render_data_analysis_landing() -> None:
    st.subheader("Data Analysis")
    st.markdown("### Choose an analysis method")
    vector_col, profile_col = st.columns(2)
    with vector_col:
        st.markdown("#### 3D Vector Plot")
        st.write("Visualize 3D airflow direction and velocity magnitude in the room.")
        if st.button("3D Vector Plot", type="primary", key="select_3d_vector_plot"):
            st.session_state["analysis_method"] = "3d_vector_plot"
            st.rerun()
    with profile_col:
        st.markdown("#### Vertical Profile Plot")
        st.write("Plot vertical distributions of air velocity, temperature, and contaminant concentration.")
        if st.button("Vertical Profile Plot", type="primary", key="select_vertical_profile_plot"):
            st.session_state["analysis_method"] = "vertical_profile_plot"
            st.rerun()


def render_data_analysis_tool() -> None:
    st.session_state.setdefault("analysis_method", None)
    init_analysis_state()
    init_vertical_profile_state()
    st.session_state.setdefault("replicate_analysis_state", {})
    st.session_state.setdefault("project_metadata", {})
    render_project_controls()
    if st.session_state["analysis_method"] == "3d_vector_plot":
        render_3d_vector_plot_tool()
    elif st.session_state["analysis_method"] == "vertical_profile_plot":
        render_vertical_profile_tool()
    else:
        render_data_analysis_landing()


def main() -> None:
    st.set_page_config(
        page_title="Experimental Data Toolkit",
        layout="wide",
    )
    inject_global_styles()

    st.title("Experimental Data Toolkit")
    st.caption("A lightweight web tool for routine experimental data processing.")

    dat_tab, merge_tab, analysis_tab = st.tabs(["DAT → XLSX", "Merge CSV", "Data Analysis"])
    with dat_tab:
        render_dat_to_xlsx_tool()
    with merge_tab:
        render_merge_csv_tool()
    with analysis_tab:
        render_data_analysis_tool()


if __name__ == "__main__":
    main()
