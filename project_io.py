from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from io import BytesIO, StringIO
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

import pandas as pd

from analysis_processor import VectorRecord, vector_records_to_dataframe
from vertical_profile_processor import (
    VerticalProfileRecord,
    vertical_profile_records_to_unified_dataframe,
)


FORMAT_VERSION = 1
PROJECT_EXTENSION = ".edtproj"


class ProjectLoadError(ValueError):
    pass


def json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def clean_scalar(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: clean_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_scalar(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def vector_record_to_dict(record: VectorRecord) -> dict:
    return asdict(record)


def vector_record_from_dict(data: dict) -> VectorRecord:
    return VectorRecord(
        position_number=int(data["position_number"]),
        height_index=int(data["height_index"]),
        x=float(data["x"]),
        y=float(data["y"]),
        z=float(data["z"]),
        vx=float(data["vx"]),
        vy=float(data["vy"]),
        vz=float(data["vz"]),
        magnitude=float(data["magnitude"]),
        source_filename=str(data.get("source_filename", "")),
    )


def vertical_record_to_dict(record: VerticalProfileRecord) -> dict:
    data = asdict(record)
    data["replicate_means"] = list(record.replicate_means)
    data["source_files"] = list(record.source_files)
    return {key: clean_scalar(value) for key, value in data.items()}


def vertical_record_from_dict(data: dict) -> VerticalProfileRecord:
    return VerticalProfileRecord(
        dataset_type=str(data.get("dataset_type", "Experimental")),
        position_number=int(data["position_number"]),
        height_id=str(data["height_id"]),
        height_ft=float(data["height_ft"]),
        normalized_height=float(data.get("normalized_height") or 0.0),
        variable=str(data["variable"]),
        mean=float(data["mean"]),
        standard_deviation=float(data.get("standard_deviation") or 0.0),
        sample_count=int(data.get("sample_count") or 0),
        normalized_value=data.get("normalized_value"),
        source_file=str(data.get("source_file", "")),
        unit=str(data.get("unit", "")),
        contaminant_name=str(data.get("contaminant_name", "")),
        source_column=str(data.get("source_column", "")),
        replicate_means=tuple(data.get("replicate_means") or []),
        replicate_standard_deviation=data.get("replicate_standard_deviation"),
        normalized_standard_deviation=data.get("normalized_standard_deviation"),
        replicate_count=int(data.get("replicate_count") or 1),
        source_files=tuple(data.get("source_files") or []),
    )


def project_filename(case_number: str | None = None, case_date: str | date | None = None) -> str:
    if case_number and case_date:
        if isinstance(case_date, date):
            date_text = case_date.isoformat()
        else:
            date_text = str(case_date)
        return f"C{str(case_number).strip()}_{date_text.replace('-', '_')}{PROJECT_EXTENSION}"
    return f"Experimental_Data_Project{PROJECT_EXTENSION}"


def build_project_state(
    case_number: str | None = None,
    case_date: str | date | None = None,
    vector_records: list[VectorRecord] | None = None,
    vector_settings: dict | None = None,
    vertical_records: list[VerticalProfileRecord] | None = None,
    vertical_settings: dict | None = None,
    replicate_analysis: dict | None = None,
    ui_preferences: dict | None = None,
    app_name: str = "Experimental Data Toolkit",
    app_version: str = "local",
) -> dict:
    analyses = []
    if vector_records:
        analyses.append("3d_vector")
    if vertical_records:
        analyses.append("vertical_profile")
    if replicate_analysis:
        analyses.append("replicate_mean_sd")

    metadata = {
        "format_version": FORMAT_VERSION,
        "app_name": app_name,
        "app_version": app_version,
        "case_number": case_number,
        "case_date": case_date.isoformat() if isinstance(case_date, date) else case_date,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "analyses": analyses,
    }
    return {
        "format_version": FORMAT_VERSION,
        "metadata": metadata,
        "vector_analysis": {
            "records": [vector_record_to_dict(record) for record in (vector_records or [])],
            "settings": vector_settings or {},
        },
        "vertical_profile_analysis": {
            "records": [vertical_record_to_dict(record) for record in (vertical_records or [])],
            "settings": vertical_settings or {},
        },
        "replicate_analysis": replicate_analysis or {},
        "ui_preferences": ui_preferences or {},
    }


def project_state_to_jsonable(state: dict) -> dict:
    state = dict(state)
    state["format_version"] = FORMAT_VERSION
    metadata = dict(state.get("metadata") or {})
    metadata["format_version"] = FORMAT_VERSION
    state["metadata"] = metadata
    replicate_analysis = dict(state.get("replicate_analysis") or {})
    replicate_analysis.pop("processed_dataframe", None)
    replicate_analysis.pop("summary_dataframe", None)
    replicate_analysis.pop("completed_cases", None)
    state["replicate_analysis"] = replicate_analysis
    return state


def save_project_archive(state: dict) -> bytes:
    output = BytesIO()
    json_state = project_state_to_jsonable(state)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "project.json",
            json.dumps(json_state, indent=2, default=json_default),
        )
        archive.writestr(
            "metadata.json",
            json.dumps(json_state.get("metadata", {}), indent=2, default=json_default),
        )
        vector_records = [
            vector_record_from_dict(record)
            for record in json_state.get("vector_analysis", {}).get("records", [])
        ]
        if vector_records:
            archive.writestr("vector_summary.csv", vector_records_to_dataframe(vector_records).to_csv(index=False))
        vertical_records = [
            vertical_record_from_dict(record)
            for record in json_state.get("vertical_profile_analysis", {}).get("records", [])
        ]
        if vertical_records:
            variables = json_state.get("vertical_profile_analysis", {}).get("settings", {}).get("selected_variables")
            archive.writestr(
                "vertical_profile_summary.csv",
                vertical_profile_records_to_unified_dataframe(vertical_records, variables).to_csv(index=False),
            )
        replicate_analysis = state.get("replicate_analysis") or {}
        processed_dataframe = replicate_analysis.get("processed_dataframe")
        summary_dataframe = replicate_analysis.get("summary_dataframe")
        if isinstance(processed_dataframe, pd.DataFrame):
            archive.writestr("replicate_processed.csv", processed_dataframe.to_csv(index=False))
        if isinstance(summary_dataframe, pd.DataFrame):
            archive.writestr("replicate_summary.csv", summary_dataframe.to_csv(index=False))
    return output.getvalue()


def load_project_archive(file_bytes: bytes) -> dict:
    try:
        with ZipFile(BytesIO(file_bytes), "r") as archive:
            if "project.json" not in archive.namelist():
                raise ProjectLoadError("Project file is missing project.json.")
            try:
                state = json.loads(archive.read("project.json").decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ProjectLoadError(f"project.json is corrupted: {exc}") from exc
            version = state.get("format_version")
            if version != FORMAT_VERSION:
                raise ProjectLoadError(f"Unsupported project format_version: {version}.")

            replicate_analysis = dict(state.get("replicate_analysis") or {})
            if "replicate_processed.csv" in archive.namelist():
                replicate_analysis["processed_dataframe"] = pd.read_csv(
                    StringIO(archive.read("replicate_processed.csv").decode("utf-8"))
                )
            if "replicate_summary.csv" in archive.namelist():
                replicate_analysis["summary_dataframe"] = pd.read_csv(
                    StringIO(archive.read("replicate_summary.csv").decode("utf-8"))
                )
            state["replicate_analysis"] = replicate_analysis
            return state
    except BadZipFile as exc:
        raise ProjectLoadError("Project file is not a valid .edtproj ZIP archive.") from exc


def decode_project_state(state: dict) -> dict:
    if state.get("format_version") != FORMAT_VERSION:
        raise ProjectLoadError(f"Unsupported project format_version: {state.get('format_version')}.")
    return {
        "metadata": state.get("metadata") or {},
        "vector_records": [
            vector_record_from_dict(record)
            for record in state.get("vector_analysis", {}).get("records", [])
        ],
        "vector_settings": state.get("vector_analysis", {}).get("settings", {}),
        "vertical_records": [
            vertical_record_from_dict(record)
            for record in state.get("vertical_profile_analysis", {}).get("records", [])
        ],
        "vertical_settings": state.get("vertical_profile_analysis", {}).get("settings", {}),
        "replicate_analysis": state.get("replicate_analysis") or {},
        "ui_preferences": state.get("ui_preferences") or {},
    }
