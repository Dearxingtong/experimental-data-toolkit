from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


PROFILE_VARIABLES = ("Air Velocity", "Temperature", "Contaminant Concentration")
PROFILE_MODES = ("Raw Profiles",)
DATASET_TYPES = ("Experimental", "CFD", "Experimental vs CFD")
DEFAULT_ROOM_HEIGHT_FT = 8.75
POSITION_STYLE_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    6: "#8c564b",
}
VELOCITY_COMPONENT_COLUMNS = (
    ("Vx1", "Vy1", "Vz1"),
    ("Vx2", "Vy2", "Vz2"),
    ("Vx3", "Vy3", "Vz3"),
)
TEMPERATURE_DEFAULT_COLUMNS = ("Temp1", "Temp2", "Temp3")
FT_TO_M = 0.3048
MPS_TO_FPM = 196.850394
ROOM_HEIGHT_FT = 8.75
ROOM_HEIGHT_M = ROOM_HEIGHT_FT * FT_TO_M
VERTICAL_PROFILE_AXIS_LIMITS = {
    ("si", "Air Velocity"): (0.0, 0.35),
    ("ip", "Air Velocity"): (0.0, 70.0),
    ("si", "Temperature"): (23.0, 25.0),
    ("ip", "Temperature"): (73.0, 77.0),
}


@dataclass
class VerticalProfileRecord:
    dataset_type: str
    position_number: int
    height_id: str
    height_ft: float
    normalized_height: float
    variable: str
    mean: float
    standard_deviation: float
    sample_count: int
    normalized_value: float | None
    source_file: str
    unit: str
    contaminant_name: str = ""
    source_column: str = ""
    replicate_means: tuple[float | None, ...] = tuple()
    replicate_standard_deviation: float | None = None
    normalized_standard_deviation: float | None = None
    replicate_count: int = 1
    source_files: tuple[str, ...] = tuple()


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if not pd.to_numeric(dataframe[column], errors="coerce").dropna().empty
    ]


def validate_profile_heights(heights: list[float], room_height_ft: float | None = None) -> list[str]:
    errors: list[str] = []
    for index, height in enumerate(heights, start=1):
        try:
            parsed_height = float(height)
        except (TypeError, ValueError):
            errors.append(f"Height {index} must be numeric.")
            continue
        if parsed_height <= 0:
            errors.append(f"Height {index} must be greater than 0 ft.")
        if room_height_ft is not None and room_height_ft > 0 and parsed_height > room_height_ft:
            errors.append(f"Height {index} must be less than or equal to the room height.")
    return errors


def validate_columns_exist(dataframe: pd.DataFrame, columns: list[str], label: str) -> list[str]:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        return [f"{label} missing required columns: {', '.join(missing)}."]
    return []


def numeric_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(dataframe[column], errors="coerce").dropna()


def summarize_series(series: pd.Series, label: str) -> tuple[float, float, int, str | None]:
    if series.empty:
        return 0.0, 0.0, 0, f"{label} has no valid numeric samples."
    return float(series.mean()), float(series.std(ddof=1)) if len(series) > 1 else 0.0, int(series.count()), None


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(pd.Series(values, dtype="float64").std(ddof=1))


def compute_air_speed_profile(dataframe: pd.DataFrame) -> tuple[list[tuple[float, float, int]], list[str]]:
    errors: list[str] = []
    for component_columns in VELOCITY_COMPONENT_COLUMNS:
        errors.extend(validate_columns_exist(dataframe, list(component_columns), "Air velocity"))
    if errors:
        return [], errors

    profile: list[tuple[float, float, int]] = []
    for height_index, (vx_column, vy_column, vz_column) in enumerate(VELOCITY_COMPONENT_COLUMNS, start=1):
        vx = pd.to_numeric(dataframe[vx_column], errors="coerce")
        vy = pd.to_numeric(dataframe[vy_column], errors="coerce")
        vz = pd.to_numeric(dataframe[vz_column], errors="coerce")
        speed = (vx**2 + vy**2 + vz**2).pow(0.5).dropna()
        mean, standard_deviation, sample_count, error = summarize_series(
            speed,
            f"Air velocity H{height_index}",
        )
        if error:
            errors.append(error)
        profile.append((mean, standard_deviation, sample_count))
    return profile, errors


def compute_air_speed_profile_from_mean_std_columns(dataframe: pd.DataFrame) -> tuple[list[tuple[float, float, int]], list[str], bool]:
    errors: list[str] = []
    for component_columns in VELOCITY_COMPONENT_COLUMNS:
        errors.extend(validate_columns_exist(dataframe, list(component_columns), "Air velocity"))
    if errors:
        return [], errors, False

    has_std_columns = True
    profile: list[tuple[float, float, int]] = []
    for height_index, (vx_column, vy_column, vz_column) in enumerate(VELOCITY_COMPONENT_COLUMNS, start=1):
        vx = pd.to_numeric(dataframe[vx_column], errors="coerce")
        vy = pd.to_numeric(dataframe[vy_column], errors="coerce")
        vz = pd.to_numeric(dataframe[vz_column], errors="coerce")
        speed = (vx**2 + vy**2 + vz**2).pow(0.5).dropna()
        mean = float(speed.mean()) if not speed.empty else 0.0
        sample_count = int(speed.count())
        std_columns = [f"{vx_column}_std", f"{vy_column}_std", f"{vz_column}_std"]
        if all(column in dataframe.columns for column in std_columns):
            component_stds = [
                pd.to_numeric(dataframe[column], errors="coerce").dropna().mean()
                for column in std_columns
            ]
            standard_deviation = float(pd.Series(component_stds, dtype="float64").pow(2).sum() ** 0.5)
        else:
            has_std_columns = False
            standard_deviation = 0.0
        if not sample_count:
            errors.append(f"Air velocity H{height_index} has no valid numeric samples.")
        profile.append((mean, standard_deviation, sample_count))
    return profile, errors, has_std_columns


def default_temperature_columns(dataframe: pd.DataFrame) -> list[str | None]:
    columns = []
    for column in TEMPERATURE_DEFAULT_COLUMNS:
        columns.append(column if column in dataframe.columns else None)
    return columns


def compute_column_profile(
    dataframe: pd.DataFrame,
    columns: list[str],
    variable_label: str,
) -> tuple[list[tuple[float, float, int]], list[str]]:
    errors = validate_columns_exist(dataframe, columns, variable_label)
    if errors:
        return [], errors

    profile: list[tuple[float, float, int]] = []
    for height_index, column in enumerate(columns, start=1):
        series = numeric_series(dataframe, column)
        mean, standard_deviation, sample_count, error = summarize_series(
            series,
            f"{variable_label} H{height_index}",
        )
        if error:
            errors.append(error)
        profile.append((mean, standard_deviation, sample_count))
    return profile, errors


def compute_column_profile_from_mean_std_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    variable_label: str,
) -> tuple[list[tuple[float, float, int]], list[str], bool]:
    errors = validate_columns_exist(dataframe, columns, variable_label)
    if errors:
        return [], errors, False

    has_std_columns = True
    profile: list[tuple[float, float, int]] = []
    for height_index, column in enumerate(columns, start=1):
        series = numeric_series(dataframe, column)
        if series.empty:
            profile.append((0.0, 0.0, 0))
            errors.append(f"{variable_label} H{height_index} has no valid numeric samples.")
            continue
        std_column = f"{column}_std"
        mean = float(series.mean())
        sample_count = int(series.count())
        if std_column in dataframe.columns:
            std_series = numeric_series(dataframe, std_column)
            standard_deviation = float(std_series.mean()) if not std_series.empty else 0.0
        else:
            has_std_columns = False
            standard_deviation = 0.0
        profile.append((mean, standard_deviation, sample_count))
    return profile, errors, has_std_columns


def compute_temperature_profile(dataframe: pd.DataFrame, columns: list[str]) -> tuple[list[tuple[float, float, int]], list[str]]:
    return compute_column_profile(dataframe, columns, "Temperature")


def compute_contaminant_profile(dataframe: pd.DataFrame, columns: list[str]) -> tuple[list[tuple[float, float, int]], list[str]]:
    return compute_column_profile(dataframe, columns, "Contaminant concentration")


def normalize_height(height_ft: float, room_height_ft: float) -> float:
    return height_ft / room_height_ft


def normalize_velocity(mean_speed: float, supply_velocity: float) -> float:
    return mean_speed / supply_velocity


def normalize_temperature(mean_temperature: float, tin: float, tout: float) -> float:
    return (mean_temperature - tin) / (tout - tin)


def normalize_concentration(mean_concentration: float, cin: float, cout: float) -> float:
    return (mean_concentration - cin) / (cout - cin)


def validate_normalization_inputs(
    selected_variables: list[str],
    profile_mode: str,
    supply_velocity: float | None = None,
    tin: float | None = None,
    tout: float | None = None,
    cin: float | None = None,
    cout: float | None = None,
) -> list[str]:
    return []


def build_vertical_profile_records(
    dataframe: pd.DataFrame,
    dataset_type: str,
    position_number: int,
    heights_ft: list[float],
    selected_variables: list[str],
    profile_mode: str,
    room_height_ft: float,
    source_file: str,
    supply_velocity: float | None = None,
    temperature_columns: list[str] | None = None,
    tin: float | None = None,
    tout: float | None = None,
    contaminant_columns: list[str] | None = None,
    contaminant_name: str = "Contaminant",
    concentration_unit: str = "",
    cin: float | None = None,
    cout: float | None = None,
) -> tuple[list[VerticalProfileRecord], list[str]]:
    errors: list[str] = []
    if not selected_variables:
        errors.append("Select at least one vertical-profile variable.")
    if dataset_type != "Experimental":
        errors.append("CFD input support is reserved for a future update.")
    profile_mode = "Raw Profiles"
    errors.extend(validate_profile_heights(heights_ft, room_height_ft))
    errors.extend(
        validate_normalization_inputs(
            selected_variables,
            profile_mode,
            supply_velocity,
            tin,
            tout,
            cin,
            cout,
        )
    )
    if errors:
        return [], errors

    records: list[VerticalProfileRecord] = []
    profile_sources: dict[str, tuple[list[tuple[float, float, int]], list[str], str, list[str]]] = {}

    if "Air Velocity" in selected_variables:
        profile, profile_errors = compute_air_speed_profile(dataframe)
        profile_sources["Air Velocity"] = (profile, profile_errors, "m/s", ["speed"] * 3)
    if "Temperature" in selected_variables:
        columns = temperature_columns or list(TEMPERATURE_DEFAULT_COLUMNS)
        profile, profile_errors = compute_temperature_profile(dataframe, columns)
        profile_sources["Temperature"] = (profile, profile_errors, "deg C", columns)
    if "Contaminant Concentration" in selected_variables:
        columns = contaminant_columns or []
        profile, profile_errors = compute_contaminant_profile(dataframe, columns)
        profile_sources["Contaminant Concentration"] = (
            profile,
            profile_errors,
            concentration_unit,
            columns,
        )

    for _variable, (_profile, profile_errors, _unit, _columns) in profile_sources.items():
        errors.extend(profile_errors)
    if errors:
        return [], errors

    for variable, (profile, _profile_errors, unit, source_columns) in profile_sources.items():
        for height_index, (height_ft, stats) in enumerate(zip(heights_ft, profile), start=1):
            mean, standard_deviation, sample_count = stats
            records.append(
                VerticalProfileRecord(
                    dataset_type=dataset_type,
                    position_number=position_number,
                    height_id=f"H{height_index}",
                    height_ft=float(height_ft),
                    normalized_height=0.0,
                    variable=variable,
                    mean=mean,
                    standard_deviation=standard_deviation,
                    sample_count=sample_count,
                    normalized_value=None,
                    source_file=source_file,
                    unit=unit,
                    contaminant_name=contaminant_name if variable == "Contaminant Concentration" else "",
                    source_column=source_columns[height_index - 1],
                    normalized_standard_deviation=None,
                )
            )

    records.sort(key=lambda record: (record.variable, record.position_number, record.height_ft))
    return records, []


def build_vertical_profile_records_from_sources(
    dataframes_by_variable: dict[str, pd.DataFrame | None],
    source_files_by_variable: dict[str, str],
    dataset_type: str,
    position_number: int,
    heights_ft: list[float],
    selected_variables: list[str],
    profile_mode: str,
    room_height_ft: float,
    supply_velocity: float | None = None,
    temperature_columns: list[str] | None = None,
    tin: float | None = None,
    tout: float | None = None,
    contaminant_columns: list[str] | None = None,
    contaminant_name: str = "Contaminant",
    concentration_unit: str = "",
    cin: float | None = None,
    cout: float | None = None,
) -> tuple[list[VerticalProfileRecord], list[str]]:
    errors: list[str] = []
    if not selected_variables:
        errors.append("Select at least one vertical-profile variable.")
    if dataset_type != "Experimental":
        errors.append("CFD input support is reserved for a future update.")
    profile_mode = "Raw Profiles"
    errors.extend(validate_profile_heights(heights_ft, room_height_ft))
    errors.extend(
        validate_normalization_inputs(
            selected_variables,
            profile_mode,
            supply_velocity,
            tin,
            tout,
            cin,
            cout,
        )
    )
    if errors:
        return [], errors

    records: list[VerticalProfileRecord] = []
    for variable in selected_variables:
        dataframe = dataframes_by_variable.get(variable)
        if dataframe is None:
            continue

        used_explicit_std_columns = False
        if variable == "Air Velocity":
            profile, profile_errors, used_explicit_std_columns = compute_air_speed_profile_from_mean_std_columns(dataframe)
            unit = "m/s"
            source_columns = ["speed"] * 3
        elif variable == "Temperature":
            source_columns = temperature_columns or list(TEMPERATURE_DEFAULT_COLUMNS)
            profile, profile_errors, used_explicit_std_columns = compute_column_profile_from_mean_std_columns(
                dataframe,
                source_columns,
                "Temperature",
            )
            unit = "deg C"
        elif variable == "Contaminant Concentration":
            source_columns = contaminant_columns or []
            profile, profile_errors, used_explicit_std_columns = compute_column_profile_from_mean_std_columns(
                dataframe,
                source_columns,
                "Contaminant concentration",
            )
            unit = concentration_unit
        else:
            continue
        if not used_explicit_std_columns:
            if variable == "Air Velocity":
                profile, profile_errors = compute_air_speed_profile(dataframe)
            elif variable == "Temperature":
                profile, profile_errors = compute_temperature_profile(dataframe, source_columns)
            else:
                profile, profile_errors = compute_contaminant_profile(dataframe, source_columns)
        if profile_errors:
            errors.extend(f"{variable}: {error}" for error in profile_errors)
            continue

        for height_index, (height_ft, stats) in enumerate(zip(heights_ft, profile), start=1):
            mean, standard_deviation, sample_count = stats
            records.append(
                VerticalProfileRecord(
                    dataset_type=dataset_type,
                    position_number=position_number,
                    height_id=f"H{height_index}",
                    height_ft=float(height_ft),
                    normalized_height=0.0,
                    variable=variable,
                    mean=mean,
                    standard_deviation=standard_deviation,
                    sample_count=sample_count,
                    normalized_value=None,
                    source_file=source_files_by_variable.get(variable, ""),
                    unit=unit,
                    contaminant_name=contaminant_name if variable == "Contaminant Concentration" else "",
                    source_column=source_columns[height_index - 1] if len(source_columns) >= height_index else "",
                    normalized_standard_deviation=None,
                )
            )

    records.sort(key=lambda record: (record.variable, record.position_number, record.height_ft))
    if not records and not errors:
        errors.append("Upload at least one source CSV for the selected variables.")
    return records, errors


def normalize_standard_deviation(
    variable: str,
    standard_deviation: float | None,
    supply_velocity: float | None = None,
    tin: float | None = None,
    tout: float | None = None,
    cin: float | None = None,
    cout: float | None = None,
) -> float | None:
    if standard_deviation is None:
        return None
    if variable == "Air Velocity":
        return standard_deviation / abs(float(supply_velocity))
    if variable == "Temperature":
        return standard_deviation / abs(float(tout) - float(tin))
    if variable == "Contaminant Concentration":
        return standard_deviation / abs(float(cout) - float(cin))
    return None


def normalized_sd_for_record(
    variable: str,
    standard_deviation: float | None,
    profile_mode: str,
    supply_velocity: float | None = None,
    tin: float | None = None,
    tout: float | None = None,
    cin: float | None = None,
    cout: float | None = None,
) -> float | None:
    return None


def build_replicate_vertical_profile_records(
    replicate_dataframes_by_variable: dict[str, list[pd.DataFrame]],
    source_files_by_variable: dict[str, list[str]],
    dataset_type: str,
    position_number: int,
    heights_ft: list[float],
    selected_variables: list[str],
    profile_mode: str,
    room_height_ft: float,
    supply_velocity: float | None = None,
    temperature_columns: list[str] | None = None,
    tin: float | None = None,
    tout: float | None = None,
    contaminant_columns: list[str] | None = None,
    contaminant_name: str = "Contaminant",
    concentration_unit: str = "",
    cin: float | None = None,
    cout: float | None = None,
) -> tuple[list[VerticalProfileRecord], list[str]]:
    errors: list[str] = []
    if dataset_type != "Experimental":
        errors.append("CFD input support is reserved for a future update.")
    errors.extend(validate_profile_heights(heights_ft, room_height_ft))
    profile_mode = "Raw Profiles"
    if errors:
        return [], errors

    records: list[VerticalProfileRecord] = []
    for variable in selected_variables:
        replicate_dataframes = [dataframe for dataframe in replicate_dataframes_by_variable.get(variable, []) if dataframe is not None]
        if not replicate_dataframes:
            continue

        source_files = source_files_by_variable.get(variable, [])
        if variable == "Air Velocity":
            profile_getter = compute_air_speed_profile
            unit = "m/s"
            source_columns = ["speed"] * 3
        elif variable == "Temperature":
            columns = temperature_columns or list(TEMPERATURE_DEFAULT_COLUMNS)
            profile_getter = lambda dataframe, selected_columns=columns: compute_temperature_profile(dataframe, selected_columns)
            unit = "deg C"
            source_columns = columns
        elif variable == "Contaminant Concentration":
            columns = contaminant_columns or []
            profile_getter = lambda dataframe, selected_columns=columns: compute_contaminant_profile(dataframe, selected_columns)
            unit = concentration_unit
            source_columns = columns
        else:
            continue

        replicate_profiles: list[list[tuple[float, float, int]]] = []
        for replicate_index, dataframe in enumerate(replicate_dataframes, start=1):
            profile, profile_errors = profile_getter(dataframe)
            if profile_errors:
                errors.extend(
                    f"{variable} replicate {replicate_index}: {error}"
                    for error in profile_errors
                )
            else:
                replicate_profiles.append(profile)
        if errors:
            continue

        for height_index, height_ft in enumerate(heights_ft, start=1):
            replicate_means = [
                profile[height_index - 1][0]
                for profile in replicate_profiles
                if len(profile) >= height_index
            ]
            if not replicate_means:
                continue
            profile_mean = float(pd.Series(replicate_means, dtype="float64").mean())
            replicate_sd = sample_std(replicate_means)
            sample_count = sum(
                profile[height_index - 1][2]
                for profile in replicate_profiles
                if len(profile) >= height_index
            )
            records.append(
                VerticalProfileRecord(
                    dataset_type=dataset_type,
                    position_number=position_number,
                    height_id=f"H{height_index}",
                    height_ft=float(height_ft),
                    normalized_height=0.0,
                    variable=variable,
                    mean=profile_mean,
                    standard_deviation=replicate_sd if replicate_sd is not None else 0.0,
                    sample_count=sample_count,
                    normalized_value=None,
                    source_file=", ".join(source_files),
                    unit=unit,
                    contaminant_name=contaminant_name if variable == "Contaminant Concentration" else "",
                    source_column=source_columns[height_index - 1] if len(source_columns) >= height_index else "",
                    replicate_means=tuple(replicate_means),
                    replicate_standard_deviation=replicate_sd,
                    normalized_standard_deviation=None,
                    replicate_count=len(replicate_means),
                    source_files=tuple(source_files),
                )
            )

    records.sort(key=lambda record: (record.variable, record.position_number, record.height_ft))
    if not records and not errors:
        errors.append("Upload at least one replicate source CSV for the selected variables.")
    return records, errors


def vertical_profile_records_to_dataframe(records: list[VerticalProfileRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Dataset Type": record.dataset_type,
                "Position": f"P{record.position_number}",
                "Height ID": record.height_id,
                "Height (ft)": record.height_ft,
                "Variable": record.variable,
                "Mean": record.mean,
                "Standard Deviation": record.standard_deviation,
                "Sample Count": record.sample_count,
                "Unit": record.unit,
                "Contaminant Name": record.contaminant_name,
                "Source Column": record.source_column,
                "Source File": record.source_file,
                "Replicate 1 Mean": record.replicate_means[0] if len(record.replicate_means) > 0 else pd.NA,
                "Replicate 2 Mean": record.replicate_means[1] if len(record.replicate_means) > 1 else pd.NA,
                "Replicate 3 Mean": record.replicate_means[2] if len(record.replicate_means) > 2 else pd.NA,
                "Replicate Mean": record.mean,
                "Replicate SD": record.replicate_standard_deviation,
                "Replicate Count": record.replicate_count,
                "Source File 1": record.source_files[0] if len(record.source_files) > 0 else pd.NA,
                "Source File 2": record.source_files[1] if len(record.source_files) > 1 else pd.NA,
                "Source File 3": record.source_files[2] if len(record.source_files) > 2 else pd.NA,
            }
            for record in records
        ]
    )


def vertical_profile_records_to_unified_dataframe(
    records: list[VerticalProfileRecord],
    selected_variables: list[str] | None = None,
) -> pd.DataFrame:
    variables = selected_variables or sorted({record.variable for record in records})
    grouped: dict[tuple[str, str, float], dict[str, object]] = {}
    for record in records:
        key = (f"P{record.position_number}", record.height_id, record.height_ft)
        row = grouped.setdefault(
            key,
            {
                "Dataset Type": record.dataset_type,
                "Position": f"P{record.position_number}",
                "Height ID": record.height_id,
                "Height (ft)": record.height_ft,
                "Height (m)": record.height_ft * FT_TO_M,
            },
        )
        prefix = record.variable
        row[f"{prefix} Mean"] = record.mean
        row[f"{prefix} Standard Deviation"] = record.standard_deviation
        row[f"{prefix} Sample Count"] = record.sample_count
        row[f"{prefix} Replicate 1 Mean"] = record.replicate_means[0] if len(record.replicate_means) > 0 else pd.NA
        row[f"{prefix} Replicate 2 Mean"] = record.replicate_means[1] if len(record.replicate_means) > 1 else pd.NA
        row[f"{prefix} Replicate 3 Mean"] = record.replicate_means[2] if len(record.replicate_means) > 2 else pd.NA
        row[f"{prefix} Replicate Mean"] = record.mean
        row[f"{prefix} Replicate SD"] = record.replicate_standard_deviation
        row[f"{prefix} Replicate Count"] = record.replicate_count
        row[f"{prefix} Unit"] = record.unit
        row[f"{prefix} Source File"] = record.source_file
        row[f"{prefix} Source File 1"] = record.source_files[0] if len(record.source_files) > 0 else pd.NA
        row[f"{prefix} Source File 2"] = record.source_files[1] if len(record.source_files) > 1 else pd.NA
        row[f"{prefix} Source File 3"] = record.source_files[2] if len(record.source_files) > 2 else pd.NA
        row[f"{prefix} Source Column"] = record.source_column
        if record.variable == "Air Velocity":
            row["Velocity Mean (m/s)"] = record.mean
            row["Velocity SD (m/s)"] = profile_plot_standard_deviation(record, "Raw Profiles")
            row["Velocity Mean (fpm)"] = record.mean * MPS_TO_FPM
            row["Velocity SD (fpm)"] = (
                profile_plot_standard_deviation(record, "Raw Profiles") * MPS_TO_FPM
                if profile_plot_standard_deviation(record, "Raw Profiles") is not None
                else pd.NA
            )
        elif record.variable == "Temperature":
            row["Temperature Mean (°C)"] = record.mean
            row["Temperature SD (°C)"] = profile_plot_standard_deviation(record, "Raw Profiles")
            row["Temperature Mean (°F)"] = record.mean * 9.0 / 5.0 + 32.0
            row["Temperature SD (°F)"] = (
                profile_plot_standard_deviation(record, "Raw Profiles") * 9.0 / 5.0
                if profile_plot_standard_deviation(record, "Raw Profiles") is not None
                else pd.NA
            )
        elif record.variable == "Contaminant Concentration":
            row["Contaminant Mean"] = record.mean
            row["Contaminant SD"] = profile_plot_standard_deviation(record, "Raw Profiles")
            row["Contaminant Unit"] = record.unit

    rows = [grouped[key] for key in sorted(grouped, key=lambda item: (item[0], item[2]))]
    dataframe = pd.DataFrame(rows)
    base_columns = ["Dataset Type", "Position", "Height ID", "Height (ft)", "Height (m)"]
    converted_columns = [
        "Velocity Mean (m/s)",
        "Velocity SD (m/s)",
        "Velocity Mean (fpm)",
        "Velocity SD (fpm)",
        "Temperature Mean (°C)",
        "Temperature SD (°C)",
        "Temperature Mean (°F)",
        "Temperature SD (°F)",
        "Contaminant Mean",
        "Contaminant SD",
        "Contaminant Unit",
    ]
    variable_columns: list[str] = []
    for variable in variables:
        variable_columns.extend(
            [
                f"{variable} Mean",
                f"{variable} Standard Deviation",
                f"{variable} Sample Count",
                f"{variable} Replicate 1 Mean",
                f"{variable} Replicate 2 Mean",
                f"{variable} Replicate 3 Mean",
                f"{variable} Replicate Mean",
                f"{variable} Replicate SD",
                f"{variable} Replicate Count",
                f"{variable} Unit",
                f"{variable} Source File",
                f"{variable} Source File 1",
                f"{variable} Source File 2",
                f"{variable} Source File 3",
                f"{variable} Source Column",
            ]
        )
    for column in base_columns + converted_columns + variable_columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA
    return dataframe[base_columns + converted_columns + variable_columns]


def x_axis_label(
    variable: str,
    profile_mode: str,
    contaminant_name: str = "Contaminant",
    concentration_unit: str = "",
    unit_system: str = "si",
) -> str:
    if variable == "Air Velocity":
        return "Air velocity (fpm)" if unit_system == "ip" else "Air velocity (m/s)"
    if variable == "Temperature":
        return "Temperature (°F)" if unit_system == "ip" else "Temperature (°C)"
    unit_text = f" ({concentration_unit})" if concentration_unit else ""
    return f"{contaminant_name} concentration{unit_text}"


def y_axis_label(profile_mode: str, unit_system: str = "si") -> str:
    return "Height (ft)" if unit_system == "ip" else "Height (m)"


def profile_figure_title(profile_mode: str, unit_system: str = "si") -> str:
    return "Vertical Profiles — IP Units" if unit_system == "ip" else "Vertical Profiles — SI Units"


def subplot_title(variable: str, contaminant_name: str = "Contaminant") -> str:
    if variable == "Contaminant Concentration":
        return f"{contaminant_name} Concentration"
    return variable


def profile_plot_standard_deviation(record: VerticalProfileRecord, profile_mode: str) -> float | None:
    if record.replicate_standard_deviation is not None and pd.notna(record.replicate_standard_deviation):
        return float(record.replicate_standard_deviation)
    if record.standard_deviation is not None and pd.notna(record.standard_deviation):
        return float(record.standard_deviation)
    return None


def convert_profile_value(record: VerticalProfileRecord, unit_system: str) -> float:
    if unit_system == "ip":
        if record.variable == "Air Velocity":
            return record.mean * MPS_TO_FPM
        if record.variable == "Temperature":
            return record.mean * 9.0 / 5.0 + 32.0
    return record.mean


def convert_profile_standard_deviation(record: VerticalProfileRecord, profile_mode: str, unit_system: str) -> float | None:
    standard_deviation = profile_plot_standard_deviation(record, profile_mode)
    if standard_deviation is None:
        return None
    if unit_system == "ip":
        if record.variable == "Air Velocity":
            return standard_deviation * MPS_TO_FPM
        if record.variable == "Temperature":
            return standard_deviation * 9.0 / 5.0
    return standard_deviation


def convert_profile_height(record: VerticalProfileRecord, unit_system: str) -> float:
    return record.height_ft if unit_system == "ip" else record.height_ft * FT_TO_M


def records_have_plot_standard_deviation(records: list[VerticalProfileRecord], profile_mode: str) -> bool:
    return any(
        standard_deviation is not None and standard_deviation > 0
        for standard_deviation in (
            profile_plot_standard_deviation(record, profile_mode)
            for record in records
        )
    )


def plot_vertical_profiles(
    records: list[VerticalProfileRecord],
    selected_variables: list[str],
    profile_mode: str,
    contaminant_name: str = "Contaminant",
    concentration_unit: str = "",
    show_error_bars: bool = True,
    unit_system: str = "si",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panel_count = len(selected_variables)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(4.4 * panel_count, 5.6),
        sharey=True,
        squeeze=False,
    )
    axes_list = list(axes[0])

    for axis, variable in zip(axes_list, selected_variables):
        for position_number in sorted(
            {
                record.position_number
                for record in records
                if record.variable == variable
            }
        ):
            position_records = sorted(
                [
                    record
                    for record in records
                    if record.variable == variable and record.position_number == position_number
                ],
                key=lambda record: record.height_ft,
            )
            x_values = [convert_profile_value(record, unit_system) for record in position_records]
            y_values = [convert_profile_height(record, unit_system) for record in position_records]
            x_errors = (
                [
                    convert_profile_standard_deviation(record, profile_mode, unit_system)
                    for record in position_records
                ]
                if show_error_bars
                else [None for _record in position_records]
            )
            axis.errorbar(
                x_values,
                y_values,
                xerr=[
                    error if error is not None and pd.notna(error) else 0.0
                    for error in x_errors
                ] if show_error_bars and any(error is not None and pd.notna(error) for error in x_errors) else None,
                marker="o",
                linestyle="-",
                linewidth=1.4,
                markersize=4,
                color=POSITION_STYLE_COLORS[position_number],
                ecolor=POSITION_STYLE_COLORS[position_number],
                elinewidth=0.8,
                capsize=3,
                label=f"P{position_number}",
            )
        axis.set_xlabel(x_axis_label(variable, profile_mode, contaminant_name, concentration_unit, unit_system))
        axis.set_title(subplot_title(variable, contaminant_name), pad=10)
        axis_limit = VERTICAL_PROFILE_AXIS_LIMITS.get((unit_system, variable))
        if axis_limit is not None:
            axis.set_xlim(*axis_limit)
        axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.8)
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_color("#111827")

    axes_list[0].set_ylabel(y_axis_label(profile_mode, unit_system))
    axes_list[0].set_ylim(0.0, ROOM_HEIGHT_FT if unit_system == "ip" else ROOM_HEIGHT_M)
    handles = [
        Line2D([0], [0], color=color, marker="o", linewidth=1.4, label=f"P{position_number}")
        for position_number, color in POSITION_STYLE_COLORS.items()
        if any(record.position_number == position_number for record in records)
    ]
    fig.suptitle(profile_figure_title(profile_mode, unit_system), y=0.98, fontsize=14)
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=min(len(handles), 6),
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    return fig


def export_vertical_profile_eps(fig) -> bytes:
    output = BytesIO()
    fig.savefig(output, format="eps", dpi=300, bbox_inches="tight", facecolor="white")
    return output.getvalue()


def export_vertical_profile_png(fig) -> bytes:
    output = BytesIO()
    fig.savefig(output, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    return output.getvalue()


def vertical_profile_summary_csv(
    records: list[VerticalProfileRecord],
    selected_variables: list[str] | None = None,
) -> bytes:
    return vertical_profile_records_to_unified_dataframe(records, selected_variables).to_csv(index=False).encode("utf-8-sig")


def vertical_profile_summary_xlsx(
    records: list[VerticalProfileRecord],
    selected_variables: list[str] | None = None,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        vertical_profile_records_to_unified_dataframe(records, selected_variables).to_excel(
            writer,
            index=False,
            sheet_name="Summary",
        )
    return output.getvalue()


def build_vertical_profile_zip(files: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for filename, file_bytes in files:
            archive.writestr(filename, file_bytes)
    return output.getvalue()
