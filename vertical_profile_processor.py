from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import sqrt
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


PROFILE_VARIABLES = ("Air Velocity", "Temperature", "Contaminant Concentration")
PROFILE_MODES = ("Normalized Profiles", "Raw Profiles")
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


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if not pd.to_numeric(dataframe[column], errors="coerce").dropna().empty
    ]


def validate_profile_heights(heights: list[float], room_height_ft: float) -> list[str]:
    errors: list[str] = []
    if room_height_ft <= 0:
        errors.append("Room height must be greater than 0 ft.")
    for index, height in enumerate(heights, start=1):
        try:
            parsed_height = float(height)
        except (TypeError, ValueError):
            errors.append(f"Height {index} must be numeric.")
            continue
        if parsed_height <= 0:
            errors.append(f"Height {index} must be greater than 0 ft.")
        if room_height_ft > 0 and parsed_height > room_height_ft:
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
    if profile_mode != "Normalized Profiles":
        return []

    errors: list[str] = []
    if "Air Velocity" in selected_variables and (supply_velocity is None or supply_velocity <= 0):
        errors.append("Supply Air Velocity Us must be greater than 0 m/s.")
    if "Temperature" in selected_variables:
        if tin is None or tout is None:
            errors.append("Tin and Tout are required for normalized temperature.")
        elif tin == tout:
            errors.append("Tout must not equal Tin for normalized temperature.")
    if "Contaminant Concentration" in selected_variables:
        if cin is None or cout is None:
            errors.append("Cin and Cout are required for normalized concentration.")
        elif cin == cout:
            errors.append("Cout must not equal Cin for normalized concentration.")
    return errors


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
    if profile_mode not in PROFILE_MODES:
        errors.append("Profile mode must be Raw Profiles or Normalized Profiles.")
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
            normalized_value = None
            if profile_mode == "Normalized Profiles":
                if variable == "Air Velocity":
                    normalized_value = normalize_velocity(mean, float(supply_velocity))
                elif variable == "Temperature":
                    normalized_value = normalize_temperature(mean, float(tin), float(tout))
                elif variable == "Contaminant Concentration":
                    normalized_value = normalize_concentration(mean, float(cin), float(cout))

            records.append(
                VerticalProfileRecord(
                    dataset_type=dataset_type,
                    position_number=position_number,
                    height_id=f"H{height_index}",
                    height_ft=float(height_ft),
                    normalized_height=normalize_height(float(height_ft), room_height_ft),
                    variable=variable,
                    mean=mean,
                    standard_deviation=standard_deviation,
                    sample_count=sample_count,
                    normalized_value=normalized_value,
                    source_file=source_file,
                    unit=unit,
                    contaminant_name=contaminant_name if variable == "Contaminant Concentration" else "",
                    source_column=source_columns[height_index - 1],
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
    if profile_mode not in PROFILE_MODES:
        errors.append("Profile mode must be Raw Profiles or Normalized Profiles.")
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
    source_builders = {
        "Air Velocity": (
            compute_air_speed_profile,
            "m/s",
            ["speed"] * 3,
        ),
        "Temperature": (
            lambda dataframe: compute_temperature_profile(
                dataframe,
                temperature_columns or list(TEMPERATURE_DEFAULT_COLUMNS),
            ),
            "deg C",
            temperature_columns or list(TEMPERATURE_DEFAULT_COLUMNS),
        ),
        "Contaminant Concentration": (
            lambda dataframe: compute_contaminant_profile(dataframe, contaminant_columns or []),
            concentration_unit,
            contaminant_columns or [],
        ),
    }

    for variable in selected_variables:
        dataframe = dataframes_by_variable.get(variable)
        if dataframe is None:
            continue

        profile_builder, unit, source_columns = source_builders[variable]
        profile, profile_errors = profile_builder(dataframe)
        if profile_errors:
            errors.extend(f"{variable}: {error}" for error in profile_errors)
            continue

        for height_index, (height_ft, stats) in enumerate(zip(heights_ft, profile), start=1):
            mean, standard_deviation, sample_count = stats
            normalized_value = None
            if profile_mode == "Normalized Profiles":
                if variable == "Air Velocity":
                    normalized_value = normalize_velocity(mean, float(supply_velocity))
                elif variable == "Temperature":
                    normalized_value = normalize_temperature(mean, float(tin), float(tout))
                elif variable == "Contaminant Concentration":
                    normalized_value = normalize_concentration(mean, float(cin), float(cout))

            records.append(
                VerticalProfileRecord(
                    dataset_type=dataset_type,
                    position_number=position_number,
                    height_id=f"H{height_index}",
                    height_ft=float(height_ft),
                    normalized_height=normalize_height(float(height_ft), room_height_ft),
                    variable=variable,
                    mean=mean,
                    standard_deviation=standard_deviation,
                    sample_count=sample_count,
                    normalized_value=normalized_value,
                    source_file=source_files_by_variable.get(variable, ""),
                    unit=unit,
                    contaminant_name=contaminant_name if variable == "Contaminant Concentration" else "",
                    source_column=source_columns[height_index - 1] if len(source_columns) >= height_index else "",
                )
            )

    records.sort(key=lambda record: (record.variable, record.position_number, record.height_ft))
    if not records and not errors:
        errors.append("Upload at least one source CSV for the selected variables.")
    return records, errors


def vertical_profile_records_to_dataframe(records: list[VerticalProfileRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Dataset Type": record.dataset_type,
                "Position": f"P{record.position_number}",
                "Height ID": record.height_id,
                "Height (ft)": record.height_ft,
                "Normalized Height": record.normalized_height,
                "Variable": record.variable,
                "Mean": record.mean,
                "Standard Deviation": record.standard_deviation,
                "Sample Count": record.sample_count,
                "Normalized Value": record.normalized_value,
                "Unit": record.unit,
                "Contaminant Name": record.contaminant_name,
                "Source Column": record.source_column,
                "Source File": record.source_file,
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
                "Normalized Height": record.normalized_height,
            },
        )
        prefix = record.variable
        row[f"{prefix} Mean"] = record.mean
        row[f"{prefix} Standard Deviation"] = record.standard_deviation
        row[f"{prefix} Sample Count"] = record.sample_count
        row[f"{prefix} Normalized Value"] = record.normalized_value
        row[f"{prefix} Unit"] = record.unit
        row[f"{prefix} Source File"] = record.source_file
        row[f"{prefix} Source Column"] = record.source_column

    rows = [grouped[key] for key in sorted(grouped, key=lambda item: (item[0], item[2]))]
    dataframe = pd.DataFrame(rows)
    base_columns = ["Dataset Type", "Position", "Height ID", "Height (ft)", "Normalized Height"]
    variable_columns: list[str] = []
    for variable in variables:
        variable_columns.extend(
            [
                f"{variable} Mean",
                f"{variable} Standard Deviation",
                f"{variable} Sample Count",
                f"{variable} Normalized Value",
                f"{variable} Unit",
                f"{variable} Source File",
                f"{variable} Source Column",
            ]
        )
    for column in base_columns + variable_columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA
    return dataframe[base_columns + variable_columns]


def x_axis_label(variable: str, profile_mode: str, contaminant_name: str = "Contaminant", concentration_unit: str = "") -> str:
    if profile_mode == "Normalized Profiles":
        if variable == "Air Velocity":
            return "Normalized velocity, U*"
        if variable == "Temperature":
            return "Normalized temperature, theta"
        return "Normalized concentration, C*"
    if variable == "Air Velocity":
        return "Air velocity (m/s)"
    if variable == "Temperature":
        return "Temperature (deg C)"
    unit_text = f" ({concentration_unit})" if concentration_unit else ""
    return f"{contaminant_name} concentration{unit_text}"


def y_axis_label(profile_mode: str) -> str:
    if profile_mode == "Normalized Profiles":
        return "Normalized height, Z = z/H"
    return "Height (ft)"


def profile_figure_title(profile_mode: str) -> str:
    if profile_mode == "Normalized Profiles":
        return "Normalized Vertical Profiles"
    return "Vertical Profiles"


def subplot_title(variable: str, contaminant_name: str = "Contaminant") -> str:
    if variable == "Contaminant Concentration":
        return f"{contaminant_name} Concentration"
    return variable


def plot_vertical_profiles(
    records: list[VerticalProfileRecord],
    selected_variables: list[str],
    profile_mode: str,
    contaminant_name: str = "Contaminant",
    concentration_unit: str = "",
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
    records_frame = vertical_profile_records_to_dataframe(records)

    for axis, variable in zip(axes_list, selected_variables):
        variable_frame = records_frame[records_frame["Variable"] == variable].copy()
        for position_number in sorted(
            int(position.replace("P", ""))
            for position in variable_frame["Position"].drop_duplicates()
        ):
            position_frame = variable_frame[variable_frame["Position"] == f"P{position_number}"].sort_values("Height (ft)")
            x_values = (
                position_frame["Normalized Value"]
                if profile_mode == "Normalized Profiles"
                else position_frame["Mean"]
            )
            y_values = (
                position_frame["Normalized Height"]
                if profile_mode == "Normalized Profiles"
                else position_frame["Height (ft)"]
            )
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.4,
                markersize=4,
                color=POSITION_STYLE_COLORS[position_number],
                label=f"P{position_number}",
            )
        axis.set_xlabel(x_axis_label(variable, profile_mode, contaminant_name, concentration_unit))
        axis.set_title(subplot_title(variable, contaminant_name), pad=10)
        axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.8)
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_color("#111827")

    axes_list[0].set_ylabel(y_axis_label(profile_mode))
    if profile_mode == "Normalized Profiles":
        axes_list[0].set_ylim(0, 1)

    handles = [
        Line2D([0], [0], color=color, marker="o", linewidth=1.4, label=f"P{position_number}")
        for position_number, color in POSITION_STYLE_COLORS.items()
        if any(record.position_number == position_number for record in records)
    ]
    fig.suptitle(profile_figure_title(profile_mode), y=0.98, fontsize=14)
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
