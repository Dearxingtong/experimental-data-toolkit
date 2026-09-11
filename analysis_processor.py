from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import cos, radians, sin, sqrt

import pandas as pd


FT_TO_M = 0.3048
M_PER_S_TO_FT_PER_S = 3.280839895
ROOM_DIMENSIONS = (8.0, 10.0, 8.75)
POSITION_COORDINATES = {
    1: (2.67, 2.5),
    2: (2.67, 5.0),
    3: (2.67, 7.5),
    4: (5.33, 2.5),
    5: (5.33, 5.0),
    6: (5.33, 7.5),
}
ANALYSIS_REQUIRED_COLUMNS = (
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
MAX_ARROW_LENGTH_FRACTION = 0.12
VIEW_ELEVATION = 15
VIEW_AZIMUTH = 10
VIEW_ROLL = 0
VELOCITY_COLORMAP = "turbo"


@dataclass
class MeanVector:
    height_index: int
    vx: float
    vy: float
    vz: float
    magnitude: float


@dataclass
class VectorRecord:
    position_number: int
    height_index: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    magnitude: float
    source_filename: str


@dataclass(frozen=True)
class UnitSpec:
    system: str
    title: str
    axis_unit: str
    velocity_unit: str
    coordinate_scale: float
    velocity_scale: float


UNIT_SPECS = {
    "imperial": UnitSpec(
        system="imperial",
        title="3D Air Velocity Vector Map - Imperial Units",
        axis_unit="ft",
        velocity_unit="ft/s",
        coordinate_scale=1.0,
        velocity_scale=M_PER_S_TO_FT_PER_S,
    ),
    "si": UnitSpec(
        system="si",
        title="3D Air Velocity Vector Map - SI Units",
        axis_unit="m",
        velocity_unit="m/s",
        coordinate_scale=FT_TO_M,
        velocity_scale=1.0,
    ),
}


def validate_analysis_csv(dataframe: pd.DataFrame, filename: str = "uploaded CSV") -> str | None:
    if dataframe.empty:
        return f"{filename} is empty or does not contain airflow data."

    missing_columns = [
        column for column in ANALYSIS_REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        return (
            f"{filename} is missing required airflow columns: "
            f"{', '.join(missing_columns)}"
        )

    for column in ANALYSIS_REQUIRED_COLUMNS:
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        if converted.dropna().empty:
            return f"{filename} column {column} does not contain numeric values."

    return None


def validate_heights(heights: list[float]) -> str | None:
    room_height = ROOM_DIMENSIONS[2]
    for index, height in enumerate(heights, start=1):
        try:
            parsed_height = float(height)
        except (TypeError, ValueError):
            return f"Height {index} must be numeric."
        if parsed_height < 0 or parsed_height > room_height:
            return f"Height {index} must be between 0 and {room_height} ft."
    return None


def position_to_xy(position_number: int) -> tuple[float, float]:
    if position_number not in POSITION_COORDINATES:
        raise ValueError("Position Number must be between 1 and 6.")
    return POSITION_COORDINATES[position_number]


def room_dimensions(unit_system: str = "imperial") -> tuple[float, float, float]:
    unit_spec = UNIT_SPECS[unit_system]
    return tuple(dimension * unit_spec.coordinate_scale for dimension in ROOM_DIMENSIONS)


def position_to_xy_in_units(position_number: int, unit_system: str = "imperial") -> tuple[float, float]:
    unit_spec = UNIT_SPECS[unit_system]
    x_coordinate, y_coordinate = position_to_xy(position_number)
    return x_coordinate * unit_spec.coordinate_scale, y_coordinate * unit_spec.coordinate_scale


def validate_position_coordinates() -> None:
    expected_coordinates = {
        1: (2.67, 2.5),
        2: (2.67, 5.0),
        3: (2.67, 7.5),
        4: (5.33, 2.5),
        5: (5.33, 5.0),
        6: (5.33, 7.5),
    }
    if POSITION_COORDINATES != expected_coordinates:
        raise ValueError("Position coordinate mapping does not match the room layout.")


def normalized_direction(vx: float, vy: float, vz: float, magnitude: float) -> tuple[float, float, float]:
    if magnitude <= 0:
        return 0.0, 0.0, 0.0
    return vx / magnitude, vy / magnitude, vz / magnitude


def converted_record_values(record: VectorRecord, unit_system: str) -> dict[str, float]:
    unit_spec = UNIT_SPECS[unit_system]
    return {
        "x": record.x * unit_spec.coordinate_scale,
        "y": record.y * unit_spec.coordinate_scale,
        "z": record.z * unit_spec.coordinate_scale,
        "vx": record.vx * unit_spec.velocity_scale,
        "vy": record.vy * unit_spec.velocity_scale,
        "vz": record.vz * unit_spec.velocity_scale,
        "magnitude": record.magnitude * unit_spec.velocity_scale,
    }


def direction_matches_after_unit_conversion(record: VectorRecord, tolerance: float = 1e-12) -> bool:
    si_direction = normalized_direction(record.vx, record.vy, record.vz, record.magnitude)
    imperial_values = converted_record_values(record, "imperial")
    imperial_direction = normalized_direction(
        imperial_values["vx"],
        imperial_values["vy"],
        imperial_values["vz"],
        imperial_values["magnitude"],
    )
    return all(
        abs(si_value - imperial_value) <= tolerance
        for si_value, imperial_value in zip(si_direction, imperial_direction)
    )


def camera_eye_from_angles(
    azimuth: float = VIEW_AZIMUTH,
    elevation: float = VIEW_ELEVATION,
    radius: float = 1.85,
) -> dict[str, float]:
    azimuth_rad = radians(azimuth)
    elevation_rad = radians(elevation)
    return {
        "x": radius * cos(elevation_rad) * cos(azimuth_rad),
        "y": radius * cos(elevation_rad) * sin(azimuth_rad),
        "z": radius * sin(elevation_rad),
    }


def camera_up_from_roll(roll: float = VIEW_ROLL) -> dict[str, float]:
    roll_rad = radians(roll)
    return {
        "x": sin(roll_rad),
        "y": 0.0,
        "z": cos(roll_rad),
    }


def apply_matplotlib_camera(ax, elevation: float, azimuth: float, roll: float) -> None:
    try:
        ax.view_init(elev=elevation, azim=azimuth, roll=roll)
    except TypeError:
        ax.view_init(elev=elevation, azim=azimuth)


def compute_mean_vectors(dataframe: pd.DataFrame) -> list[MeanVector]:
    mean_vectors: list[MeanVector] = []
    for height_index in (1, 2, 3):
        vx = pd.to_numeric(dataframe[f"Vx{height_index}"], errors="coerce").mean()
        vy = pd.to_numeric(dataframe[f"Vy{height_index}"], errors="coerce").mean()
        vz = pd.to_numeric(dataframe[f"Vz{height_index}"], errors="coerce").mean()
        magnitude = sqrt(vx**2 + vy**2 + vz**2)
        mean_vectors.append(MeanVector(height_index, vx, vy, vz, magnitude))
    return mean_vectors


def build_vector_records(
    position_number: int,
    heights: list[float],
    mean_vectors: list[MeanVector],
    source_filename: str,
) -> list[VectorRecord]:
    if len(heights) != 3 or len(mean_vectors) != 3:
        raise ValueError("Exactly three heights and three mean vectors are required.")

    x, y = position_to_xy(position_number)
    records: list[VectorRecord] = []
    for height, mean_vector in zip(heights, mean_vectors):
        records.append(
            VectorRecord(
                position_number=position_number,
                height_index=mean_vector.height_index,
                x=x,
                y=y,
                z=float(height),
                vx=mean_vector.vx,
                vy=mean_vector.vy,
                vz=mean_vector.vz,
                magnitude=mean_vector.magnitude,
                source_filename=source_filename,
            )
        )
    return records


def duplicate_position_height_errors(
    existing_records: list[VectorRecord],
    new_records: list[VectorRecord],
) -> list[str]:
    existing_keys = {
        (record.position_number, round(record.z, 6))
        for record in existing_records
    }
    errors = []
    new_keys = set()
    for record in new_records:
        key = (record.position_number, round(record.z, 6))
        if key in existing_keys:
            errors.append(
                f"P{record.position_number}-H{record.height_index} duplicates an existing "
                f"position + height combination at {record.z:g} ft."
            )
        if key in new_keys:
            errors.append(
                f"P{record.position_number}-H{record.height_index} duplicates another "
                f"height in this upload at {record.z:g} ft."
            )
        new_keys.add(key)
    return errors


def vector_records_to_dataframe(vector_records: list[VectorRecord], unit_system: str = "si") -> pd.DataFrame:
    unit_spec = UNIT_SPECS[unit_system]
    return pd.DataFrame(
        [
            {
                "Position": f"P{record.position_number}",
                "Height ID": f"H{record.height_index}",
                f"X ({unit_spec.axis_unit})": converted_record_values(record, unit_system)["x"],
                f"Y ({unit_spec.axis_unit})": converted_record_values(record, unit_system)["y"],
                f"Z ({unit_spec.axis_unit})": converted_record_values(record, unit_system)["z"],
                f"Mean Vx ({unit_spec.velocity_unit})": converted_record_values(record, unit_system)["vx"],
                f"Mean Vy ({unit_spec.velocity_unit})": converted_record_values(record, unit_system)["vy"],
                f"Mean Vz ({unit_spec.velocity_unit})": converted_record_values(record, unit_system)["vz"],
                f"Velocity magnitude ({unit_spec.velocity_unit})": converted_record_values(
                    record,
                    unit_system,
                )["magnitude"],
                "Source File": record.source_filename,
            }
            for record in vector_records
        ]
    )


def display_vector_components(
    vector_records: list[VectorRecord],
    unit_system: str = "imperial",
) -> tuple[list[float], list[float], list[float]]:
    """Scale true vector directions by relative magnitude for readable room-scale plots."""
    if not vector_records:
        return [], [], []

    converted_values = [converted_record_values(record, unit_system) for record in vector_records]
    maximum_magnitude = max(value["magnitude"] for value in converted_values)
    if maximum_magnitude <= 0:
        return (
            [0.0 for _record in vector_records],
            [0.0 for _record in vector_records],
            [0.0 for _record in vector_records],
        )

    max_arrow_length = max(room_dimensions(unit_system)) * MAX_ARROW_LENGTH_FRACTION
    display_us: list[float] = []
    display_vs: list[float] = []
    display_ws: list[float] = []
    for value in converted_values:
        if value["magnitude"] <= 0:
            display_us.append(0.0)
            display_vs.append(0.0)
            display_ws.append(0.0)
            continue
        direction = normalized_direction(
            value["vx"],
            value["vy"],
            value["vz"],
            value["magnitude"],
        )
        display_length = value["magnitude"] / maximum_magnitude * max_arrow_length
        display_us.append(direction[0] * display_length)
        display_vs.append(direction[1] * display_length)
        display_ws.append(direction[2] * display_length)
    return display_us, display_vs, display_ws


def magnitude_range(vector_records: list[VectorRecord], unit_system: str) -> tuple[float, float]:
    magnitudes = [
        converted_record_values(record, unit_system)["magnitude"]
        for record in vector_records
    ]
    if not magnitudes:
        return 0.0, 1.0
    minimum = min(magnitudes)
    maximum = max(magnitudes)
    if minimum == maximum:
        maximum = minimum + 1.0
    return minimum, maximum


def room_edge_coordinates(unit_system: str) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    x_max, y_max, z_max = room_dimensions(unit_system)
    return [
        ((0, 0, 0), (x_max, 0, 0)),
        ((0, y_max, 0), (x_max, y_max, 0)),
        ((0, 0, 0), (0, y_max, 0)),
        ((x_max, 0, 0), (x_max, y_max, 0)),
        ((0, 0, z_max), (x_max, 0, z_max)),
        ((0, y_max, z_max), (x_max, y_max, z_max)),
        ((0, 0, z_max), (0, y_max, z_max)),
        ((x_max, 0, z_max), (x_max, y_max, z_max)),
        ((0, 0, 0), (0, 0, z_max)),
        ((x_max, 0, 0), (x_max, 0, z_max)),
        ((0, y_max, 0), (0, y_max, z_max)),
        ((x_max, y_max, 0), (x_max, y_max, z_max)),
    ]


def plot_interactive_air_vectors(
    vector_records: list[VectorRecord],
    unit_system: str = "si",
    azimuth: float = VIEW_AZIMUTH,
    elevation: float = VIEW_ELEVATION,
    roll: float = VIEW_ROLL,
):
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale

    validate_position_coordinates()
    unit_spec = UNIT_SPECS[unit_system]
    x_max, y_max, z_max = room_dimensions(unit_system)
    magnitude_min, magnitude_max = magnitude_range(vector_records, unit_system)
    fig = go.Figure()

    for start, end in room_edge_coordinates(unit_system):
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line={"color": "#9ca3af", "width": 3},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    display_us, display_vs, display_ws = display_vector_components(vector_records, unit_system)
    origin_x = []
    origin_y = []
    origin_z = []
    arrowhead_x = []
    arrowhead_y = []
    arrowhead_z = []
    arrowhead_magnitudes = []
    for record, display_u, display_v, display_w in zip(
        vector_records,
        display_us,
        display_vs,
        display_ws,
    ):
        value = converted_record_values(record, unit_system)
        origin_x.append(value["x"])
        origin_y.append(value["y"])
        origin_z.append(value["z"])
        end_x = value["x"] + display_u
        end_y = value["y"] + display_v
        end_z = value["z"] + display_w
        normalized_magnitude = (
            (value["magnitude"] - magnitude_min) / (magnitude_max - magnitude_min)
            if magnitude_max > magnitude_min
            else 0.0
        )
        arrow_color = sample_colorscale(VELOCITY_COLORMAP, [normalized_magnitude])[0]
        fig.add_trace(
            go.Scatter3d(
                x=[value["x"], end_x],
                y=[value["y"], end_y],
                z=[value["z"], end_z],
                mode="lines",
                line={"color": arrow_color, "width": 4},
                showlegend=False,
                hovertext=(
                    f"P{record.position_number}-H{record.height_index}<br>"
                    f"{value['magnitude']:.6g} {unit_spec.velocity_unit}"
                ),
                hoverinfo="text",
            )
        )
        if display_u != 0 or display_v != 0 or display_w != 0:
            fig.add_trace(
                go.Cone(
                    x=[end_x],
                    y=[end_y],
                    z=[end_z],
                    u=[display_u],
                    v=[display_v],
                    w=[display_w],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=max(room_dimensions(unit_system)) * 0.035,
                    colorscale=[[0, arrow_color], [1, arrow_color]],
                    showscale=False,
                    showlegend=False,
                    hovertext=[
                        (
                            f"P{record.position_number}-H{record.height_index}<br>"
                            f"{value['magnitude']:.6g} {unit_spec.velocity_unit}"
                        )
                    ],
                    hoverinfo="text",
                )
            )
        arrowhead_x.append(end_x)
        arrowhead_y.append(end_y)
        arrowhead_z.append(end_z)
        arrowhead_magnitudes.append(value["magnitude"])

    if vector_records:
        fig.add_trace(
            go.Scatter3d(
                x=origin_x,
                y=origin_y,
                z=origin_z,
                mode="markers",
                marker={"size": 3, "color": "#000000"},
                name="Vector origins",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=arrowhead_x,
                y=arrowhead_y,
                z=arrowhead_z,
                mode="markers",
                marker={
                    "size": 1,
                    "opacity": 0,
                    "color": arrowhead_magnitudes,
                    "colorscale": VELOCITY_COLORMAP,
                    "cmin": magnitude_min,
                    "cmax": magnitude_max,
                    "colorbar": {"title": f"Velocity magnitude ({unit_spec.velocity_unit})"},
                    "showscale": True,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"{unit_spec.title} - Interactive Preview",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=650,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        scene={
            "xaxis": {"title": f"X ({unit_spec.axis_unit})", "range": [0, x_max]},
            "yaxis": {"title": f"Y ({unit_spec.axis_unit})", "range": [0, y_max]},
            "zaxis": {"title": f"Z ({unit_spec.axis_unit})", "range": [0, z_max]},
            "aspectmode": "manual",
            "aspectratio": {"x": x_max, "y": y_max, "z": z_max},
            "camera": {
                "projection": {"type": "orthographic"},
                "eye": camera_eye_from_angles(azimuth, elevation),
                "up": camera_up_from_roll(roll),
            },
        },
    )
    return fig


def plot_3d_air_vectors(
    vector_records: list[VectorRecord],
    unit_system: str = "imperial",
    azimuth: float = VIEW_AZIMUTH,
    elevation: float = VIEW_ELEVATION,
    roll: float = VIEW_ROLL,
    show_vector_labels: bool = False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    fig = plt.figure(figsize=(10.5, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    validate_position_coordinates()
    unit_spec = UNIT_SPECS[unit_system]

    x_max, y_max, z_max = room_dimensions(unit_system)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_zlim(0, z_max)
    ax.set_xlabel(f"X ({unit_spec.axis_unit})")
    ax.set_ylabel(f"Y ({unit_spec.axis_unit})")
    ax.set_zlabel(f"Z ({unit_spec.axis_unit})")
    ax.set_title(unit_spec.title, pad=18)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((x_max, y_max, z_max))
    apply_matplotlib_camera(ax, elevation, azimuth, roll)

    x_grid = [
        0,
        position_to_xy_in_units(1, unit_system)[0],
        position_to_xy_in_units(4, unit_system)[0],
        x_max,
    ]
    y_grid = [
        0,
        position_to_xy_in_units(1, unit_system)[1],
        position_to_xy_in_units(2, unit_system)[1],
        position_to_xy_in_units(3, unit_system)[1],
        y_max,
    ]
    for x_coordinate in x_grid:
        ax.plot(
            [x_coordinate, x_coordinate],
            [0, y_max],
            [0, 0],
            color="#d1d5db",
            linewidth=0.6,
            alpha=0.65,
        )
    for y_coordinate in y_grid:
        ax.plot(
            [0, x_max],
            [y_coordinate, y_coordinate],
            [0, 0],
            color="#d1d5db",
            linewidth=0.6,
            alpha=0.65,
        )

    for start, end in room_edge_coordinates(unit_system):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="#9ca3af",
            linewidth=0.9,
            alpha=0.75,
        )

    if vector_records:
        converted_values = [converted_record_values(record, unit_system) for record in vector_records]
        xs = [value["x"] for value in converted_values]
        ys = [value["y"] for value in converted_values]
        zs = [value["z"] for value in converted_values]
        us, vs, ws = display_vector_components(vector_records, unit_system)
        magnitudes = [value["magnitude"] for value in converted_values]
        magnitude_min, magnitude_max = magnitude_range(vector_records, unit_system)
        norm = colors.Normalize(vmin=magnitude_min, vmax=magnitude_max)
        colormap = plt.get_cmap(VELOCITY_COLORMAP)
        arrow_colors = [colormap(norm(magnitude)) for magnitude in magnitudes]

        ax.scatter(
            xs,
            ys,
            zs,
            marker="o",
            s=14,
            color="#000000",
            depthshade=False,
        )
        ax.quiver(
            xs,
            ys,
            zs,
            us,
            vs,
            ws,
            length=1.0,
            normalize=False,
            color=arrow_colors,
            arrow_length_ratio=0.22,
            linewidth=1.4,
        )
        scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap)
        scalar_map.set_array(magnitudes)
        colorbar = fig.colorbar(scalar_map, ax=ax, shrink=0.68, pad=0.08)
        colorbar.set_label(f"Velocity magnitude ({unit_spec.velocity_unit})")

    fig.text(
        0.5,
        0.02,
        "Arrow direction represents mean airflow direction. Arrow length is proportional to "
        "velocity magnitude and scaled for visualization. Colors indicate velocity magnitude.",
        ha="center",
        fontsize=9,
    )
    ax.grid(True, alpha=0.35)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def export_figure_eps(fig) -> bytes:
    output = BytesIO()
    fig.savefig(output, format="eps", bbox_inches="tight", facecolor="white")
    return output.getvalue()


def export_figure_png(fig) -> bytes:
    output = BytesIO()
    fig.savefig(output, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    return output.getvalue()


def close_figure(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)
