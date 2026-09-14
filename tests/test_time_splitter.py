from __future__ import annotations

from datetime import date, time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import importlib.util
import json
import sys
import types

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import analysis_processor as analysis
import dat_batch
import dat_processor as dat
import project_io
import replicate_processor as replicate
import time_splitter as splitter
import vertical_profile_processor as vertical


def make_workbook_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            dataframe.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


def load_app_module():
    sys.modules["streamlit"] = types.SimpleNamespace()
    spec = importlib.util.spec_from_file_location("toolkit_app", PROJECT_ROOT / "app.py")
    app = importlib.util.module_from_spec(spec)
    sys.modules["toolkit_app"] = app
    spec.loader.exec_module(app)
    return app


def analyze_dataframe(dataframe: pd.DataFrame, filename: str = "P1_Round1.xlsx"):
    workbook_bytes = make_workbook_bytes({"Cleaned_Data": dataframe})
    analysis = splitter.analyze_workbook_bytes(workbook_bytes, filename)
    assert analysis.errors == []
    return analysis


def duration_range(label: str, start_time: time, duration_minutes: int) -> splitter.PositionRange:
    position_range, error = splitter.build_position_range_from_duration(
        label,
        start_time,
        duration_minutes,
    )
    assert error is None
    return position_range


def dated_duration_range(
    label: str,
    selected_date: date,
    start_time: time,
    duration_minutes: int,
    position_number: int,
) -> splitter.PositionRange:
    position_range, error = splitter.build_position_range_from_duration(
        label,
        start_time,
        duration_minutes,
        selected_date=selected_date,
        position_number=position_number,
    )
    assert error is None
    return position_range


def assert_position_conditions(result, boundary1, boundary2):
    workbook1 = pd.read_excel(BytesIO(result.output_files[0][1]), sheet_name=None, engine="openpyxl")
    workbook2 = pd.read_excel(BytesIO(result.output_files[1][1]), sheet_name=None, engine="openpyxl")
    workbook3 = pd.read_excel(BytesIO(result.output_files[2][1]), sheet_name=None, engine="openpyxl")
    part1 = workbook1["Cleaned_Data"]
    part2 = workbook2["Cleaned_Data"]
    part3 = workbook3["Cleaned_Data"]

    ts1 = pd.to_datetime(part1["TIMESTAMP"], format="mixed")
    ts2 = pd.to_datetime(part2["TIMESTAMP"], format="mixed")
    ts3 = pd.to_datetime(part3["TIMESTAMP"], format="mixed")

    assert (ts1 < boundary1).all()
    assert ((ts2 >= boundary1) & (ts2 < boundary2)).all()
    assert (ts3 >= boundary2).all()


def test_standard_split():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-07-10 13:00:00",
                "2026-07-10 13:38:59.950",
                "2026-07-10 13:39:00.020",
                "2026-07-10 15:49:59.980",
                "2026-07-10 15:50:00.030",
                "2026-07-10 17:00:00",
            ],
            "U1": [1, 2, 3, 4, 5, 6],
        }
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook(analysis, time(13, 38), time(15, 49))
    assert error is None
    assert result.boundary1 == pd.Timestamp("2026-07-10 13:39:00")
    assert result.boundary2 == pd.Timestamp("2026-07-10 15:50:00")
    assert_position_conditions(result, result.boundary1, result.boundary2)


def test_different_user_defined_times_are_dynamic():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-07-10 13:00:00",
                "2026-07-10 14:02:59.999",
                "2026-07-10 14:03:00.001",
                "2026-07-10 16:17:59.999",
                "2026-07-10 16:18:00.001",
                "2026-07-10 17:00:00",
            ],
            "U1": [1, 2, 3, 4, 5, 6],
        }
    )
    analysis = analyze_dataframe(dataframe, "P1_Round2.xlsx")
    result, error = splitter.split_workbook(analysis, time(14, 2), time(16, 17))
    assert error is None
    assert result.boundary1 == pd.Timestamp("2026-07-10 14:03:00")
    assert result.boundary2 == pd.Timestamp("2026-07-10 16:18:00")
    assert_position_conditions(result, result.boundary1, result.boundary2)


def test_no_exact_boundary_observation():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-07-10 13:38:59.970",
                "2026-07-10 13:39:00.020",
                "2026-07-10 13:40:00.020",
            ],
            "U1": [1, 2, 3],
        }
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook(analysis, time(13, 38), time(13, 39))
    assert error is None
    assert result.position_summaries[0].rows == 1
    assert result.position_summaries[1].rows == 1
    assert result.position_summaries[2].rows == 1


def test_row_accounting_and_no_duplicates():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2026-07-10 13:00:00", periods=12, freq="30min"),
            "U1": range(12),
        }
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook(analysis, time(13, 59), time(16, 59))
    assert error is None
    qc = result.qc_results[0]
    assert qc.total_rows == qc.original_rows
    assert qc.all_rows_accounted_for
    assert qc.no_duplicate_assignments


def test_invalid_cut_order():
    dataframe = pd.DataFrame(
        {"TIMESTAMP": pd.date_range("2026-07-10 13:00:00", periods=6, freq="1h")}
    )
    result, error = splitter.split_workbook(analyze_dataframe(dataframe), time(15, 49), time(13, 38))
    assert result is None
    assert error == "End of Position 2 must be later than End of Position 1."


def test_cut_outside_experimental_range():
    dataframe = pd.DataFrame(
        {"TIMESTAMP": pd.date_range("2026-07-10 13:00:00", periods=6, freq="1h")}
    )
    result, error = splitter.split_workbook(analyze_dataframe(dataframe), time(12, 30), time(15, 0))
    assert result is None
    assert error == "Selected cut time is outside the experimental time range."


def test_empty_segment_rejected():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-07-10 13:00:00",
                "2026-07-10 17:00:00",
            ]
        }
    )
    result, error = splitter.split_workbook(analyze_dataframe(dataframe), time(13, 0), time(13, 1))
    assert result is None
    assert error == "Position 2 would be empty with the selected cut times."


def test_missing_timestamp_is_graceful():
    workbook_bytes = make_workbook_bytes({"Sheet1": pd.DataFrame({"A": [1, 2, 3]})})
    analysis = splitter.analyze_workbook_bytes(workbook_bytes, "missing_timestamp.xlsx")
    assert not analysis.can_split
    assert analysis.errors == [
        "missing_timestamp.xlsx does not contain a worksheet with a TIMESTAMP column."
    ]


def test_unparseable_timestamp_is_reported():
    workbook_bytes = make_workbook_bytes(
        {"Sheet1": pd.DataFrame({"TIMESTAMP": ["not a timestamp"], "U1": [1]})}
    )
    analysis = splitter.analyze_workbook_bytes(workbook_bytes, "bad_timestamp.xlsx")
    assert not analysis.can_split
    assert "unparseable TIMESTAMP values" in analysis.errors[0]


def test_multiple_worksheets_are_split_with_same_boundaries():
    cleaned = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-07-10 13:00:00",
                "2026-07-10 13:39:00",
                "2026-07-10 15:50:00",
            ],
            "U1": [1, 2, 3],
        }
    )
    raw = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-07-10 13:00:00",
                "2026-07-10 13:39:00",
                "2026-07-10 15:50:00",
            ],
            "RECORD": [10, 11, 12],
        }
    )
    notes = pd.DataFrame({"Note": ["preserve me"]})
    analysis = splitter.analyze_workbook_bytes(
        make_workbook_bytes({"Cleaned_Data": cleaned, "Raw_Data": raw, "Notes": notes}),
        "multi_sheet.xlsx",
    )
    result, error = splitter.split_workbook(analysis, time(13, 38), time(15, 49))
    assert error is None
    for output_name, output_bytes in result.output_files:
        workbook = pd.read_excel(BytesIO(output_bytes), sheet_name=None, engine="openpyxl")
        assert set(workbook) == {"Cleaned_Data", "Raw_Data", "Notes"}
        assert len(workbook["Cleaned_Data"]) == 1
        assert len(workbook["Raw_Data"]) == 1
        assert workbook["Notes"].equals(notes)


def test_multiple_uploaded_files_can_use_different_times():
    dataframe1 = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-07-10 13:00:00",
                "2026-07-10 13:39:00",
                "2026-07-10 15:50:00",
            ]
        }
    )
    dataframe2 = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-07-10 14:00:00",
                "2026-07-10 14:03:00",
                "2026-07-10 16:18:00",
            ]
        }
    )
    result1, error1 = splitter.split_workbook(analyze_dataframe(dataframe1, "P1_Round1.xlsx"), time(13, 38), time(15, 49))
    result2, error2 = splitter.split_workbook(analyze_dataframe(dataframe2, "P1_Round2.xlsx"), time(14, 2), time(16, 17))
    assert error1 is None
    assert error2 is None
    assert result1.boundary1 == pd.Timestamp("2026-07-10 13:39:00")
    assert result2.boundary1 == pd.Timestamp("2026-07-10 14:03:00")


def test_output_naming_and_zip_generation():
    assert splitter.position_output_filename("P1_Round1.xlsx", 1) == "P1_Round1_position1.xlsx"
    assert splitter.position_output_filename("P1_Round1.xlsx", 2) == "P1_Round1_position2.xlsx"
    assert splitter.position_output_filename("P1_Round1.xlsx", 3) == "P1_Round1_position3.xlsx"
    archive_bytes = splitter.build_zip(
        [
            ("P1_Round1_position1.xlsx", b"one"),
            ("P1_Round1_position2.xlsx", b"two"),
            ("P1_Round1_position3.xlsx", b"three"),
        ]
    )
    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert archive.namelist() == [
            "P1_Round1_position1.xlsx",
            "P1_Round1_position2.xlsx",
            "P1_Round1_position3.xlsx",
        ]


def sample_dat_text():
    return '''"TOA5","CR350Series","metadata"
""TIMESTAMP"",""RECORD"",""A_U_ms"",""A_V_ms"",""A_W_ms"",""A_SonicTemp_C"",""A_SensorStatus"",""B_U_ms"",""B_V_ms"",""B_W_ms"",""B_SonicTemp_C"",""B_SensorStatus"",""C_U_ms"",""C_V_ms"",""C_W_ms"",""C_SonicTemp_C"",""C_SensorStatus""
"TS","RN","m/s","m/s","m/s","C","","m/s","m/s","m/s","C","","m/s","m/s","m/s","C",""
"","","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp","Smp"
"2026-07-10 16:39:08.4",1,0.111,-0.222,0.333,22.444,0,1.111,1.222,1.333,23.444,0,2.111,2.222,2.333,24.444,0
"2026-07-10 16:40:00.020",2,0.211,-0.322,0.433,22.544,0,1.211,1.322,1.433,23.544,0,2.211,2.322,2.433,24.544,0
"2026-07-10 16:41:00.020",3,0.311,-0.422,0.533,22.644,0,1.311,1.422,1.533,23.644,0,2.311,2.422,2.533,24.644,0
'''


def test_existing_dat_to_xlsx_still_works():
    dat_text = sample_dat_text()
    processed, error = dat.process_dat_bytes(dat_text.encode("utf-8"), "P1_Round3.dat")
    assert error is None
    assert list(processed.cleaned_dataframe.columns) == list(dat.FINAL_COLUMNS)
    assert processed.cleaned_dataframe.loc[0, "U1"] == 0.111


def test_integrated_dat_without_splitting_still_generates_full_xlsx():
    processed, error = dat.process_dat_bytes(sample_dat_text().encode("utf-8"), "P1_Round1.dat")
    assert error is None
    full_name = dat.dat_output_filename(processed.filename)
    full_bytes = dat.dat_to_xlsx_bytes(processed)
    workbook = pd.read_excel(BytesIO(full_bytes), sheet_name=None, engine="openpyxl")
    assert full_name == "P1_Round1.xlsx"
    assert set(workbook) == {"Cleaned_Data", "Raw_Data"}
    assert workbook["Cleaned_Data"].shape == (3, 13)
    assert workbook["Raw_Data"].shape == (3, 17)


def test_integrated_dat_with_splitting_generates_full_and_positions():
    processed, error = dat.process_dat_bytes(sample_dat_text().encode("utf-8"), "P1_Round1.dat")
    assert error is None
    full_file = (dat.dat_output_filename(processed.filename), dat.dat_to_xlsx_bytes(processed))
    analysis = splitter.analyze_workbook_dataframes(
        {"Cleaned_Data": processed.cleaned_dataframe, "Raw_Data": processed.raw_dataframe},
        full_file[0],
    )
    result, split_error = splitter.split_workbook(analysis, time(16, 39), time(16, 40))
    assert split_error is None
    generated_files = [full_file] + result.output_files
    assert [filename for filename, _bytes in generated_files] == [
        "P1_Round1.xlsx",
        "P1_Round1_position1.xlsx",
        "P1_Round1_position2.xlsx",
        "P1_Round1_position3.xlsx",
    ]
    for _output_name, output_bytes in result.output_files:
        workbook = pd.read_excel(BytesIO(output_bytes), sheet_name=None, engine="openpyxl")
        assert set(workbook) == {"Cleaned_Data", "Raw_Data"}
        assert len(workbook["Cleaned_Data"]) == 1
        assert len(workbook["Raw_Data"]) == 1
    assert all(qc.passed for qc in result.qc_results)
    assert [qc.position_rows for qc in result.qc_results] == [(1, 1, 1), (1, 1, 1)]


def test_integrated_dat_splitting_uses_user_defined_times_per_file():
    dat_text_one = sample_dat_text()
    dat_text_two = (
        sample_dat_text()
        .replace("16:39", "14:02")
        .replace("16:40", "14:03")
        .replace("16:41", "16:18")
    )
    processed_one, error_one = dat.process_dat_bytes(dat_text_one.encode("utf-8"), "P1_Round1.dat")
    processed_two, error_two = dat.process_dat_bytes(dat_text_two.encode("utf-8"), "P1_Round2.dat")
    assert error_one is None
    assert error_two is None

    analysis_one = splitter.analyze_workbook_dataframes(
        {"Cleaned_Data": processed_one.cleaned_dataframe, "Raw_Data": processed_one.raw_dataframe},
        "P1_Round1.xlsx",
    )
    analysis_two = splitter.analyze_workbook_dataframes(
        {"Cleaned_Data": processed_two.cleaned_dataframe, "Raw_Data": processed_two.raw_dataframe},
        "P1_Round2.xlsx",
    )
    result_one, error_one = splitter.split_workbook(analysis_one, time(16, 39), time(16, 40))
    result_two, error_two = splitter.split_workbook(analysis_two, time(14, 2), time(14, 3))
    assert error_one is None
    assert error_two is None
    assert result_one.boundary1 == pd.Timestamp("2026-07-10 16:40:00")
    assert result_two.boundary1 == pd.Timestamp("2026-07-10 14:03:00")


def transform_source_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TIMESTAMP": ["2026-07-10 16:39:08.4", "2026-07-10 16:39:09.4"],
            "U1": [2.0, None],
            "V1": [3.0, 30.0],
            "W1": [4.0, 40.0],
            "Temp1": [21.0, 22.0],
            "U2": [5.0, 50.0],
            "V2": [6.0, None],
            "W2": [7.0, 70.0],
            "Temp2": [23.0, 24.0],
            "U3": [8.0, 80.0],
            "V3": [9.0, 90.0],
            "W3": [10.0, None],
            "Temp3": [25.0, 26.0],
        }
    )


def test_position_velocity_transform_p01_and_p03_use_first_rule():
    for position_number in (1, 3):
        transformed = dat.add_transformed_velocity_columns(
            transform_source_dataframe(),
            position_number,
        )
        first_row = transformed.iloc[0]
        assert first_row["Vx1"] == 4.0
        assert first_row["Vy1"] == -2.0
        assert first_row["Vz1"] == -3.0
        assert first_row["Vx2"] == 7.0
        assert first_row["Vy2"] == -5.0
        assert first_row["Vz2"] == -6.0
        assert first_row["Vx3"] == 10.0
        assert first_row["Vy3"] == -8.0
        assert first_row["Vz3"] == -9.0


def test_position_velocity_transform_p04_and_p06_use_second_rule():
    for position_number in (4, 6):
        transformed = dat.add_transformed_velocity_columns(
            transform_source_dataframe(),
            position_number,
        )
        first_row = transformed.iloc[0]
        assert first_row["Vx1"] == -4.0
        assert first_row["Vy1"] == 2.0
        assert first_row["Vz1"] == -3.0
        assert first_row["Vx2"] == -7.0
        assert first_row["Vy2"] == 5.0
        assert first_row["Vz2"] == -6.0
        assert first_row["Vx3"] == -10.0
        assert first_row["Vy3"] == 8.0
        assert first_row["Vz3"] == -9.0


def test_position_velocity_transform_column_order_and_nan_handling():
    transformed = dat.add_transformed_velocity_columns(transform_source_dataframe(), 1)
    assert list(transformed.columns) == list(dat.POSITION_CLEANED_COLUMNS)
    assert pd.isna(transformed.loc[1, "Vy1"])
    assert pd.isna(transformed.loc[1, "Vz2"])
    assert pd.isna(transformed.loc[1, "Vx3"])


def test_hhmm_text_validation():
    app = load_app_module()
    valid_time, error = app.parse_hhmm_time("08:35")
    assert error is None
    assert valid_time == time(8, 35)
    valid_time, error = app.parse_hhmm_time("23:05")
    assert error is None
    assert valid_time == time(23, 5)

    for invalid_value in ["1:38 PM", "25:00", "13:75", "abc", "1:38", ""]:
        parsed_time, error = app.parse_hhmm_time(invalid_value)
        assert parsed_time is None
        assert error == "Please enter time in HH:MM format."


def test_hhmm_part_validation_accepts_and_normalizes_values():
    start_time, end_time, errors = splitter.build_time_range_from_parts(
        "P01",
        "8",
        "5",
        "13",
        "00",
    )
    assert errors == []
    assert start_time == time(8, 5)
    assert start_time.strftime("%H:%M") == "08:05"
    assert end_time == time(13, 0)
    assert end_time.strftime("%H:%M") == "13:00"

    start_time, end_time, errors = splitter.build_time_range_from_parts(
        "P02",
        "0",
        "0",
        "23",
        "59",
    )
    assert errors == []
    assert start_time == time(0, 0)
    assert end_time == time(23, 59)


def test_hhmm_part_validation_rejects_blank_invalid_and_out_of_range_values():
    start_time, end_time, errors = splitter.build_time_range_from_parts(
        "P02",
        "24",
        "60",
        "abc",
        "",
    )
    assert start_time is None
    assert end_time is None
    assert errors == [
        "P02 Start Hour must be between 00 and 23.",
        "P02 Start Minute must be between 00 and 59.",
        "P02 End Hour must be between 00 and 23.",
        "Please enter P02 End Minute.",
    ]

    start_time, end_time, errors = splitter.build_time_range_from_parts(
        "P03",
        "-1",
        "75",
        "30",
        "-1",
    )
    assert start_time is None
    assert end_time is None
    assert errors == [
        "P03 Start Hour must be between 00 and 23.",
        "P03 Start Minute must be between 00 and 59.",
        "P03 End Hour must be between 00 and 23.",
        "P03 End Minute must be between 00 and 59.",
    ]


def test_hhmm_parts_reconstruct_to_position_range_times():
    start_time, end_time, errors = splitter.build_time_range_from_parts(
        "P01",
        "12",
        "40",
        "12",
        "45",
    )
    assert errors == []
    assert start_time == time(12, 40)
    assert end_time == time(12, 45)


def test_start_time_options_use_dataset_minute_range():
    options = splitter.generate_minute_time_options(
        pd.Timestamp("2026-08-26 12:34:46"),
        pd.Timestamp("2026-08-26 13:03:59"),
    )
    assert options[0] == time(12, 34)
    assert options[-1] == time(13, 3)
    assert options[1] == time(12, 35)
    assert len(options) == 30


def test_duration_options_and_validation():
    assert splitter.duration_options() == list(range(46))
    assert splitter.format_duration_option(5) == "5 min"
    position_range, error = splitter.build_position_range_from_duration(
        "P01",
        time(12, 40),
        5,
    )
    assert error is None
    assert position_range.start_time == time(12, 40)
    assert position_range.duration_minutes == 5
    assert position_range.end_time == time(12, 45)
    assert position_range.end_is_exclusive

    position_range, error = splitter.build_position_range_from_duration(
        "P02",
        time(12, 40),
        0,
    )
    assert position_range is None
    assert error == "Please select a duration greater than 0 minutes for P02."

    assert splitter.build_position_range_from_duration("P03", time(12, 40), 1)[1] is None
    assert splitter.build_position_range_from_duration("P04", time(12, 40), 45)[1] is None


def test_duration_range_uses_half_open_interval():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-08-26 12:39:59.999",
                "2026-08-26 12:40:00",
                "2026-08-26 12:40:00.020",
                "2026-08-26 12:44:59.970",
                "2026-08-26 12:45:00",
                "2026-08-26 12:45:00.020",
            ],
            "U1": range(6),
        }
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [duration_range("P01", time(12, 40), 5)],
        ["C03_P01_26.08.26.xlsx"],
    )
    assert error is None
    workbook = pd.read_excel(BytesIO(result.output_files[0][1]), sheet_name=None, engine="openpyxl")
    split_frame = workbook["Cleaned_Data"]
    assert list(split_frame["U1"]) == [1, 2, 3]


def test_duration_range_extending_beyond_dataset_is_rejected():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-08-26 12:34:46",
                "2026-08-26 13:03:59",
            ]
        }
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [duration_range("P03", time(13, 0), 10)],
        ["C03_P03_26.08.26.xlsx"],
    )
    assert error is None
    assert result is not None
    assert result.position_summaries[0].rows == 1


def test_duration_range_empty_position_is_rejected():
    dataframe = pd.DataFrame(
        {"TIMESTAMP": ["2026-08-26 13:00:00", "2026-08-26 14:00:00"]}
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [duration_range("P01", time(13, 30), 5)],
        ["C03_P01_26.08.26.xlsx"],
    )
    assert result is None
    assert error == "P01: requested range contains no observations."


def test_duration_range_adjacent_positions_are_valid():
    dataframe = pd.DataFrame(
        {"TIMESTAMP": pd.date_range("2026-08-26 12:40:00", periods=11, freq="1min")}
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [
            duration_range("P01", time(12, 40), 5),
            duration_range("P02", time(12, 45), 5),
        ],
        ["C03_P01_26.08.26.xlsx", "C03_P02_26.08.26.xlsx"],
    )
    assert error is None
    assert [summary.rows for summary in result.position_summaries] == [5, 5]
    assert result.qc_results[0].no_duplicate_assignments


def test_multiday_workbook_analysis_and_date_time_options():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-08-28 14:36:01",
                "2026-08-28 14:37:00",
                "2026-09-10 12:39:00",
                "2026-09-10 12:40:09",
            ]
        }
    )
    analysis = analyze_dataframe(dataframe, "multi_day.xlsx")
    primary = analysis.primary_sheet
    assert primary is not None
    assert splitter.available_dates_from_timestamps(primary.parsed_timestamps) == [
        pd.Timestamp("2026-08-28").date(),
        pd.Timestamp("2026-09-10").date(),
    ]
    september_options = splitter.generate_minute_time_options_for_date(
        primary.parsed_timestamps,
        pd.Timestamp("2026-09-10").date(),
    )
    assert september_options[0] == time(12, 39)
    assert september_options[-1] == time(12, 40)


def test_date_start_time_combines_to_full_datetime_and_cross_midnight_end_format():
    position_range = dated_duration_range(
        "P01",
        pd.Timestamp("2026-09-10").date(),
        time(23, 59),
        3,
        1,
    )
    assert position_range.start_datetime == pd.Timestamp("2026-09-10 23:59:00").to_pydatetime()
    assert position_range.end_datetime == pd.Timestamp("2026-09-11 00:02:00").to_pydatetime()
    assert splitter.format_calculated_end(position_range.start_datetime, position_range.end_datetime) == "2026-09-11 00:02"


def test_multiday_split_uses_full_datetime_half_open_slicing():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-09-09 12:09:00",
                "2026-09-10 12:09:00",
                "2026-09-10 12:09:59.999",
                "2026-09-10 12:10:00",
            ],
            "U1": [0, 1, 2, 3],
        }
    )
    analysis = analyze_dataframe(dataframe, "multi_day.xlsx")
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [dated_duration_range("P01", pd.Timestamp("2026-09-10").date(), time(12, 9), 1, 1)],
        ["C03_P01_26.09.10_1209.xlsx"],
    )
    assert error is None
    workbook = pd.read_excel(BytesIO(result.output_files[0][1]), sheet_name=None, engine="openpyxl")
    assert list(workbook["Cleaned_Data"]["U1"]) == [1, 2]


def test_cross_midnight_split_uses_full_datetimes():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-09-10 23:59:00",
                "2026-09-11 00:00:00",
                "2026-09-11 00:01:59.999",
                "2026-09-11 00:02:00",
            ],
            "U1": [1, 2, 3, 4],
        }
    )
    analysis = analyze_dataframe(dataframe, "cross_midnight.xlsx")
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [dated_duration_range("P01", pd.Timestamp("2026-09-10").date(), time(23, 59), 3, 1)],
        ["C03_P01_26.09.10_2359.xlsx"],
    )
    assert error is None
    workbook = pd.read_excel(BytesIO(result.output_files[0][1]), sheet_name=None, engine="openpyxl")
    assert list(workbook["Cleaned_Data"]["U1"]) == [1, 2, 3]


def test_repeated_position_duplicate_rules_use_position_and_full_start_datetime():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2026-09-10 12:09:00", periods=5, freq="1min"),
        }
    )
    analysis = analyze_dataframe(dataframe, "repeat_positions.xlsx")
    valid_ranges = [
        dated_duration_range("P01", pd.Timestamp("2026-09-10").date(), time(12, 9), 1, 1),
        dated_duration_range("P01", pd.Timestamp("2026-09-10").date(), time(12, 10), 1, 1),
        dated_duration_range("P03", pd.Timestamp("2026-09-10").date(), time(12, 9), 1, 3),
    ]
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        valid_ranges,
        ["a.xlsx", "b.xlsx", "c.xlsx"],
    )
    assert error is None
    assert result is not None

    duplicate_ranges = [
        dated_duration_range("P01", pd.Timestamp("2026-09-10").date(), time(12, 9), 1, 1),
        dated_duration_range("P01", pd.Timestamp("2026-09-10").date(), time(12, 9), 1, 1),
    ]
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        duplicate_ranges,
        ["a.xlsx", "b.xlsx"],
    )
    assert result is None
    assert error == "P01: duplicate Position and Start Datetime."


def test_same_position_same_time_different_date_is_valid():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-09-09 12:09:00",
                "2026-09-10 12:09:00",
            ],
        }
    )
    analysis = analyze_dataframe(dataframe, "repeat_dates.xlsx")
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [
            dated_duration_range("P01", pd.Timestamp("2026-09-09").date(), time(12, 9), 1, 1),
            dated_duration_range("P01", pd.Timestamp("2026-09-10").date(), time(12, 9), 1, 1),
        ],
        ["a.xlsx", "b.xlsx"],
    )
    assert error is None
    assert result is not None


def test_repeated_position_output_filenames_include_start_time_when_needed():
    app = load_app_module()
    ranges = [
        dated_duration_range("P01", pd.Timestamp("2026-09-10").date(), time(11, 59), 3, 1),
        dated_duration_range("P01", pd.Timestamp("2026-09-10").date(), time(12, 9), 3, 1),
        dated_duration_range("P03", pd.Timestamp("2026-09-10").date(), time(12, 4), 3, 3),
    ]
    filenames = [
        app.split_position_workbook_filename("01", pd.Timestamp("2026-08-26").date(), ranges, position_range)
        for position_range in ranges
    ]
    assert filenames == [
        "C01_P01_26.09.10_1159.xlsx",
        "C01_P01_26.09.10_1209.xlsx",
        "C01_P03_26.09.10.xlsx",
    ]


def test_integrated_dat_batch_output_contains_xlsx_and_csv_files():
    app = load_app_module()
    processed, error = dat.process_dat_bytes(sample_dat_text().encode("utf-8"), "P1_Round1.dat")
    assert error is None
    batch_output, batch_error = app.build_completed_dat_case(
        processed,
        "03",
        pd.Timestamp("2026-08-26").date(),
        True,
        [
            duration_range("P01", time(16, 39), 1),
            duration_range("P02", time(16, 40), 1),
            duration_range("P03", time(16, 41), 1),
        ],
    )
    assert batch_error is None
    assert [filename for filename, _bytes in batch_output.files] == [
        "C03_All_26.08.26.xlsx",
        "C03_All_26.08.26.csv",
        "C03_P01_26.08.26.xlsx",
        "C03_P01_26.08.26.csv",
        "C03_P02_26.08.26.xlsx",
        "C03_P02_26.08.26.csv",
        "C03_P03_26.08.26.xlsx",
        "C03_P03_26.08.26.csv",
    ]
    for filename, output_bytes in batch_output.files:
        if filename.endswith(".xlsx"):
            workbook = pd.read_excel(BytesIO(output_bytes), sheet_name=None, engine="openpyxl")
            assert set(workbook) == {"Cleaned_Data", "Raw_Data"}
        else:
            csv_dataframe = pd.read_csv(BytesIO(output_bytes))
            assert "TIMESTAMP" in csv_dataframe.columns
            assert filename.endswith(".csv")


def test_integrated_dat_positions_have_transformed_cleaned_data_only():
    app = load_app_module()
    processed, error = dat.process_dat_bytes(sample_dat_text().encode("utf-8"), "P1_Round1.dat")
    assert error is None
    completed_case, process_error = app.build_completed_dat_case(
        processed,
        "03",
        pd.Timestamp("2026-08-26").date(),
        True,
        [
            duration_range("P01", time(16, 39), 1),
            duration_range("P02", time(16, 40), 1),
            duration_range("P03", time(16, 41), 1),
        ],
    )
    assert process_error is None

    all_workbook = pd.read_excel(BytesIO(completed_case.files[0][1]), sheet_name=None, engine="openpyxl")
    assert list(all_workbook["Cleaned_Data"].columns) == list(dat.FINAL_COLUMNS)
    assert not any(column in all_workbook["Cleaned_Data"].columns for column in dat.VELOCITY_COLUMNS)

    files_by_name = dict(completed_case.files)
    position_workbook = pd.read_excel(
        BytesIO(files_by_name["C03_P01_26.08.26.xlsx"]),
        sheet_name=None,
        engine="openpyxl",
    )
    cleaned = position_workbook["Cleaned_Data"]
    raw = position_workbook["Raw_Data"]
    assert list(cleaned.columns) == list(dat.POSITION_CLEANED_COLUMNS)
    assert not any(column in raw.columns for column in dat.VELOCITY_COLUMNS)
    assert cleaned.loc[0, "Vx1"] == cleaned.loc[0, "W1"]
    assert cleaned.loc[0, "Vy1"] == -cleaned.loc[0, "U1"]
    assert cleaned.loc[0, "Vz1"] == -cleaned.loc[0, "V1"]


def test_dat_full_and_position_csv_content_rules():
    app = load_app_module()
    processed, error = dat.process_dat_bytes(sample_dat_text().encode("utf-8"), "P1_Round1.dat")
    assert error is None
    completed_case, process_error = app.build_completed_dat_case(
        processed,
        "12",
        pd.Timestamp("2026-08-26").date(),
        True,
        [
            duration_range("P01", time(16, 39), 1),
            duration_range("P02", time(16, 40), 1),
            duration_range("P03", time(16, 41), 1),
        ],
    )
    assert process_error is None
    files_by_name = dict(completed_case.files)

    full_csv = pd.read_csv(BytesIO(files_by_name["C12_All_26.08.26.csv"]))
    assert list(full_csv.columns) == list(dat.FINAL_COLUMNS)
    assert not any(column in full_csv.columns for column in dat.VELOCITY_COLUMNS)

    position_csv = pd.read_csv(BytesIO(files_by_name["C12_P01_26.08.26.csv"]))
    position_xlsx = pd.read_excel(
        BytesIO(files_by_name["C12_P01_26.08.26.xlsx"]),
        sheet_name="Cleaned_Data",
        engine="openpyxl",
    )
    assert list(position_csv.columns) == list(dat.POSITION_CLEANED_COLUMNS)
    assert pd.to_datetime(position_csv["TIMESTAMP"]).equals(
        pd.to_datetime(position_xlsx["TIMESTAMP"])
    )
    pd.testing.assert_frame_equal(
        position_csv.drop(columns=["TIMESTAMP"]),
        position_xlsx.drop(columns=["TIMESTAMP"]),
        check_dtype=False,
    )
    assert position_csv.loc[0, "Vx1"] == position_csv.loc[0, "W1"]
    assert position_csv.loc[0, "Vy1"] == -position_csv.loc[0, "U1"]
    assert position_csv.loc[0, "Vz1"] == -position_csv.loc[0, "V1"]


def test_dat_p04_csv_uses_second_transform_rule():
    app = load_app_module()
    dat_text = sample_dat_text() + '"2026-07-10 16:42:00.020",4,0.411,-0.522,0.633,22.744,0,1.411,1.522,1.633,23.744,0,2.411,2.522,2.633,24.744,0\n'
    processed, error = dat.process_dat_bytes(dat_text.encode("utf-8"), "P1_Round1.dat")
    assert error is None
    completed_case, process_error = app.build_completed_dat_case(
        processed,
        "12",
        pd.Timestamp("2026-08-26").date(),
        True,
        [
            duration_range("P01", time(16, 39), 1),
            duration_range("P02", time(16, 40), 1),
            duration_range("P03", time(16, 41), 1),
            duration_range("P04", time(16, 42), 1),
        ],
    )
    assert process_error is None
    p04_csv = pd.read_csv(BytesIO(dict(completed_case.files)["C12_P04_26.08.26.csv"]))
    assert p04_csv.loc[0, "Vx1"] == -p04_csv.loc[0, "W1"]
    assert p04_csv.loc[0, "Vy1"] == p04_csv.loc[0, "U1"]
    assert p04_csv.loc[0, "Vz1"] == -p04_csv.loc[0, "V1"]


def test_current_case_zip_contains_xlsx_and_csv_without_raw_csv():
    app = load_app_module()
    processed, error = dat.process_dat_bytes(sample_dat_text().encode("utf-8"), "P1_Round1.dat")
    assert error is None
    completed_case, process_error = app.build_completed_dat_case(
        processed,
        "12",
        pd.Timestamp("2026-08-26").date(),
        True,
        [
            duration_range("P01", time(16, 39), 1),
            duration_range("P02", time(16, 40), 1),
            duration_range("P03", time(16, 41), 1),
        ],
    )
    assert process_error is None
    with ZipFile(BytesIO(dat.build_xlsx_zip(completed_case.files))) as archive:
        names = archive.namelist()
    assert names == [
        "C12_All_26.08.26.xlsx",
        "C12_All_26.08.26.csv",
        "C12_P01_26.08.26.xlsx",
        "C12_P01_26.08.26.csv",
        "C12_P02_26.08.26.xlsx",
        "C12_P02_26.08.26.csv",
        "C12_P03_26.08.26.xlsx",
        "C12_P03_26.08.26.csv",
    ]
    assert not any("_Raw.csv" in name for name in names)


def test_integrated_dat_batch_many_files_zip_generation():
    app = load_app_module()
    all_outputs = []
    for index in range(15):
        processed, error = dat.process_dat_bytes(
            sample_dat_text().encode("utf-8"),
            f"P{index + 1}_Round1.dat",
        )
        assert error is None
        case_number = f"{index + 1:02d}"
        batch_output, batch_error = app.build_completed_dat_case(
            processed,
            case_number,
            pd.Timestamp("2026-08-26").date(),
            True,
            [
                duration_range("P01", time(16, 39), 1),
                duration_range("P02", time(16, 40), 1),
                duration_range("P03", time(16, 41), 1),
            ],
        )
        assert batch_error is None
        all_outputs.extend(batch_output.files)

    assert len(all_outputs) == 120
    archive_bytes = dat.build_xlsx_zip(all_outputs)
    with ZipFile(BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        assert len(names) == 120
        assert "C01_All_26.08.26.xlsx" in names
        assert "C01_All_26.08.26.csv" in names
        assert "C15_P03_26.08.26.xlsx" in names
        assert "C15_P03_26.08.26.csv" in names


def test_case_and_filename_formatting():
    case_number, error = dat_batch.normalize_case_number("3")
    assert error is None
    assert case_number == "03"
    assert dat_batch.normalize_case_number("12")[0] == "12"
    assert dat_batch.normalize_case_number("")[1] == "Please enter a Case No."
    assert dat_batch.normalize_case_number("C03")[1] == "Case No must contain numbers only."
    case_date = pd.Timestamp("2026-08-26").date()
    assert dat_batch.format_case_date(case_date) == "26.08.26"
    assert dat_batch.all_workbook_filename("03", case_date) == "C03_All_26.08.26.xlsx"
    assert dat_batch.position_workbook_filename("03", 1, case_date) == "C03_P01_26.08.26.xlsx"
    assert "Date" not in dat_batch.all_workbook_filename("03", case_date)
    assert "Date" not in dat_batch.position_workbook_filename("03", 1, case_date)


def test_no_split_generates_only_all_workbook_with_complete_data():
    app = load_app_module()
    processed, error = dat.process_dat_bytes(sample_dat_text().encode("utf-8"), "RawData001.dat")
    assert error is None
    completed_case, process_error = app.build_completed_dat_case(
        processed,
        "03",
        pd.Timestamp("2026-08-26").date(),
        False,
        [],
    )
    assert process_error is None
    assert [filename for filename, _bytes in completed_case.files] == [
        "C03_All_26.08.26.xlsx",
        "C03_All_26.08.26.csv",
    ]
    workbook = pd.read_excel(BytesIO(completed_case.files[0][1]), sheet_name=None, engine="openpyxl")
    assert workbook["Cleaned_Data"].shape == (3, 13)
    assert workbook["Raw_Data"].shape == (3, 17)
    csv_dataframe = pd.read_csv(BytesIO(completed_case.files[1][1]))
    assert list(csv_dataframe.columns) == list(dat.FINAL_COLUMNS)
    assert not any(column in csv_dataframe.columns for column in dat.VELOCITY_COLUMNS)


def build_long_case_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TIMESTAMP": pd.date_range("2026-08-26 13:00:00", periods=20, freq="1min"),
            "U1": range(20),
        }
    )


def split_with_position_count(position_count: int):
    dataframe = build_long_case_dataframe()
    analysis = analyze_dataframe(dataframe, "C03_All_26.08.26.xlsx")
    ranges = [
        duration_range(f"P{index + 1:02d}", time(13, index * 2), 1)
        for index in range(position_count)
    ]
    filenames = [
        dat_batch.position_workbook_filename("03", index + 1, pd.Timestamp("2026-08-26").date())
        for index in range(position_count)
    ]
    return splitter.split_workbook_by_position_ranges(analysis, ranges, filenames)


def test_split_yes_with_2_3_5_and_6_positions():
    for position_count in (2, 3, 5, 6):
        result, error = split_with_position_count(position_count)
        assert error is None
        assert len(result.output_files) == position_count
        assert len(result.position_summaries) == position_count
        assert result.output_files[0][0] == "C03_P01_26.08.26.xlsx"


def test_dat_case_validation_allows_split_count_above_six():
    app = load_app_module()
    app.st.session_state = {"dat_completed_cases": []}
    ranges = [
        splitter.PositionRange(
            "P01",
            time(13, index),
            time(13, index),
            position_number=1,
        )
        for index in range(7)
    ]
    case_number, errors = app.validate_case_inputs(
        "3",
        pd.Timestamp("2026-08-26").date(),
        True,
        7,
        ranges,
        [],
    )
    assert case_number == "03"
    assert errors == []


def test_position_range_start_inclusive_and_end_minute_inclusive():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-08-26 13:00:00",
                "2026-08-26 13:00:59.999",
                "2026-08-26 13:01:00",
            ]
        }
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [splitter.PositionRange("P01", time(13, 0), time(13, 0))],
        ["C03_P01_26.08.26.xlsx"],
    )
    assert error is None
    workbook = pd.read_excel(BytesIO(result.output_files[0][1]), sheet_name=None, engine="openpyxl")
    assert len(workbook["Cleaned_Data"]) == 2


def test_position_gaps_are_allowed_without_lost_row_qc_failure():
    dataframe = pd.DataFrame(
        {
            "TIMESTAMP": [
                "2026-08-26 13:00:00",
                "2026-08-26 13:30:00",
                "2026-08-26 14:00:00",
            ]
        }
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [
            splitter.PositionRange("P01", time(13, 0), time(13, 0)),
            splitter.PositionRange("P02", time(14, 0), time(14, 0)),
        ],
        ["C03_P01_26.08.26.xlsx", "C03_P02_26.08.26.xlsx"],
    )
    assert error is None
    assert [summary.rows for summary in result.position_summaries] == [1, 1]
    assert result.qc_results[0].no_duplicate_assignments


def test_position_overlaps_are_allowed():
    dataframe = pd.DataFrame(
        {"TIMESTAMP": pd.date_range("2026-08-26 13:00:00", periods=121, freq="1min")}
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [
            splitter.PositionRange("P01", time(13, 0), time(14, 0)),
            splitter.PositionRange("P02", time(13, 50), time(15, 0)),
        ],
        ["C03_P01_26.08.26.xlsx", "C03_P02_26.08.26.xlsx"],
    )
    assert error is None
    assert result is not None
    assert [summary.rows for summary in result.position_summaries] == [61, 71]
    assert result.qc_results[0].no_duplicate_assignments


def test_position_start_later_than_end_is_rejected():
    dataframe = pd.DataFrame(
        {"TIMESTAMP": pd.date_range("2026-08-26 13:00:00", periods=121, freq="1min")}
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [splitter.PositionRange("P01", time(14, 0), time(13, 0))],
        ["C03_P01_26.08.26.xlsx"],
    )
    assert result is None
    assert error == "P01: Start Time cannot be later than End Time."


def test_position_with_zero_observations_is_rejected():
    dataframe = pd.DataFrame(
        {"TIMESTAMP": ["2026-08-26 13:00:00", "2026-08-26 15:00:00"]}
    )
    analysis = analyze_dataframe(dataframe)
    result, error = splitter.split_workbook_by_position_ranges(
        analysis,
        [splitter.PositionRange("P01", time(14, 0), time(14, 0))],
        ["C03_P01_26.08.26.xlsx"],
    )
    assert result is None
    assert error == "P01: requested range contains no observations."


def test_duplicate_case_date_detection_helper():
    app = load_app_module()
    case_date = pd.Timestamp("2026-08-26").date()
    completed_case = app.CompletedDatCase(
        original_filename="one.dat",
        case_number="03",
        case_date=case_date,
        split_enabled=False,
        position_count=0,
        position_ranges=[],
        all_rows=3,
        raw_rows=3,
        files=[("C03_All_26.08.26.xlsx", b"xlsx")],
        position_summaries=tuple(),
        qc_results=[],
    )
    app.st.session_state = {"dat_completed_cases": [completed_case]}
    assert app.case_already_used("03", case_date)
    assert not app.case_already_used("04", case_date)


def test_case_validation_accepts_entered_case_number():
    app = load_app_module()
    app.st.session_state = {"dat_completed_cases": []}
    case_number, errors = app.validate_case_inputs(
        "3",
        pd.Timestamp("2026-08-26").date(),
        False,
        0,
        [],
        [],
    )
    assert case_number == "03"
    assert errors == []


def test_case_validation_reports_empty_case_and_missing_date():
    app = load_app_module()
    app.st.session_state = {"dat_completed_cases": []}
    case_number, errors = app.validate_case_inputs("", None, False, 0, [], [])
    assert case_number is None
    assert "Please enter a Case No." in errors
    assert "Please select a Case Date." in errors


def test_case_validation_reports_incomplete_splits():
    app = load_app_module()
    app.st.session_state = {"dat_completed_cases": []}
    case_number, errors = app.validate_case_inputs(
        "3",
        pd.Timestamp("2026-08-26").date(),
        True,
        3,
        [splitter.PositionRange("P01", time(13, 0), time(13, 30))],
        ["Please select a duration greater than 0 minutes for P02."],
    )
    assert case_number == "03"
    assert "Please select Position, Date, Start Time, and Duration for all Splits." in errors
    assert "Please select a duration greater than 0 minutes for P02." in errors


def test_dat_workflow_source_has_no_empty_edt_card_placeholder():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert 'st.markdown(\'<div class="edt-card">\'' not in app_source
    assert 'value=""' in app_source
    assert "Process another file?" in app_source
    assert "edt-next-step" in app_source


def test_dat_workflow_source_removes_technical_details_and_preview():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert "Processing details" not in app_source
    assert "Preview cleaned data" not in app_source
    assert "Detected and renamed columns" not in app_source
    assert "Removed columns" not in app_source
    assert "Download All Data" not in app_source
    assert "Full Dataset" in app_source
    assert "Download Full XLSX" in app_source
    assert "Download Full CSV" in app_source


def test_dat_workflow_has_no_fixed_batch_limit():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert "MAX_DAT_UPLOADS" not in app_source
    assert "Maximum batch size" not in app_source
    assert "Files processed in current batch:" in app_source


def test_dat_workflow_source_uses_splits_and_limits_physical_positions_to_six():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert "MAX_DAT_PHYSICAL_POSITIONS = 6" in app_source
    assert "Number of Splits" in app_source
    assert "Number of Positions" not in app_source
    assert "options=list(range(1, MAX_DAT_PHYSICAL_POSITIONS + 1))" in app_source
    assert "max_value=MAX_DAT_PHYSICAL_POSITIONS" not in app_source
    assert "max_value=10" not in app_source


def test_dat_workflow_source_has_processing_status_messages():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert "st.status" in app_source
    assert "Processing C{case_number}..." in app_source
    assert "Reading DAT file..." in app_source
    assert "Creating cleaned dataset..." in app_source
    assert "Generating Full Dataset..." in app_source
    assert "Generating Position datasets..." in app_source
    assert "Applying velocity transformations..." in app_source
    assert "Creating XLSX and CSV outputs..." in app_source
    assert "Running QC..." in app_source
    assert "Preparing downloads..." in app_source
    assert "Processing completed successfully." in app_source
    assert "Processing failed: {process_error}" in app_source


def test_dat_workflow_source_uses_start_time_and_duration_inputs():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert "Start Time" in app_source
    assert "Date" in app_source
    assert "Split Time Ranges" in app_source
    assert "Duration (min)" in app_source
    assert "Calculated End" in app_source
    assert "generate_minute_time_options_for_date" in app_source
    assert "duration_options" in app_source
    assert "build_position_range_from_duration" in app_source
    assert "Start HH" not in app_source
    assert "Start MM" not in app_source
    assert "End HH" not in app_source
    assert "End MM" not in app_source
    assert "placeholder=\"HH\"" not in app_source
    assert "placeholder=\"MM\"" not in app_source


def test_current_result_survives_session_state_rerun():
    app = load_app_module()
    processed, error = dat.process_dat_bytes(sample_dat_text().encode("utf-8"), "RawData001.dat")
    assert error is None
    completed_case, process_error = app.build_completed_dat_case(
        processed,
        "12",
        pd.Timestamp("2026-08-26").date(),
        False,
        [],
    )
    assert process_error is None
    app.st.session_state = {
        "dat_completed_cases": [completed_case],
        "dat_current_result": completed_case,
    }
    assert app.st.session_state["dat_current_result"].files[0][0] == "C12_All_26.08.26.xlsx"
    assert app.st.session_state["dat_current_result"].files[0][1]


def test_process_next_file_preserves_completed_cases_before_reset():
    app = load_app_module()
    case_date = pd.Timestamp("2026-08-26").date()
    completed_case = app.CompletedDatCase(
        original_filename="one.dat",
        case_number="12",
        case_date=case_date,
        split_enabled=False,
        position_count=0,
        position_ranges=[],
        all_rows=3,
        raw_rows=3,
        files=[("C12_All_26.08.26.xlsx", b"xlsx")],
        position_summaries=tuple(),
        qc_results=[],
    )
    app.st.session_state = {
        "dat_completed_cases": [completed_case],
        "dat_current_result": completed_case,
        "dat_uploader_version": 0,
    }
    app.reset_current_dat_form()
    assert app.st.session_state["dat_current_result"] is None
    assert app.st.session_state["dat_completed_cases"] == [completed_case]
    assert app.st.session_state["dat_uploader_version"] == 1


def test_download_keys_are_unique_across_cases_and_positions():
    case_date = pd.Timestamp("2026-08-26").date()
    keys = set()
    for case_number in ("01", "02"):
        filenames = [
            dat_batch.all_workbook_filename(case_number, case_date),
            dat_batch.position_workbook_filename(case_number, 1, case_date),
            dat_batch.position_workbook_filename(case_number, 2, case_date),
        ]
        for filename in filenames:
            key = f"current_download_{case_number}_{dat_batch.format_case_date(case_date)}_{filename}"
            assert key not in keys
            keys.add(key)


def test_light_theme_config_exists():
    config_text = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text()
    assert 'base = "light"' in config_text
    assert 'backgroundColor = "#FFFFFF"' in config_text
    assert 'textColor = "#111827"' in config_text


def test_existing_merge_csv_still_works():
    app = load_app_module()

    class FakeUpload:
        def __init__(self, name, content):
            self.name = name
            self._content = content.encode("utf-8")

        def getvalue(self):
            return self._content

    uploads = [
        FakeUpload("one.csv", "sample,value\nA,1\nB,2\n"),
        FakeUpload("two.csv", "sample,value\nC,3\n"),
    ]
    datasets, errors = app.load_uploaded_csvs(uploads)
    assert errors == []
    merged, error = app.merge_datasets(datasets, True, True)
    assert error is None
    assert list(merged.columns) == ["sample", "value", "_source_file"]
    assert len(merged) == 3


def analysis_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Vx1": [1.0, 3.0],
            "Vy1": [2.0, 4.0],
            "Vz1": [2.0, 4.0],
            "Vx2": [0.0, 2.0],
            "Vy2": [0.0, 2.0],
            "Vz2": [1.0, 3.0],
            "Vx3": [-1.0, -3.0],
            "Vy3": [1.0, 3.0],
            "Vz3": [2.0, 2.0],
        }
    )


def test_analysis_csv_validation_and_mean_vectors():
    dataframe = analysis_dataframe()
    assert analysis.validate_analysis_csv(dataframe, "air.csv") is None
    missing_error = analysis.validate_analysis_csv(
        dataframe.drop(columns=["Vz3"]),
        "bad.csv",
    )
    assert missing_error == "bad.csv is missing required airflow columns: Vz3"
    assert analysis.validate_heights([0, 4.25, 8.75]) is None
    assert analysis.validate_heights([0, 9, 4]) == "Height 2 must be between 0 and 8.75 ft."

    means = analysis.compute_mean_vectors(dataframe)
    assert means[0].vx == 2.0
    assert means[0].vy == 3.0
    assert means[0].vz == 3.0
    assert round(means[0].magnitude, 6) == round((2.0**2 + 3.0**2 + 3.0**2) ** 0.5, 6)


def test_analysis_vector_records_coordinates_and_duplicates():
    means = analysis.compute_mean_vectors(analysis_dataframe())
    records = analysis.build_vector_records(4, [1.0, 4.0, 7.0], means, "p4.csv")
    assert [(record.x, record.y) for record in records] == [(5.33, 2.5)] * 3
    assert [record.z for record in records] == [1.0, 4.0, 7.0]
    assert records[0].position_number == 4
    assert records[0].height_index == 1
    duplicate_errors = analysis.duplicate_position_height_errors(
        records,
        analysis.build_vector_records(4, [1.0, 5.0, 8.0], means, "p4_again.csv"),
    )
    assert duplicate_errors == [
        "P4-H1 duplicates an existing position + height combination at 1 ft."
    ]
    within_upload_errors = analysis.duplicate_position_height_errors(
        [],
        analysis.build_vector_records(2, [1.0, 1.0, 8.0], means, "p2.csv"),
    )
    assert within_upload_errors == [
        "P2-H2 duplicates another height in this upload at 1 ft."
    ]


def test_analysis_display_vectors_are_normalized_without_changing_true_values():
    records = [
        analysis.VectorRecord(1, 1, 2.67, 2.5, 1.0, 0.3, 0.4, 0.0, 0.5, "p1.csv"),
        analysis.VectorRecord(1, 2, 2.67, 2.5, 4.0, 0.0, 0.0, 0.0, 0.0, "p1.csv"),
    ]
    display_us, display_vs, display_ws = analysis.display_vector_components(records, "imperial")
    max_length = max(analysis.room_dimensions("imperial")) * analysis.MAX_ARROW_LENGTH_FRACTION
    assert round(display_us[0], 6) == round(0.6 * max_length, 6)
    assert round(display_vs[0], 6) == round(0.8 * max_length, 6)
    assert display_ws[0] == 0.0
    assert (display_us[1], display_vs[1], display_ws[1]) == (0.0, 0.0, 0.0)

    summary = analysis.vector_records_to_dataframe(records, "si")
    assert summary.loc[0, "Mean Vx (m/s)"] == 0.3
    assert summary.loc[0, "Mean Vy (m/s)"] == 0.4
    assert summary.loc[0, "Mean Vz (m/s)"] == 0.0
    assert summary.loc[0, "Velocity magnitude (m/s)"] == 0.5


def test_analysis_si_and_imperial_unit_conversions_are_correct():
    record = analysis.VectorRecord(6, 3, 5.33, 7.5, 8.75, 0.3, -0.4, 0.5, (0.3**2 + 0.4**2 + 0.5**2) ** 0.5, "p6.csv")
    assert analysis.room_dimensions("imperial") == (8.0, 10.0, 8.75)
    assert tuple(round(value, 6) for value in analysis.room_dimensions("si")) == (2.4384, 3.048, 2.667)
    assert analysis.position_to_xy_in_units(1, "imperial") == (2.67, 2.5)
    assert analysis.position_to_xy_in_units(6, "imperial") == (5.33, 7.5)
    assert analysis.position_to_xy_in_units(1, "si") == (2.67 * 0.3048, 2.5 * 0.3048)
    assert analysis.position_to_xy_in_units(6, "si") == (5.33 * 0.3048, 7.5 * 0.3048)

    si_values = analysis.converted_record_values(record, "si")
    imperial_values = analysis.converted_record_values(record, "imperial")
    assert si_values["z"] == 8.75 * 0.3048
    assert imperial_values["z"] == 8.75
    assert imperial_values["vx"] == 0.3 * analysis.M_PER_S_TO_FT_PER_S
    assert imperial_values["vy"] == -0.4 * analysis.M_PER_S_TO_FT_PER_S
    assert imperial_values["vz"] == 0.5 * analysis.M_PER_S_TO_FT_PER_S
    assert round(imperial_values["magnitude"], 12) == round(record.magnitude * analysis.M_PER_S_TO_FT_PER_S, 12)
    assert analysis.direction_matches_after_unit_conversion(record)


def test_analysis_si_and_imperial_display_scaling_is_equivalent():
    records = [
        analysis.VectorRecord(1, 1, 2.67, 2.5, 1.0, 0.1, 0.0, 0.0, 0.1, "p1.csv"),
        analysis.VectorRecord(2, 1, 2.67, 5.0, 2.0, 0.0, 0.2, 0.0, 0.2, "p2.csv"),
    ]
    imperial_us, imperial_vs, imperial_ws = analysis.display_vector_components(records, "imperial")
    si_us, si_vs, si_ws = analysis.display_vector_components(records, "si")
    for imperial_component, si_component in zip(
        imperial_us + imperial_vs + imperial_ws,
        si_us + si_vs + si_ws,
    ):
        assert round(si_component, 12) == round(imperial_component * analysis.FT_TO_M, 12)


def test_analysis_position_coordinates_are_exact_and_not_swapped():
    expected = {
        1: (2.67, 2.5),
        2: (2.67, 5.0),
        3: (2.67, 7.5),
        4: (5.33, 2.5),
        5: (5.33, 5.0),
        6: (5.33, 7.5),
    }
    analysis.validate_position_coordinates()
    assert analysis.POSITION_COORDINATES == expected
    assert [analysis.position_to_xy(position)[0] for position in (1, 2, 3)] == [2.67, 2.67, 2.67]
    assert [analysis.position_to_xy(position)[0] for position in (4, 5, 6)] == [5.33, 5.33, 5.33]
    assert [analysis.position_to_xy(position)[1] for position in (1, 2, 3)] == [2.5, 5.0, 7.5]
    assert [analysis.position_to_xy(position)[1] for position in (4, 5, 6)] == [2.5, 5.0, 7.5]


def test_analysis_plot_source_uses_velocity_colormap_colorbar_and_no_position_labels():
    source = (PROJECT_ROOT / "analysis_processor.py").read_text()
    assert "POSITION_COLORS" not in source
    assert "VELOCITY_COLORMAP = \"turbo\"" in source
    assert "fig.colorbar" in source
    assert "colorbar.set_label(f\"Velocity magnitude ({unit_spec.velocity_unit})\")" in source
    assert "display_vector_components(vector_records, unit_system)" in source
    assert "MAX_ARROW_LENGTH_FRACTION" in source
    assert "ax.set_proj_type(\"ortho\")" in source
    assert "VIEW_ELEVATION" in source
    assert "VIEW_AZIMUTH" in source
    assert "VIEW_ROLL" in source
    assert "apply_matplotlib_camera(ax, elevation, azimuth, roll)" in source
    assert "plot_interactive_air_vectors" in source
    assert "camera_eye_from_angles" in source
    assert "show_vector_labels: bool = False" in source
    assert "| |V| =" not in source
    assert "ax.text(" not in source
    assert "mode=\"markers+text\"" not in source
    assert "Vector origins" in source
    assert "floor_x" not in source
    assert "floor_y" not in source
    assert "floor_z" not in source
    assert "Colors identify measurement positions" not in source
    assert "Colors indicate velocity magnitude" in source


def test_analysis_camera_defaults_and_helpers():
    assert analysis.VIEW_AZIMUTH == 10
    assert analysis.VIEW_ELEVATION == 15
    assert analysis.VIEW_ROLL == 0
    eye = analysis.camera_eye_from_angles(10, 15)
    assert set(eye) == {"x", "y", "z"}
    assert round((eye["x"] ** 2 + eye["y"] ** 2 + eye["z"] ** 2) ** 0.5, 6) == 1.85
    assert analysis.camera_up_from_roll(0) == {"x": 0.0, "y": 0.0, "z": 1.0}


def test_analysis_plot_exports_eps_and_png():
    if importlib.util.find_spec("matplotlib") is None:
        requirements = (PROJECT_ROOT / "requirements.txt").read_text()
        assert "matplotlib" in requirements
        return

    means = analysis.compute_mean_vectors(analysis_dataframe())
    records = analysis.build_vector_records(1, [1.0, 4.0, 7.0], means, "p1.csv")
    si_fig = analysis.plot_3d_air_vectors(records, "si", azimuth=15, elevation=35, roll=5)
    imperial_fig = analysis.plot_3d_air_vectors(records, "imperial", azimuth=15, elevation=35, roll=5)
    si_eps_bytes = analysis.export_figure_eps(si_fig)
    si_png_bytes = analysis.export_figure_png(si_fig)
    imperial_eps_bytes = analysis.export_figure_eps(imperial_fig)
    imperial_png_bytes = analysis.export_figure_png(imperial_fig)
    assert si_eps_bytes.startswith(b"%!PS-Adobe")
    assert si_png_bytes.startswith(b"\x89PNG")
    assert imperial_eps_bytes.startswith(b"%!PS-Adobe")
    assert imperial_png_bytes.startswith(b"\x89PNG")


def test_analysis_interactive_plot_dependency_is_declared_or_available():
    if importlib.util.find_spec("plotly") is None:
        requirements = (PROJECT_ROOT / "requirements.txt").read_text()
        assert "plotly" in requirements
        return

    means = analysis.compute_mean_vectors(analysis_dataframe())
    records = analysis.build_vector_records(1, [1.0, 4.0, 7.0], means, "p1.csv")
    fig = analysis.plot_interactive_air_vectors(records, "si", 10, 15, 0)
    assert fig.layout.scene.camera.projection.type == "orthographic"
    assert fig.layout.scene.xaxis.title.text == "X (m)"
    assert "Velocity magnitude (m/s)" in str(fig.to_dict())
    assert "markers+text" not in str(fig.to_dict())


def test_interactive_origin_markers_match_vector_start_points():
    if importlib.util.find_spec("plotly") is None:
        requirements = (PROJECT_ROOT / "requirements.txt").read_text()
        assert "plotly" in requirements
        return

    means = analysis.compute_mean_vectors(analysis_dataframe())
    records = analysis.build_vector_records(4, [1.0, 4.0, 7.0], means, "p4.csv")
    fig = analysis.plot_interactive_air_vectors(records, "imperial", 10, 15, 0)
    origin_trace = [
        trace for trace in fig.data
        if getattr(trace, "name", None) == "Vector origins"
    ][0]
    assert list(origin_trace.x) == [record.x for record in records]
    assert list(origin_trace.y) == [record.y for record in records]
    assert list(origin_trace.z) == [record.z for record in records]


def vertical_profile_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Vx1": [3.0, 0.0],
            "Vy1": [4.0, 0.0],
            "Vz1": [0.0, 0.0],
            "Vx2": [0.0, 0.0],
            "Vy2": [0.0, 6.0],
            "Vz2": [0.0, 8.0],
            "Vx3": [1.0, 2.0],
            "Vy3": [2.0, 2.0],
            "Vz3": [2.0, 1.0],
            "Temp1": [20.0, 22.0],
            "Temp2": [24.0, 26.0],
            "Temp3": [28.0, 30.0],
            "T_low": [21.0, 23.0],
            "T_mid": [25.0, 27.0],
            "T_high": [29.0, 31.0],
            "CO2_A": [400.0, 500.0],
            "CO2_B": [600.0, 700.0],
            "CO2_C": [800.0, 900.0],
        }
    )


def test_vertical_velocity_profile_averages_instantaneous_speed():
    profile, errors = vertical.compute_air_speed_profile(vertical_profile_dataframe())
    assert errors == []
    assert profile[0][0] == 2.5
    assert profile[1][0] == 5.0
    expected_h3 = (((1.0**2 + 2.0**2 + 2.0**2) ** 0.5) + ((2.0**2 + 2.0**2 + 1.0**2) ** 0.5)) / 2
    assert profile[2][0] == expected_h3


def test_vertical_profile_raw_mode_is_the_only_profile_mode():
    assert vertical.PROFILE_MODES == ("Raw Profiles",)


def test_vertical_temperature_auto_and_manual_mapping():
    dataframe = vertical_profile_dataframe()
    assert vertical.default_temperature_columns(dataframe) == ["Temp1", "Temp2", "Temp3"]
    auto_profile, auto_errors = vertical.compute_temperature_profile(dataframe, ["Temp1", "Temp2", "Temp3"])
    manual_profile, manual_errors = vertical.compute_temperature_profile(dataframe, ["T_low", "T_mid", "T_high"])
    assert auto_errors == []
    assert manual_errors == []
    assert [stat[0] for stat in auto_profile] == [21.0, 25.0, 29.0]
    assert [stat[0] for stat in manual_profile] == [22.0, 26.0, 30.0]


def test_vertical_contaminant_mapping_unit_and_summary():
    records, errors = vertical.build_vertical_profile_records(
        vertical_profile_dataframe(),
        "Experimental",
        2,
        [2.0, 4.0, 6.0],
        ["Contaminant Concentration"],
        "Raw Profiles",
        8.75,
        "p2.csv",
        contaminant_columns=["CO2_A", "CO2_B", "CO2_C"],
        contaminant_name="CO2",
        concentration_unit="ppm",
        cin=400.0,
        cout=800.0,
    )
    assert errors == []
    assert len(records) == 3
    assert records[0].unit == "ppm"
    assert records[0].contaminant_name == "CO2"
    summary = vertical.vertical_profile_records_to_dataframe(records)
    assert "Normalized Value" not in summary.columns
    assert "Normalized Height" not in summary.columns
    assert list(summary["Source Column"]) == ["CO2_A", "CO2_B", "CO2_C"]


def test_vertical_build_records_for_three_variables_and_height_order():
    records, errors = vertical.build_vertical_profile_records(
        vertical_profile_dataframe(),
        "Experimental",
        1,
        [6.0, 2.0, 4.0],
        ["Air Velocity", "Temperature", "Contaminant Concentration"],
        "Raw Profiles",
        8.75,
        "p1.csv",
        supply_velocity=2.5,
        temperature_columns=["Temp1", "Temp2", "Temp3"],
        tin=20.0,
        tout=25.0,
        contaminant_columns=["CO2_A", "CO2_B", "CO2_C"],
        contaminant_name="Tracer gas",
        concentration_unit="ppm",
        cin=400.0,
        cout=800.0,
    )
    assert errors == []
    assert len(records) == 9
    velocity_records = [record for record in records if record.variable == "Air Velocity"]
    assert [record.height_ft for record in velocity_records] == [2.0, 4.0, 6.0]
    assert velocity_records[0].mean == 5.0
    assert velocity_records[0].height_ft == 2.0


def test_vertical_profile_variables_can_use_independent_source_files():
    velocity_frame = vertical_profile_dataframe()[["Vx1", "Vy1", "Vz1", "Vx2", "Vy2", "Vz2", "Vx3", "Vy3", "Vz3"]]
    temperature_frame = vertical_profile_dataframe()[["T_low", "T_mid", "T_high"]]
    contaminant_frame = vertical_profile_dataframe()[["CO2_A", "CO2_B", "CO2_C"]]
    records, errors = vertical.build_vertical_profile_records_from_sources(
        {
            "Air Velocity": velocity_frame,
            "Temperature": temperature_frame,
            "Contaminant Concentration": contaminant_frame,
        },
        {
            "Air Velocity": "velocity.csv",
            "Temperature": "temperature.csv",
            "Contaminant Concentration": "contaminant.csv",
        },
        "Experimental",
        3,
        [2.0, 4.0, 6.0],
        ["Air Velocity", "Temperature", "Contaminant Concentration"],
        "Raw Profiles",
        8.75,
        supply_velocity=2.5,
        temperature_columns=["T_low", "T_mid", "T_high"],
        tin=20.0,
        tout=30.0,
        contaminant_columns=["CO2_A", "CO2_B", "CO2_C"],
        contaminant_name="CO2",
        concentration_unit="ppm",
        cin=400.0,
        cout=900.0,
    )
    assert errors == []
    assert len(records) == 9
    assert {record.source_file for record in records if record.variable == "Air Velocity"} == {"velocity.csv"}
    assert {record.source_file for record in records if record.variable == "Temperature"} == {"temperature.csv"}
    assert {record.source_file for record in records if record.variable == "Contaminant Concentration"} == {"contaminant.csv"}

    unified = vertical.vertical_profile_records_to_unified_dataframe(
        records,
        ["Air Velocity", "Temperature", "Contaminant Concentration"],
    )
    assert len(unified) == 3
    assert list(unified["Position"]) == ["P3", "P3", "P3"]
    assert "Air Velocity Mean" in unified.columns
    assert "Temperature Mean" in unified.columns
    assert "Contaminant Concentration Mean" in unified.columns
    assert list(unified["Air Velocity Source File"]) == ["velocity.csv"] * 3
    assert list(unified["Temperature Source File"]) == ["temperature.csv"] * 3
    assert list(unified["Contaminant Concentration Source File"]) == ["contaminant.csv"] * 3


def test_vertical_profile_missing_selected_variable_remains_blank_in_unified_summary():
    records, errors = vertical.build_vertical_profile_records_from_sources(
        {"Air Velocity": vertical_profile_dataframe()[["Vx1", "Vy1", "Vz1", "Vx2", "Vy2", "Vz2", "Vx3", "Vy3", "Vz3"]]},
        {"Air Velocity": "velocity.csv"},
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity", "Temperature"],
        "Raw Profiles",
        8.75,
        supply_velocity=2.5,
        tin=20.0,
        tout=30.0,
    )
    assert errors == []
    assert {record.variable for record in records} == {"Air Velocity"}
    unified = vertical.vertical_profile_records_to_unified_dataframe(records, ["Air Velocity", "Temperature"])
    assert len(unified) == 3
    assert unified["Temperature Mean"].isna().all()
    assert unified["Temperature Source File"].isna().all()


def replicate_frames():
    frames = []
    for offset, filename in [(0.0, "rep1.csv"), (1.0, "rep2.csv"), (2.0, "rep3.csv")]:
        frame = vertical_profile_dataframe().copy()
        for column in ["Vx1", "Vy1", "Vz1", "Vx2", "Vy2", "Vz2", "Vx3", "Vy3", "Vz3", "Temp1", "Temp2", "Temp3", "CO2_A", "CO2_B", "CO2_C"]:
            frame[column] = frame[column] + offset
        frame["TIMESTAMP"] = pd.date_range("2026-09-10 12:00:00", periods=len(frame), freq="1s")
        frame["Note"] = ["first", "second"]
        frames.append(replicate.ReplicateDataset(filename, frame))
    return frames


def test_replicate_rowwise_mean_sd_measurement_sequence_alignment_and_exports():
    datasets = replicate_frames()
    datasets[1].dataframe["TIMESTAMP"] = pd.date_range("2026-09-10 14:15:00", periods=len(datasets[1].dataframe), freq="1s")
    datasets[2].dataframe["TIMESTAMP"] = pd.date_range("2026-09-10 15:21:00", periods=len(datasets[2].dataframe), freq="1s")
    processed, report, errors = replicate.rowwise_replicate_statistics(
        datasets,
        include_optional_numeric=False,
    )
    assert errors == []
    assert report.method == "measurement_sequence"
    assert report.matched_rows == 2
    assert processed["TIMESTAMP"].iloc[0] == datasets[0].dataframe["TIMESTAMP"].iloc[0]
    assert processed["Vx1"].iloc[0] == 4.0
    assert processed["Vx1_std"].iloc[0] == 1.0
    assert "Vx1_mean" not in processed.columns
    assert "TIMESTAMP_std" not in processed.columns
    assert "Note_std" not in processed.columns
    assert processed["Note"].iloc[0] == "first"
    vx1_index = list(processed.columns).index("Vx1")
    assert list(processed.columns)[vx1_index + 1] == "Vx1_std"
    assert replicate.dataframe_to_csv_bytes(processed).startswith(b"\xef\xbb\xbf")
    assert replicate.dataframe_to_xlsx_bytes(processed).startswith(b"PK")
    zip_bytes = replicate.build_replicate_zip([("processed.csv", b"csv"), ("summary.xlsx", b"xlsx")])
    with ZipFile(BytesIO(zip_bytes)) as archive:
        assert set(archive.namelist()) == {"processed.csv", "summary.xlsx"}


def test_replicate_measurement_sequence_uses_common_minimum_row_count():
    datasets = replicate_frames()
    datasets[1] = replicate.ReplicateDataset(
        datasets[1].filename,
        pd.concat([datasets[1].dataframe, datasets[1].dataframe.iloc[[0]]], ignore_index=True),
    )
    datasets[2] = replicate.ReplicateDataset(
        datasets[2].filename,
        pd.concat([datasets[2].dataframe, datasets[2].dataframe.iloc[[0]], datasets[2].dataframe.iloc[[1]]], ignore_index=True),
    )
    processed, report, errors = replicate.rowwise_replicate_statistics(
        datasets,
        include_optional_numeric=False,
    )
    assert errors == []
    assert report.method == "measurement_sequence"
    assert report.matched_rows == 2
    assert len(processed) == 2
    assert report.unmatched_rows == {"rep1.csv": 0, "rep2.csv": 1, "rep3.csv": 2}


def test_replicate_matching_columns_missing_column_rejection():
    datasets = replicate_frames()
    datasets[1] = replicate.ReplicateDataset(
        datasets[1].filename,
        datasets[1].dataframe.drop(columns=["Vx1"]),
    )
    columns, errors = replicate.matching_numeric_columns(datasets, include_optional_numeric=False)
    assert columns == []
    assert "Replicate 2 is missing column: Vx1." in errors


def test_replicate_vertical_profile_mean_speed_and_between_replicate_sd():
    datasets = replicate_frames()
    records, errors = vertical.build_replicate_vertical_profile_records(
        {"Air Velocity": [dataset.dataframe for dataset in datasets]},
        {"Air Velocity": [dataset.filename for dataset in datasets]},
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity"],
        "Raw Profiles",
        8.75,
        supply_velocity=2.0,
    )
    assert errors == []
    h1_record = [record for record in records if record.height_id == "H1"][0]
    expected_replicate_means = []
    for dataset in datasets:
        speed = (dataset.dataframe["Vx1"] ** 2 + dataset.dataframe["Vy1"] ** 2 + dataset.dataframe["Vz1"] ** 2).pow(0.5)
        expected_replicate_means.append(float(speed.mean()))
    assert list(h1_record.replicate_means) == expected_replicate_means
    assert h1_record.replicate_standard_deviation == pd.Series(expected_replicate_means).std(ddof=1)
    assert h1_record.normalized_standard_deviation is None
    assert h1_record.normalized_value is None


def test_vertical_profile_uses_measurement_std_error_bars_for_multiple_positions():
    p1_records, errors = vertical.build_vertical_profile_records(
        vertical_profile_dataframe(),
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity"],
        "Raw Profiles",
        8.75,
        "p1.csv",
    )
    assert errors == []
    p2_records, errors = vertical.build_vertical_profile_records(
        vertical_profile_dataframe(),
        "Experimental",
        2,
        [2.0, 4.0, 6.0],
        ["Air Velocity"],
        "Raw Profiles",
        8.75,
        "p2.csv",
    )
    assert errors == []
    records = p1_records + p2_records
    assert vertical.records_have_plot_standard_deviation(records, "Raw Profiles")
    assert vertical.profile_plot_standard_deviation(records[0], "Raw Profiles") == records[0].standard_deviation
    if importlib.util.find_spec("matplotlib") is None:
        return
    fig = vertical.plot_vertical_profiles(records, ["Air Velocity"], "Raw Profiles", show_error_bars=True)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) >= 2
    assert [text.get_text() for text in fig.legends[0].texts] == ["P1", "P2"]
    assert vertical.export_vertical_profile_png(fig).startswith(b"\x89PNG")


def test_vertical_profile_si_ip_axis_limits_and_unit_conversions():
    records, errors = vertical.build_vertical_profile_records(
        vertical_profile_dataframe(),
        "Experimental",
        1,
        [0.67, 2.0, 3.33],
        ["Air Velocity", "Temperature"],
        "Raw Profiles",
        8.75,
        "p1.csv",
        temperature_columns=["Temp1", "Temp2", "Temp3"],
    )
    assert errors == []
    velocity_record = [record for record in records if record.variable == "Air Velocity" and record.height_id == "H1"][0]
    temperature_record = [record for record in records if record.variable == "Temperature" and record.height_id == "H1"][0]
    assert round(0.35 * vertical.MPS_TO_FPM, 1) == 68.9
    assert vertical.convert_profile_value(velocity_record, "ip") == velocity_record.mean * vertical.MPS_TO_FPM
    assert vertical.convert_profile_standard_deviation(velocity_record, "Raw Profiles", "ip") == vertical.profile_plot_standard_deviation(velocity_record, "Raw Profiles") * vertical.MPS_TO_FPM
    assert vertical.convert_profile_value(temperature_record, "ip") == temperature_record.mean * 9.0 / 5.0 + 32.0
    assert vertical.convert_profile_standard_deviation(temperature_record, "Raw Profiles", "ip") == vertical.profile_plot_standard_deviation(temperature_record, "Raw Profiles") * 9.0 / 5.0
    assert vertical.convert_profile_height(velocity_record, "si") == velocity_record.height_ft * 0.3048

    if importlib.util.find_spec("matplotlib") is None:
        return
    si_fig = vertical.plot_vertical_profiles(records, ["Air Velocity", "Temperature"], "Raw Profiles", unit_system="si")
    ip_fig = vertical.plot_vertical_profiles(records, ["Air Velocity", "Temperature"], "Raw Profiles", unit_system="ip")
    si_velocity_axis, si_temperature_axis = si_fig.axes
    ip_velocity_axis, ip_temperature_axis = ip_fig.axes
    assert si_fig._suptitle.get_text() == "Vertical Profiles — SI Units"
    assert ip_fig._suptitle.get_text() == "Vertical Profiles — IP Units"
    assert si_velocity_axis.get_xlim() == (0.0, 0.35)
    assert ip_velocity_axis.get_xlim() == (0.0, 70.0)
    assert si_temperature_axis.get_xlim() == (23.0, 25.0)
    assert ip_temperature_axis.get_xlim() == (73.0, 77.0)
    assert si_velocity_axis.get_ylabel() == "Height (m)"
    assert ip_velocity_axis.get_ylabel() == "Height (ft)"
    assert si_velocity_axis.get_ylim() == (0.0, vertical.ROOM_HEIGHT_M)
    assert ip_velocity_axis.get_ylim() == (0.0, 8.75)
    assert si_velocity_axis.get_xlabel() == "Air velocity (m/s)"
    assert ip_velocity_axis.get_xlabel() == "Air velocity (fpm)"
    assert si_temperature_axis.get_xlabel() == "Temperature (°C)"
    assert ip_temperature_axis.get_xlabel() == "Temperature (°F)"


def test_vertical_profile_uses_raw_measurement_std_for_error_bars():
    records, errors = vertical.build_vertical_profile_records(
        vertical_profile_dataframe(),
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity", "Temperature"],
        "Raw Profiles",
        8.75,
        "p1.csv",
        supply_velocity=2.0,
        temperature_columns=["Temp1", "Temp2", "Temp3"],
        tin=20.0,
        tout=30.0,
    )
    assert errors == []
    velocity_record = [record for record in records if record.variable == "Air Velocity"][0]
    temperature_record = [record for record in records if record.variable == "Temperature"][0]
    assert velocity_record.normalized_standard_deviation is None
    assert temperature_record.normalized_standard_deviation is None
    assert vertical.profile_plot_standard_deviation(velocity_record, "Raw Profiles") == velocity_record.standard_deviation


def test_vertical_profile_reads_replicate_generated_std_columns():
    dataframe = vertical_profile_dataframe().copy()
    dataframe["Vx1_std"] = 0.10
    dataframe["Vy1_std"] = 0.20
    dataframe["Vz1_std"] = 0.30
    dataframe["Vx2_std"] = 0.11
    dataframe["Vy2_std"] = 0.21
    dataframe["Vz2_std"] = 0.31
    dataframe["Vx3_std"] = 0.12
    dataframe["Vy3_std"] = 0.22
    dataframe["Vz3_std"] = 0.32
    records, errors = vertical.build_vertical_profile_records_from_sources(
        {"Air Velocity": dataframe},
        {"Air Velocity": "replicate_mean.csv"},
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity"],
        "Raw Profiles",
        8.75,
        supply_velocity=2.0,
    )
    assert errors == []
    h1 = [record for record in records if record.height_id == "H1"][0]
    expected_std = (0.10**2 + 0.20**2 + 0.30**2) ** 0.5
    assert abs(h1.standard_deviation - expected_std) < 1e-12
    assert h1.normalized_standard_deviation is None
    assert abs(h1.standard_deviation - expected_std) < 1e-12


def test_replicate_temperature_and_contaminant_between_replicate_sd_and_normalized_sd():
    datasets = replicate_frames()
    records, errors = vertical.build_replicate_vertical_profile_records(
        {
            "Temperature": [dataset.dataframe for dataset in datasets],
            "Contaminant Concentration": [dataset.dataframe for dataset in datasets],
        },
        {
            "Temperature": [dataset.filename for dataset in datasets],
            "Contaminant Concentration": [dataset.filename for dataset in datasets],
        },
        "Experimental",
        2,
        [2.0, 4.0, 6.0],
        ["Temperature", "Contaminant Concentration"],
        "Raw Profiles",
        8.75,
        temperature_columns=["Temp1", "Temp2", "Temp3"],
        tin=20.0,
        tout=30.0,
        contaminant_columns=["CO2_A", "CO2_B", "CO2_C"],
        contaminant_name="CO2",
        concentration_unit="ppm",
        cin=400.0,
        cout=900.0,
    )
    assert errors == []
    temp_h1 = [record for record in records if record.variable == "Temperature" and record.height_id == "H1"][0]
    co2_h1 = [record for record in records if record.variable == "Contaminant Concentration" and record.height_id == "H1"][0]
    assert temp_h1.replicate_standard_deviation == pd.Series(temp_h1.replicate_means).std(ddof=1)
    assert temp_h1.normalized_standard_deviation is None
    assert co2_h1.replicate_standard_deviation == pd.Series(co2_h1.replicate_means).std(ddof=1)
    assert co2_h1.normalized_standard_deviation is None


def test_vertical_profile_plot_uses_horizontal_replicate_error_bars_and_handles_single_replicate():
    datasets = replicate_frames()
    records, errors = vertical.build_replicate_vertical_profile_records(
        {"Air Velocity": [dataset.dataframe for dataset in datasets]},
        {"Air Velocity": [dataset.filename for dataset in datasets]},
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity"],
        "Raw Profiles",
        8.75,
    )
    assert errors == []
    if importlib.util.find_spec("matplotlib") is None:
        return
    fig = vertical.plot_vertical_profiles(records, ["Air Velocity"], "Raw Profiles", show_error_bars=True)
    assert len(fig.axes[0].collections) >= 1
    single_records, single_errors = vertical.build_replicate_vertical_profile_records(
        {"Air Velocity": [datasets[0].dataframe]},
        {"Air Velocity": [datasets[0].filename]},
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity"],
        "Raw Profiles",
        8.75,
    )
    assert single_errors == []
    assert all(record.replicate_standard_deviation is None for record in single_records)
    vertical.plot_vertical_profiles(single_records, ["Air Velocity"], "Raw Profiles", show_error_bars=True)


def sample_project_components():
    vector_records = [
        analysis.VectorRecord(1, 1, 2.67, 2.5, 2.0, 0.1, 0.2, 0.3, (0.1**2 + 0.2**2 + 0.3**2) ** 0.5, "p1.csv")
    ]
    vertical_records, errors = vertical.build_replicate_vertical_profile_records(
        {"Air Velocity": [dataset.dataframe for dataset in replicate_frames()]},
        {"Air Velocity": [dataset.filename for dataset in replicate_frames()]},
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity"],
        "Raw Profiles",
        8.75,
        supply_velocity=2.0,
    )
    assert errors == []
    processed, report, errors = replicate.rowwise_replicate_statistics(replicate_frames(), include_optional_numeric=False)
    assert errors == []
    replicate_state = {
        "position_number": 1,
        "case_number": "03",
        "case_date": "2026-08-25",
        "replicate_count": 3,
        "source_files": ["rep1.csv", "rep2.csv", "rep3.csv"],
        "alignment": {
            "method": report.method,
            "matched_rows": report.matched_rows,
            "unmatched_rows": report.unmatched_rows,
        },
        "base_name": "C03_P01_ReplicateMeanSD",
        "processed_dataframe": processed,
        "summary_dataframe": replicate.summary_statistics(processed),
    }
    return vector_records, vertical_records, replicate_state


def test_project_archive_creation_and_round_trip_for_all_analysis_state():
    vector_records, vertical_records, replicate_state = sample_project_components()
    state = project_io.build_project_state(
        case_number="03",
        case_date="2026-08-25",
        vector_records=vector_records,
        vector_settings={"camera": {"azimuth": 22, "elevation": 18, "roll": 1}},
        vertical_records=vertical_records,
        vertical_settings={
            "selected_variables": ["Air Velocity"],
            "profile_mode": "Raw Profiles",
            "show_error_bars": True,
            "contaminant_name": "CO2",
            "concentration_unit": "ppm",
            "normalization": {"room_height_ft": 8.75, "supply_velocity": 2.0, "tin": 20, "tout": 25, "cin": 400, "cout": 900},
        },
        replicate_analysis=replicate_state,
    )
    archive_bytes = project_io.save_project_archive(state)
    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert {"project.json", "metadata.json", "vector_summary.csv", "vertical_profile_summary.csv", "replicate_processed.csv", "replicate_summary.csv"}.issubset(set(archive.namelist()))
        project_json = json.loads(archive.read("project.json").decode("utf-8"))
        assert project_json["format_version"] == 1
        assert project_json["metadata"]["format_version"] == 1

    loaded = project_io.load_project_archive(archive_bytes)
    decoded = project_io.decode_project_state(loaded)
    assert decoded["vector_records"][0].position_number == 1
    assert decoded["vector_settings"]["camera"] == {"azimuth": 22, "elevation": 18, "roll": 1}
    assert decoded["vertical_settings"]["show_error_bars"] is True
    assert decoded["vertical_settings"]["contaminant_name"] == "CO2"
    assert decoded["vertical_settings"]["normalization"]["supply_velocity"] == 2.0
    assert decoded["vertical_records"][0].replicate_standard_deviation is not None
    assert decoded["replicate_analysis"]["alignment"]["matched_rows"] == 2
    loaded_processed = decoded["replicate_analysis"]["processed_dataframe"]
    assert list(loaded_processed.columns) == list(replicate_state["processed_dataframe"].columns)
    assert loaded_processed.shape == replicate_state["processed_dataframe"].shape
    assert list(loaded_processed["Vx1"]) == list(replicate_state["processed_dataframe"]["Vx1"])
    assert project_io.save_project_archive(loaded).startswith(b"PK")


def test_project_load_rejects_malformed_missing_and_unsupported_projects():
    try:
        project_io.load_project_archive(b"not a zip")
        assert False
    except project_io.ProjectLoadError as exc:
        assert "not a valid" in str(exc)

    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("metadata.json", "{}")
    try:
        project_io.load_project_archive(output.getvalue())
        assert False
    except project_io.ProjectLoadError as exc:
        assert "missing project.json" in str(exc)

    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("project.json", '{"format_version": 999}')
    try:
        project_io.load_project_archive(output.getvalue())
        assert False
    except project_io.ProjectLoadError as exc:
        assert "Unsupported project format_version" in str(exc)


def test_loaded_project_can_restore_session_and_regenerate_outputs():
    vector_records, vertical_records, replicate_state = sample_project_components()
    state = project_io.build_project_state(
        case_number="03",
        case_date="2026-08-25",
        vector_records=vector_records,
        vector_settings={"camera": {"azimuth": 12, "elevation": 13, "roll": 0}},
        vertical_records=vertical_records,
        vertical_settings={"selected_variables": ["Air Velocity"], "profile_mode": "Raw Profiles", "show_error_bars": True},
        replicate_analysis=replicate_state,
    )
    app = load_app_module()
    app.st.session_state = {}
    summary = app.restore_project_to_session(project_io.load_project_archive(project_io.save_project_archive(state)))
    assert summary["project"] == "03"
    assert app.st.session_state["analysis_vector_records"][0].source_filename == "p1.csv"
    assert app.st.session_state["analysis_camera_azimuth"] == 12
    assert app.st.session_state["vertical_profile_show_error_bars"] is True
    assert app.st.session_state["replicate_analysis_state"]["summary_dataframe"] is not None
    si_fig = analysis.plot_3d_air_vectors(app.st.session_state["analysis_vector_records"], "si", 12, 13, 0)
    assert analysis.export_figure_png(si_fig).startswith(b"\x89PNG")
    vp_fig = vertical.plot_vertical_profiles(app.st.session_state["vertical_profile_records"], ["Air Velocity"], "Raw Profiles", show_error_bars=True)
    assert vertical.export_vertical_profile_png(vp_fig).startswith(b"\x89PNG")


def test_start_new_analysis_clears_only_data_analysis_state():
    app = load_app_module()
    app.st.session_state = {
        "dat_completed_cases": ["keep"],
        "analysis_vector_records": [analysis.VectorRecord(1, 1, 2.67, 2.5, 2.0, 0.1, 0.2, 0.3, 0.4, "p1.csv")],
        "vertical_profile_records": ["profile"],
        "replicate_analysis_state": {"x": 1},
    }
    app.clear_data_analysis_project_state()
    assert app.st.session_state["dat_completed_cases"] == ["keep"]
    assert app.st.session_state["analysis_vector_records"] == []
    assert app.st.session_state["vertical_profile_records"] == []
    assert app.st.session_state["replicate_analysis_state"] == {}


def test_vertical_validation_rejects_missing_inputs_and_future_cfd():
    errors = vertical.validate_profile_heights([0, 9], 8.75)
    assert "Height 1 must be greater than 0 ft." in errors
    assert "Height 2 must be less than or equal to the room height." in errors
    norm_errors = vertical.validate_normalization_inputs(
        ["Air Velocity", "Temperature", "Contaminant Concentration"],
        "Raw Profiles",
        supply_velocity=0,
        tin=20,
        tout=20,
        cin=1,
        cout=1,
    )
    assert norm_errors == []
    _records, cfd_errors = vertical.build_vertical_profile_records(
        vertical_profile_dataframe(),
        "CFD",
        1,
        [1, 2, 3],
        ["Air Velocity"],
        "Raw Profiles",
        8.75,
        "cfd.csv",
    )
    assert cfd_errors == ["CFD input support is reserved for a future update."]


def test_vertical_plot_exports_and_zip():
    records, errors = vertical.build_vertical_profile_records(
        vertical_profile_dataframe(),
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity", "Temperature"],
        "Raw Profiles",
        8.75,
        "p1.csv",
        temperature_columns=["Temp1", "Temp2", "Temp3"],
    )
    assert errors == []
    if importlib.util.find_spec("matplotlib") is None:
        requirements = (PROJECT_ROOT / "requirements.txt").read_text()
        assert "matplotlib" in requirements
    else:
        si_fig = vertical.plot_vertical_profiles(records, ["Air Velocity", "Temperature"], "Raw Profiles", unit_system="si")
        ip_fig = vertical.plot_vertical_profiles(records, ["Air Velocity", "Temperature"], "Raw Profiles", unit_system="ip")
        assert vertical.export_vertical_profile_eps(si_fig).startswith(b"%!PS-Adobe")
        assert vertical.export_vertical_profile_png(si_fig).startswith(b"\x89PNG")
        assert vertical.export_vertical_profile_eps(ip_fig).startswith(b"%!PS-Adobe")
        assert vertical.export_vertical_profile_png(ip_fig).startswith(b"\x89PNG")
    csv_bytes = vertical.vertical_profile_summary_csv(records)
    xlsx_bytes = vertical.vertical_profile_summary_xlsx(records)
    zip_bytes = vertical.build_vertical_profile_zip(
        [
            ("Vertical_Profiles_SI.eps", b"eps"),
            ("Vertical_Profiles_SI.png", b"png"),
            ("Vertical_Profiles_IP.eps", b"eps"),
            ("Vertical_Profiles_IP.png", b"png"),
            ("Vertical_Profile_Summary.csv", csv_bytes),
            ("Vertical_Profile_Summary.xlsx", xlsx_bytes),
        ]
    )
    with ZipFile(BytesIO(zip_bytes)) as archive:
        assert set(archive.namelist()) == {
            "Vertical_Profiles_SI.eps",
            "Vertical_Profiles_SI.png",
            "Vertical_Profiles_IP.eps",
            "Vertical_Profiles_IP.png",
            "Vertical_Profile_Summary.csv",
            "Vertical_Profile_Summary.xlsx",
        }


def test_vertical_plot_uses_separate_figure_title_and_shared_legend():
    records, errors = vertical.build_vertical_profile_records(
        vertical_profile_dataframe(),
        "Experimental",
        1,
        [2.0, 4.0, 6.0],
        ["Air Velocity", "Temperature", "Contaminant Concentration"],
        "Raw Profiles",
        8.75,
        "p1.csv",
        supply_velocity=2.5,
        temperature_columns=["Temp1", "Temp2", "Temp3"],
        tin=20.0,
        tout=25.0,
        contaminant_columns=["CO2_A", "CO2_B", "CO2_C"],
        contaminant_name="CO2",
        concentration_unit="ppm",
        cin=400.0,
        cout=800.0,
    )
    assert errors == []
    if importlib.util.find_spec("matplotlib") is None:
        return

    fig = vertical.plot_vertical_profiles(
        records,
        ["Air Velocity", "Temperature", "Contaminant Concentration"],
        "Raw Profiles",
        "CO2",
        "ppm",
    )
    assert fig._suptitle.get_text() == "Vertical Profiles — SI Units"
    assert len(fig.legends) == 1
    assert [text.get_text() for text in fig.legends[0].texts] == ["P1"]
    assert all(axis.get_legend() is None for axis in fig.axes)
    assert [axis.get_title() for axis in fig.axes] == [
        "Air Velocity",
        "Temperature",
        "CO2 Concentration",
    ]
    assert vertical.export_vertical_profile_eps(fig).startswith(b"%!PS-Adobe")
    assert vertical.export_vertical_profile_png(fig).startswith(b"\x89PNG")


def test_app_data_analysis_outputs_include_si_imperial_and_zip_files():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    expected_names = [
        "3D_Air_Velocity_SI.eps",
        "3D_Air_Velocity_SI.png",
        "3D_Air_Velocity_Imperial.eps",
        "3D_Air_Velocity_Imperial.png",
        "Air_Velocity_Summary_SI.csv",
        "Air_Velocity_Summary_SI.xlsx",
        "Air_Velocity_Summary_Imperial.csv",
        "Air_Velocity_Summary_Imperial.xlsx",
    ]
    for expected_name in expected_names:
        assert expected_name in app_source
    assert "Download All Analysis Files (ZIP)" in app_source
    assert "plot_3d_air_vectors(vector_records, \"si\", azimuth, elevation, roll)" in app_source
    assert "plot_3d_air_vectors(vector_records, \"imperial\", azimuth, elevation, roll)" in app_source


def test_app_data_analysis_camera_controls_are_present():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert "ANALYSIS_CAMERA_DEFAULTS" in app_source
    assert "render_analysis_camera_controls" in app_source
    assert "Azimuth (deg)" in app_source
    assert "Elevation (deg)" in app_source
    assert "Roll (deg)" in app_source
    assert "Apply View" in app_source
    assert "Reset View" in app_source
    assert "Perspective" in app_source
    assert "Front" in app_source
    assert "Side" in app_source
    assert "Top" in app_source
    assert "plot_interactive_air_vectors(vector_records, \"si\", azimuth, elevation, roll)" in app_source
    assert "st.plotly_chart" in app_source
    assert "Use Current View" not in app_source
    assert "components.declare_component" not in app_source
    assert "plotly_camera" not in app_source
    assert "analysis_latest_plotly_camera" not in app_source


def test_app_camera_state_uses_pending_updates_for_widget_sync():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    set_camera_body = app_source.split("def set_analysis_camera", 1)[1].split(
        "def queue_analysis_camera_update",
        1,
    )[0]
    assert "_slider" not in set_camera_body
    assert "_number" not in set_camera_body
    assert "analysis_camera_pending" in app_source
    assert "apply_pending_analysis_camera_update()" in app_source


def test_app_pending_camera_update_syncs_canonical_and_widgets_before_render():
    app = load_app_module()
    app.st.session_state = {}
    app.init_analysis_state()
    app.queue_analysis_camera_update(15.0, 35.0, 5.0)
    app.apply_pending_analysis_camera_update()
    assert app.st.session_state["analysis_camera_azimuth"] == 15.0
    assert app.st.session_state["analysis_camera_elevation"] == 35.0
    assert app.st.session_state["analysis_camera_roll"] == 5.0
    assert app.st.session_state["analysis_camera_azimuth_slider"] == 15.0
    assert "analysis_camera_pending" not in app.st.session_state


def test_app_has_no_custom_plotly_camera_component_dependency():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    requirements = (PROJECT_ROOT / "requirements.txt").read_text()
    assert "streamlit.components" not in app_source
    assert "declare_component" not in app_source
    assert "app.plotly_camera" not in app_source
    assert "plotly_camera" not in app_source
    assert "streamlit-plotly-events" not in requirements
    assert not (PROJECT_ROOT / "components" / "plotly_camera" / "index.html").exists()


def test_app_manual_slider_changes_update_canonical_camera():
    app = load_app_module()
    app.st.session_state = {
        "analysis_camera_azimuth_slider": 42.0,
    }
    app.sync_analysis_camera_from_widget("azimuth", "analysis_camera_azimuth_slider")
    assert app.st.session_state["analysis_camera_azimuth"] == 42.0


def test_app_uses_data_analysis_tab_instead_of_split_by_time_tab():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    tabs_line = [
        line for line in app_source.splitlines() if "st.tabs" in line and "DAT" in line
    ][0]
    assert "Data Analysis" in tabs_line
    assert "Merge CSV" in tabs_line
    assert "Split by Time" not in tabs_line
    assert "render_data_analysis_tool" in app_source


def test_data_analysis_landing_contains_3d_and_vertical_methods():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    landing_body = app_source.split("def render_data_analysis_landing", 1)[1].split(
        "def render_data_analysis_tool",
        1,
    )[0]
    assert "Choose an analysis method" in app_source
    assert "3D Vector Plot" in landing_body
    assert "Visualize 3D airflow direction and velocity magnitude in the room." in landing_body
    assert "Vertical Profile Plot" in landing_body
    assert "Plot vertical distributions of air velocity, temperature, and contaminant concentration." in landing_body
    assert "Replicate Mean & SD" not in landing_body
    assert "select_replicate_mean_sd" not in landing_body
    assert "render_3d_vector_plot_tool" in app_source
    assert "render_vertical_profile_tool" in app_source


def test_replicate_mean_sd_is_under_merge_csv():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    merge_body = app_source.split("def render_merge_csv_tool", 1)[1].split(
        "def render_workbook_analysis",
        1,
    )[0]
    assert "Processing mode" in merge_body
    assert "Standard Merge" in merge_body
    assert "Replicate Mean & SD" in merge_body
    assert "render_replicate_mean_sd_tool(show_back_button=False)" in merge_body


def test_replicate_mean_sd_ui_has_three_required_uploaders_and_guard():
    app = load_app_module()
    assert app.validate_required_replicate_uploads([object(), None, None]) == (
        [],
        "Please upload all 3 replicate CSV files before processing.",
    )
    assert app.validate_required_replicate_uploads([object(), object(), None]) == (
        [],
        "Please upload all 3 replicate CSV files before processing.",
    )
    ready_uploads, error = app.validate_required_replicate_uploads([object(), object(), object()])
    assert error is None
    assert len(ready_uploads) == 3

    app_source = (PROJECT_ROOT / "app.py").read_text()
    replicate_body = app_source.split("def render_replicate_mean_sd_tool", 1)[1].split(
        "def render_project_controls",
        1,
    )[0]
    assert "Replicate 1 CSV" in replicate_body
    assert "Replicate 2 CSV" in replicate_body
    assert "Replicate 3 CSV" in replicate_body
    assert "key=f\"replicate_file_1_{input_version}\"" in replicate_body
    assert "key=f\"replicate_file_2_{input_version}\"" in replicate_body
    assert "key=f\"replicate_file_3_{input_version}\"" in replicate_body
    assert "accept_multiple_files=False" in replicate_body
    assert "accept_multiple_files=True" not in replicate_body
    assert "for uploaded_file in ready_uploads" in replicate_body
    assert "rowwise_replicate_statistics(" in replicate_body


def test_replicate_mean_sd_feedback_next_case_and_duplicate_workflow_source():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    replicate_body = app_source.split("def render_replicate_mean_sd_tool", 1)[1].split(
        "def render_project_controls",
        1,
    )[0]
    assert 'st.spinner("Processing replicate files...")' in replicate_body
    assert "Replicate processing completed successfully." in replicate_body
    assert "awaiting_next_case_choice" in replicate_body
    assert "## Process another case?" in replicate_body
    assert "Yes, process another case" in replicate_body
    assert "No, finish" in replicate_body
    assert "reset_current_replicate_inputs()" in replicate_body
    assert "has already been processed in this session" in replicate_body
    assert "I want to intentionally reprocess this case" in replicate_body
    assert "completed_cases.append(completed_case)" in replicate_body


def test_replicate_next_case_reset_preserves_completed_cases_and_other_state():
    app = load_app_module()
    processed, report, errors = replicate.rowwise_replicate_statistics(replicate_frames(), include_optional_numeric=False)
    assert errors == []
    completed_case = {
        "case_key": app.replicate_case_key("03", 1, "2026-08-25"),
        "position_number": 1,
        "case_number": "03",
        "case_date": "2026-08-25",
        "aligned_observations": report.matched_rows,
        "base_name": "C03_P01_ReplicateMeanSD",
        "summary_base_name": "C03_P01_ReplicateSummary",
        "processed_dataframe": processed,
        "summary_dataframe": replicate.summary_statistics(processed),
    }
    app.st.session_state = {
        "replicate_input_version": 4,
        "replicate_analysis_state": {
            "completed_cases": [completed_case],
            "processed_dataframe": processed,
            "summary_dataframe": replicate.summary_statistics(processed),
            "base_name": "C03_P01_ReplicateMeanSD",
            "awaiting_next_case_choice": True,
        },
        "dat_completed_cases": ["keep-dat"],
        "analysis_vector_records": ["keep-vector"],
        "vertical_profile_records": ["keep-vertical"],
        "project_metadata": {"case_number": "keep-project"},
    }
    app.reset_current_replicate_inputs()
    assert app.st.session_state["replicate_input_version"] == 5
    state = app.st.session_state["replicate_analysis_state"]
    assert state["completed_cases"] == [completed_case]
    assert state["processed_dataframe"] is None
    assert state["summary_dataframe"] is None
    assert state["base_name"] is None
    assert state["awaiting_next_case_choice"] is False
    assert app.st.session_state["dat_completed_cases"] == ["keep-dat"]
    assert app.st.session_state["analysis_vector_records"] == ["keep-vector"]
    assert app.st.session_state["vertical_profile_records"] == ["keep-vertical"]
    assert app.st.session_state["project_metadata"] == {"case_number": "keep-project"}


def test_replicate_completed_cases_summary_zip_and_duplicate_detection():
    app = load_app_module()
    processed, report, errors = replicate.rowwise_replicate_statistics(replicate_frames(), include_optional_numeric=False)
    assert errors == []
    case_one = {
        "case_key": app.replicate_case_key("03", 1, "2026-08-25"),
        "position_number": 1,
        "case_number": "03",
        "case_date": "2026-08-25",
        "aligned_observations": report.matched_rows,
        "base_name": "C03_P01_ReplicateMeanSD",
        "summary_base_name": "C03_P01_ReplicateSummary",
        "processed_dataframe": processed,
        "summary_dataframe": replicate.summary_statistics(processed),
    }
    case_two = {
        **case_one,
        "case_key": app.replicate_case_key("03", 2, "2026-08-25"),
        "position_number": 2,
        "base_name": "C03_P02_ReplicateMeanSD",
        "summary_base_name": "C03_P02_ReplicateSummary",
    }
    app.st.session_state = {"replicate_analysis_state": {"completed_cases": [case_one, case_two]}}
    assert app.is_duplicate_replicate_case("03", 1, "2026-08-25") is True
    assert app.is_duplicate_replicate_case("03", 3, "2026-08-25") is False
    zip_bytes = app.build_all_replicate_cases_zip(app.completed_replicate_cases())
    with ZipFile(BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
        assert "C03_P01_ReplicateMeanSD.csv" in names
        assert "C03_P01_ReplicateMeanSD.xlsx" in names
        assert "C03_P02_ReplicateMeanSD.csv" in names
        assert "C03_P02_ReplicateMeanSD.xlsx" in names


def test_vertical_profile_workflow_source_contains_required_controls():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert "Reset Vertical Profile Analysis" in app_source
    assert "Continue this case" in app_source
    assert "Start new case analysis" in app_source
    assert "Standard deviation columns were not found, so the profile was plotted without error bars." in app_source
    assert "Dataset Type" in app_source
    assert "CFD input support is reserved for a future update." in app_source
    assert "Air Velocity" in app_source
    assert "Temperature" in app_source
    assert "Contaminant Concentration" in app_source
    assert "Raw Profiles" in app_source
    assert "Normalized Profiles" not in app_source
    assert "Room Height H (ft)" not in app_source
    assert "Supply Air Velocity Us (m/s)" not in app_source
    assert "Supply / Inlet Temperature Tin" not in app_source
    assert "Exhaust / Outlet Temperature Tout" not in app_source
    assert "Supply / Background Concentration Cin" not in app_source
    assert "Exhaust Concentration Cout" not in app_source
    assert "Velocity CSV upload" in app_source
    assert "Temperature CSV upload" in app_source
    assert "Contaminant CSV upload" in app_source
    assert "Use Velocity CSV for Temperature" in app_source
    assert "Use Velocity CSV for Contaminant Concentration" in app_source
    assert "Height 1 Temperature Column" in app_source
    assert "Height 1 Concentration Column" in app_source
    assert "Download All Vertical Profile Files" in app_source
    assert "Show replicate SD error bars" in app_source
    assert "Use replicate files for this position" in app_source


def run_all_tests():
    tests = [
        test_standard_split,
        test_different_user_defined_times_are_dynamic,
        test_no_exact_boundary_observation,
        test_row_accounting_and_no_duplicates,
        test_invalid_cut_order,
        test_cut_outside_experimental_range,
        test_empty_segment_rejected,
        test_missing_timestamp_is_graceful,
        test_unparseable_timestamp_is_reported,
        test_multiple_worksheets_are_split_with_same_boundaries,
        test_multiple_uploaded_files_can_use_different_times,
        test_output_naming_and_zip_generation,
        test_existing_dat_to_xlsx_still_works,
        test_integrated_dat_without_splitting_still_generates_full_xlsx,
        test_integrated_dat_with_splitting_generates_full_and_positions,
        test_integrated_dat_splitting_uses_user_defined_times_per_file,
        test_position_velocity_transform_p01_and_p03_use_first_rule,
        test_position_velocity_transform_p04_and_p06_use_second_rule,
        test_position_velocity_transform_column_order_and_nan_handling,
        test_hhmm_text_validation,
        test_hhmm_part_validation_accepts_and_normalizes_values,
        test_hhmm_part_validation_rejects_blank_invalid_and_out_of_range_values,
        test_hhmm_parts_reconstruct_to_position_range_times,
        test_start_time_options_use_dataset_minute_range,
        test_duration_options_and_validation,
        test_duration_range_uses_half_open_interval,
        test_duration_range_extending_beyond_dataset_is_rejected,
        test_duration_range_empty_position_is_rejected,
        test_duration_range_adjacent_positions_are_valid,
        test_multiday_workbook_analysis_and_date_time_options,
        test_date_start_time_combines_to_full_datetime_and_cross_midnight_end_format,
        test_multiday_split_uses_full_datetime_half_open_slicing,
        test_cross_midnight_split_uses_full_datetimes,
        test_repeated_position_duplicate_rules_use_position_and_full_start_datetime,
        test_same_position_same_time_different_date_is_valid,
        test_repeated_position_output_filenames_include_start_time_when_needed,
        test_integrated_dat_batch_output_contains_xlsx_and_csv_files,
        test_integrated_dat_positions_have_transformed_cleaned_data_only,
        test_dat_full_and_position_csv_content_rules,
        test_dat_p04_csv_uses_second_transform_rule,
        test_current_case_zip_contains_xlsx_and_csv_without_raw_csv,
        test_integrated_dat_batch_many_files_zip_generation,
        test_case_and_filename_formatting,
        test_no_split_generates_only_all_workbook_with_complete_data,
        test_split_yes_with_2_3_5_and_6_positions,
        test_dat_case_validation_allows_split_count_above_six,
        test_position_range_start_inclusive_and_end_minute_inclusive,
        test_position_gaps_are_allowed_without_lost_row_qc_failure,
        test_position_overlaps_are_allowed,
        test_position_start_later_than_end_is_rejected,
        test_position_with_zero_observations_is_rejected,
        test_duplicate_case_date_detection_helper,
        test_case_validation_accepts_entered_case_number,
        test_case_validation_reports_empty_case_and_missing_date,
        test_case_validation_reports_incomplete_splits,
        test_dat_workflow_source_has_no_empty_edt_card_placeholder,
        test_dat_workflow_source_removes_technical_details_and_preview,
        test_dat_workflow_has_no_fixed_batch_limit,
        test_dat_workflow_source_uses_splits_and_limits_physical_positions_to_six,
        test_dat_workflow_source_has_processing_status_messages,
        test_dat_workflow_source_uses_start_time_and_duration_inputs,
        test_current_result_survives_session_state_rerun,
        test_process_next_file_preserves_completed_cases_before_reset,
        test_download_keys_are_unique_across_cases_and_positions,
        test_light_theme_config_exists,
        test_existing_merge_csv_still_works,
        test_analysis_csv_validation_and_mean_vectors,
        test_analysis_vector_records_coordinates_and_duplicates,
        test_analysis_display_vectors_are_normalized_without_changing_true_values,
        test_analysis_si_and_imperial_unit_conversions_are_correct,
        test_analysis_si_and_imperial_display_scaling_is_equivalent,
        test_analysis_position_coordinates_are_exact_and_not_swapped,
        test_analysis_plot_source_uses_velocity_colormap_colorbar_and_no_position_labels,
        test_analysis_camera_defaults_and_helpers,
        test_analysis_plot_exports_eps_and_png,
        test_analysis_interactive_plot_dependency_is_declared_or_available,
        test_interactive_origin_markers_match_vector_start_points,
        test_vertical_velocity_profile_averages_instantaneous_speed,
        test_vertical_profile_raw_mode_is_the_only_profile_mode,
        test_vertical_temperature_auto_and_manual_mapping,
        test_vertical_contaminant_mapping_unit_and_summary,
        test_vertical_build_records_for_three_variables_and_height_order,
        test_vertical_profile_variables_can_use_independent_source_files,
        test_vertical_profile_missing_selected_variable_remains_blank_in_unified_summary,
        test_replicate_rowwise_mean_sd_measurement_sequence_alignment_and_exports,
        test_replicate_measurement_sequence_uses_common_minimum_row_count,
        test_replicate_matching_columns_missing_column_rejection,
        test_replicate_vertical_profile_mean_speed_and_between_replicate_sd,
        test_vertical_profile_uses_measurement_std_error_bars_for_multiple_positions,
        test_vertical_profile_si_ip_axis_limits_and_unit_conversions,
        test_vertical_profile_uses_raw_measurement_std_for_error_bars,
        test_vertical_profile_reads_replicate_generated_std_columns,
        test_replicate_temperature_and_contaminant_between_replicate_sd_and_normalized_sd,
        test_vertical_profile_plot_uses_horizontal_replicate_error_bars_and_handles_single_replicate,
        test_project_archive_creation_and_round_trip_for_all_analysis_state,
        test_project_load_rejects_malformed_missing_and_unsupported_projects,
        test_loaded_project_can_restore_session_and_regenerate_outputs,
        test_start_new_analysis_clears_only_data_analysis_state,
        test_vertical_validation_rejects_missing_inputs_and_future_cfd,
        test_vertical_plot_exports_and_zip,
        test_vertical_plot_uses_separate_figure_title_and_shared_legend,
        test_app_data_analysis_outputs_include_si_imperial_and_zip_files,
        test_app_data_analysis_camera_controls_are_present,
        test_app_camera_state_uses_pending_updates_for_widget_sync,
        test_app_pending_camera_update_syncs_canonical_and_widgets_before_render,
        test_app_has_no_custom_plotly_camera_component_dependency,
        test_app_manual_slider_changes_update_canonical_camera,
        test_app_uses_data_analysis_tab_instead_of_split_by_time_tab,
        test_data_analysis_landing_contains_3d_and_vertical_methods,
        test_replicate_mean_sd_is_under_merge_csv,
        test_replicate_mean_sd_ui_has_three_required_uploaders_and_guard,
        test_replicate_mean_sd_feedback_next_case_and_duplicate_workflow_source,
        test_replicate_next_case_reset_preserves_completed_cases_and_other_state,
        test_replicate_completed_cases_summary_zip_and_duplicate_detection,
        test_vertical_profile_workflow_source_contains_required_controls,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    run_all_tests()

