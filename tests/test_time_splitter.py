from __future__ import annotations

from datetime import time
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import importlib.util
import sys
import types

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import dat_batch
import dat_processor as dat
import time_splitter as splitter


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
            splitter.PositionRange("P01", time(16, 39), time(16, 39)),
            splitter.PositionRange("P02", time(16, 40), time(16, 40)),
            splitter.PositionRange("P03", time(16, 41), time(16, 41)),
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
            splitter.PositionRange("P01", time(16, 39), time(16, 39)),
            splitter.PositionRange("P02", time(16, 40), time(16, 40)),
            splitter.PositionRange("P03", time(16, 41), time(16, 41)),
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
            splitter.PositionRange("P01", time(16, 39), time(16, 39)),
            splitter.PositionRange("P02", time(16, 40), time(16, 40)),
            splitter.PositionRange("P03", time(16, 41), time(16, 41)),
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
            splitter.PositionRange("P01", time(16, 39), time(16, 39)),
            splitter.PositionRange("P02", time(16, 40), time(16, 40)),
            splitter.PositionRange("P03", time(16, 41), time(16, 41)),
            splitter.PositionRange("P04", time(16, 42), time(16, 42)),
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
            splitter.PositionRange("P01", time(16, 39), time(16, 39)),
            splitter.PositionRange("P02", time(16, 40), time(16, 40)),
            splitter.PositionRange("P03", time(16, 41), time(16, 41)),
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
                splitter.PositionRange("P01", time(16, 39), time(16, 39)),
                splitter.PositionRange("P02", time(16, 40), time(16, 40)),
                splitter.PositionRange("P03", time(16, 41), time(16, 41)),
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
        splitter.PositionRange(f"P{index + 1:02d}", time(13, index * 2), time(13, index * 2))
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


def test_dat_case_validation_rejects_position_count_above_six():
    app = load_app_module()
    app.st.session_state = {"dat_completed_cases": []}
    case_number, errors = app.validate_case_inputs(
        "3",
        pd.Timestamp("2026-08-26").date(),
        True,
        7,
        [splitter.PositionRange(f"P{index + 1:02d}", time(13, index), time(13, index)) for index in range(7)],
        [],
    )
    assert case_number == "03"
    assert "Split Position count must be between 2 and 6." in errors


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


def test_position_overlaps_are_rejected():
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
    assert result is None
    assert error == "P01 and P02 contain overlapping time ranges."


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


def test_case_validation_reports_incomplete_positions():
    app = load_app_module()
    app.st.session_state = {"dat_completed_cases": []}
    case_number, errors = app.validate_case_inputs(
        "3",
        pd.Timestamp("2026-08-26").date(),
        True,
        3,
        [splitter.PositionRange("P01", time(13, 0), time(13, 30))],
        ["Please enter P02 Start Hour."],
    )
    assert case_number == "03"
    assert "Please enter Start Time and End Time for all Positions." in errors
    assert "Please enter P02 Start Hour." in errors


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


def test_dat_workflow_source_limits_positions_to_six():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert "MAX_DAT_POSITIONS = 6" in app_source
    assert "max_value=MAX_DAT_POSITIONS" in app_source
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


def test_dat_workflow_source_uses_separate_hh_mm_inputs():
    app_source = (PROJECT_ROOT / "app.py").read_text()
    assert "Start HH" in app_source
    assert "Start MM" in app_source
    assert "End HH" in app_source
    assert "End MM" in app_source
    assert "build_time_range_from_parts" in app_source
    assert "placeholder=\"HH\"" in app_source
    assert "placeholder=\"MM\"" in app_source


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
        test_integrated_dat_batch_output_contains_xlsx_and_csv_files,
        test_integrated_dat_positions_have_transformed_cleaned_data_only,
        test_dat_full_and_position_csv_content_rules,
        test_dat_p04_csv_uses_second_transform_rule,
        test_current_case_zip_contains_xlsx_and_csv_without_raw_csv,
        test_integrated_dat_batch_many_files_zip_generation,
        test_case_and_filename_formatting,
        test_no_split_generates_only_all_workbook_with_complete_data,
        test_split_yes_with_2_3_5_and_6_positions,
        test_dat_case_validation_rejects_position_count_above_six,
        test_position_range_start_inclusive_and_end_minute_inclusive,
        test_position_gaps_are_allowed_without_lost_row_qc_failure,
        test_position_overlaps_are_rejected,
        test_position_start_later_than_end_is_rejected,
        test_position_with_zero_observations_is_rejected,
        test_duplicate_case_date_detection_helper,
        test_case_validation_accepts_entered_case_number,
        test_case_validation_reports_empty_case_and_missing_date,
        test_case_validation_reports_incomplete_positions,
        test_dat_workflow_source_has_no_empty_edt_card_placeholder,
        test_dat_workflow_source_removes_technical_details_and_preview,
        test_dat_workflow_has_no_fixed_batch_limit,
        test_dat_workflow_source_limits_positions_to_six,
        test_dat_workflow_source_has_processing_status_messages,
        test_dat_workflow_source_uses_separate_hh_mm_inputs,
        test_current_result_survives_session_state_rerun,
        test_process_next_file_preserves_completed_cases_before_reset,
        test_download_keys_are_unique_across_cases_and_positions,
        test_light_theme_config_exists,
        test_existing_merge_csv_still_works,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    run_all_tests()

