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


def test_integrated_dat_batch_output_contains_four_files():
    app = load_app_module()
    processed, error = dat.process_dat_bytes(sample_dat_text().encode("utf-8"), "P1_Round1.dat")
    assert error is None
    batch_output, batch_error = app.build_dat_batch_output(processed, time(16, 39), time(16, 40))
    assert batch_error is None
    assert [filename for filename, _bytes in batch_output.files] == [
        "P1_Round1.xlsx",
        "P1_Round1_position1.xlsx",
        "P1_Round1_position2.xlsx",
        "P1_Round1_position3.xlsx",
    ]
    for filename, output_bytes in batch_output.files:
        workbook = pd.read_excel(BytesIO(output_bytes), sheet_name=None, engine="openpyxl")
        assert set(workbook) == {"Cleaned_Data", "Raw_Data"}
        assert filename.endswith(".xlsx")


def test_integrated_dat_batch_ten_files_zip_generation():
    app = load_app_module()
    all_outputs = []
    for index in range(10):
        processed, error = dat.process_dat_bytes(
            sample_dat_text().encode("utf-8"),
            f"P{index + 1}_Round1.dat",
        )
        assert error is None
        batch_output, batch_error = app.build_dat_batch_output(
            processed,
            time(16, 39),
            time(16, 40),
        )
        assert batch_error is None
        all_outputs.extend(batch_output.files)

    assert len(all_outputs) == 40
    archive_bytes = dat.build_xlsx_zip(all_outputs)
    with ZipFile(BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        assert len(names) == 40
        assert "P1_Round1.xlsx" in names
        assert "P10_Round1_position3.xlsx" in names


def test_dat_upload_limit_constant_is_ten():
    app = load_app_module()
    assert app.MAX_DAT_UPLOADS == 10


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
        test_hhmm_text_validation,
        test_integrated_dat_batch_output_contains_four_files,
        test_integrated_dat_batch_ten_files_zip_generation,
        test_dat_upload_limit_constant_is_ten,
        test_light_theme_config_exists,
        test_existing_merge_csv_still_works,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
