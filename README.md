# Experimental Data Toolkit

Experimental Data Toolkit is a lightweight Streamlit web application for routine experimental data processing.

## What It Does

The MVP includes two tools:

- **DAT → XLSX**: Upload up to 10 experimental `.dat` files, clean the sensor columns, enter per-file cut times, split each file into three positions, and download individual XLSX outputs or one ZIP.
- **Merge CSV Files**: Upload two or more `.csv` files, preview each file, merge them vertically by rows, and download the merged result as CSV or XLSX.
- **Split by Time**: Upload one or more experimental `.xlsx` time-series workbooks, select two per-file cut times, split every timestamped worksheet into three position files, and download the results.

The application reads uploaded files in memory and does not modify the original input files.

## Install Dependencies

From the project directory, install the required packages:

```bash
pip install -r requirements.txt
```

## Start the Application

Run:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Supported Tools

### DAT → XLSX

- Upload one to ten DAT files.
- Use the second DAT row as source column names and rows 5 onward as observations.
- Keep and rename the required sensor columns to `TIMESTAMP`, `U1`, `V1`, `W1`, `Temp1`, `U2`, `V2`, `W2`, `Temp2`, `U3`, `V3`, `W3`, and `Temp3`.
- Remove unneeded columns such as `RECORD`, `A_SensorStatus`, `B_SensorStatus`, and `C_SensorStatus` from the cleaned output.
- Create XLSX workbooks with `Cleaned_Data` and `Raw_Data` worksheets.
- Enter independent `HH:MM` cut times for every uploaded DAT file.
- Split each processed DAT file into Position 1, Position 2, and Position 3 using the validated time-splitting logic.
- Download full XLSX and position XLSX files individually, or download all generated outputs as one ZIP.

### Merge CSV Files

- Upload two or more CSV files.
- Merge files vertically by rows.
- Choose between:
  - **Require identical columns**: all files must have the same column names and column order.
  - **Keep all columns**: creates the union of all columns and preserves missing values as blank/NaN.
- Optionally add a `_source_file` column for traceability.
- Preview the merged result.
- Download the merged output as CSV or XLSX.

### Split by Time

- Upload one or multiple XLSX files.
- Parse the `TIMESTAMP` column as real datetimes before splitting.
- Select per-file `End of Position 1` and `End of Position 2` times.
- Convert selected HH:MM cut times into dynamic minute boundaries.
- Split every worksheet containing a valid `TIMESTAMP` column into three position worksheets.
- Preserve worksheets without `TIMESTAMP` unchanged in each output workbook.
- Download position files individually for one upload or as ZIP archives.
- Run QC checks to confirm all original rows are accounted for and no observations are duplicated.

## Run Tests

The split logic and regression checks can be run with:

```bash
python tests/test_time_splitter.py
```
