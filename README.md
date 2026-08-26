# Experimental Data Toolkit

Experimental Data Toolkit is a lightweight Streamlit web application for routine experimental data processing.

## What It Does

The MVP includes three tools:

- **DAT → XLSX**: Process one experimental `.dat` file at a time, assign Case No and Case Date, optionally split into 2 to 6 user-defined Positions, and download XLSX and CSV outputs.
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

- Upload and process one DAT file at a time.
- Continue adding completed DAT files to the current batch until you choose to finish it.
- Use the second DAT row as source column names and rows 5 onward as observations.
- Keep and rename the required sensor columns to `TIMESTAMP`, `U1`, `V1`, `W1`, `Temp1`, `U2`, `V2`, `W2`, `Temp2`, `U3`, `V3`, `W3`, and `Temp3`.
- Remove unneeded columns such as `RECORD`, `A_SensorStatus`, `B_SensorStatus`, and `C_SensorStatus` from the cleaned output.
- Enter required Case No and Case Date for output naming.
- Generate the complete Full Dataset for every processed case.
- Choose Split Yes or No for each file.
- If Split is Yes, choose 2 through 6 Positions and enter separate Start HH, Start MM, End HH, and End MM values for each Position.
- Position gaps are allowed; overlaps are rejected.
- Position workbooks add derived velocity columns in `Cleaned_Data` only: P01-P03 use `Vx = W`, `Vy = -U`, `Vz = -V`; P04-P06 use `Vx = -W`, `Vy = U`, `Vz = -V`.
- The Full Dataset XLSX remains untransformed, and every XLSX contains `Cleaned_Data` and `Raw_Data`.
- CSV outputs contain `Cleaned_Data` only; no Raw CSV files are created.
- Output filenames use the Case No and Case Date, such as `C03_All_26.08.26.xlsx`, `C03_All_26.08.26.csv`, `C03_P01_26.08.26.xlsx`, and `C03_P01_26.08.26.csv`.
- Download the current case outputs immediately, process the next file, or finish the batch.
- Download all generated XLSX and CSV outputs for the current case or final batch as one ZIP.

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
