# Experimental Data Toolkit

Experimental Data Toolkit is a lightweight Streamlit web application for routine experimental data processing.

## What It Does

The MVP includes two tools:

- **DAT → XLSX**: Upload one or more experimental `.dat` files, clean the sensor columns, preview the cleaned data, and download processed `.xlsx` workbooks individually or as a ZIP.
- **Merge CSV Files**: Upload two or more `.csv` files, preview each file, merge them vertically by rows, and download the merged result as CSV or XLSX.

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

- Upload one or multiple DAT files.
- Use the second DAT row as source column names and rows 5 onward as observations.
- Keep and rename the required sensor columns to `TIMESTAMP`, `U1`, `V1`, `W1`, `Temp1`, `U2`, `V2`, `W2`, `Temp2`, `U3`, `V3`, `W3`, and `Temp3`.
- Remove unneeded columns such as `RECORD`, `A_SensorStatus`, `B_SensorStatus`, and `C_SensorStatus` from the cleaned output.
- Create XLSX workbooks with `Cleaned_Data` and `Raw_Data` worksheets.
- Download one processed XLSX file or a ZIP containing all processed files.

### Merge CSV Files

- Upload two or more CSV files.
- Merge files vertically by rows.
- Choose between:
  - **Require identical columns**: all files must have the same column names and column order.
  - **Keep all columns**: creates the union of all columns and preserves missing values as blank/NaN.
- Optionally add a `_source_file` column for traceability.
- Preview the merged result.
- Download the merged output as CSV or XLSX.
