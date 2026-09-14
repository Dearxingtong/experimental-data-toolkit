# Experimental Data Toolkit

Experimental Data Toolkit is a lightweight Streamlit web application for routine experimental data processing.

## What It Does

The MVP includes three tools:

- **DAT → XLSX**: Process one experimental `.dat` file at a time, assign Case No, optionally split into user-defined measurement intervals for physical Positions P1-P6, and download XLSX and CSV outputs.
- **Merge CSV Files**: Upload CSV files, choose standard vertical merge or replicate mean/standard-deviation processing, and download CSV/XLSX/ZIP outputs.
- **Data Analysis**: Choose between 3D vector plotting and vertical profile plotting workflows for experimental room data.

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
- Enter required Case No. Case Date is retained for the Full Dataset output.
- Generate the complete Full Dataset for every processed case.
- Choose Split Yes or No for each file.
- If Split is Yes, choose the Number of Splits. Each split row independently selects one physical Position from P1-P6, a Date represented in the file, a Start Time for that date, and a Duration from 0 to 45 minutes.
- Duration `0 min` is treated as unconfigured and must be changed to a value greater than zero before processing.
- Split intervals use full datetime half-open slicing: `start_datetime <= TIMESTAMP < start_datetime + duration`.
- Repeated measurements of the same physical Position are allowed. Only duplicate `Position + Start Datetime` combinations are rejected.
- Split output filenames use the selected split Date. If the same Position appears more than once on the same date, the split start time is appended to prevent overwriting.
- Position workbooks add derived velocity columns in `Cleaned_Data` only: P01-P03 use `Vx = W`, `Vy = -U`, `Vz = -V`; P04-P06 use `Vx = -W`, `Vy = U`, `Vz = -V`.
- The Full Dataset XLSX remains untransformed, and every XLSX contains `Cleaned_Data` and `Raw_Data`.
- CSV outputs contain `Cleaned_Data` only; no Raw CSV files are created.
- Output filenames use the Case No and output date, such as `C03_All_26.08.26.xlsx`, `C03_All_26.08.26.csv`, `C03_P01_26.09.10.xlsx`, or `C03_P01_26.09.10_1209.xlsx`.
- Download the current case outputs immediately, process the next file, or finish the batch.
- Download all generated XLSX and CSV outputs for the current case or final batch as one ZIP.

### Merge CSV Files

- Choose a processing mode:
  - **Standard Merge** keeps the existing vertical CSV merge workflow.
  - **Replicate Mean & SD** combines repeated measurement files into row-wise means and sample standard deviations.

#### Standard Merge

- Upload two or more CSV files.
- Merge files vertically by rows.
- Choose a column handling mode:
  - **Require identical columns**: all files must have the same column names and column order.
  - **Keep all columns**: creates the union of all columns and preserves missing values as blank/NaN.
- Optionally add a `_source_file` column for traceability.
- Preview the merged result.
- Download the merged output as CSV or XLSX.

#### Replicate Mean & SD

- Upload three or more replicate CSV files for repeated measurements.
- Default alignment is **Measurement sequence**, which combines rows by row index rather than exact timestamp.
- Exact `TIMESTAMP` alignment remains available as an advanced option.
- When measurement-sequence alignment is used, the output `TIMESTAMP` comes from Replicate 1.
- If replicate files have different row counts, processing uses the common minimum row count and reports trailing rows excluded from each longer file.
- Numeric output columns preserve the original variable names for means, such as `U1`, and place the matching sample standard deviation column immediately after it, such as `U1_std`.
- `TIMESTAMP_std` is not created. Non-numeric metadata columns are copied from Replicate 1 without `_std` columns.
- Sample standard deviation uses `ddof=1`.
- Missing important numeric columns are rejected with a clear error.
- Download averaged outputs as CSV/XLSX, summary outputs as CSV/XLSX, or one combined ZIP.

### Data Analysis

The Data Analysis tab starts with a simple method selector:

- **3D Vector Plot** for combined 3D airflow vector maps.
- **Vertical Profile Plot** for vertical distributions of air velocity, temperature, and contaminant concentration.

The top of the Data Analysis tab also includes project controls:

- **Save Project** downloads a local `.edtproj` file containing processed analysis state.
- **Load Existing Project** restores processed 3D vector and vertical profile analysis data without re-uploading original measurement files.
- **Start New Analysis** clears only Data Analysis project state, leaving DAT → XLSX and Merge CSV state alone.

Project files are ZIP-based archives with `project.json`, `metadata.json`, and processed CSV tables when available. Version 1 stores processed data and settings only; original uploaded CSV/DAT files are intentionally not embedded.

#### 3D Vector Plot

- Add one airflow position CSV file at a time.
- Select Position 1 through 6 and enter three sensor heights in feet.
- Required CSV columns are `Vx1`, `Vy1`, `Vz1`, `Vx2`, `Vy2`, `Vz2`, `Vx3`, `Vy3`, and `Vz3`.
- Treat uploaded velocity values as SI source data in `m/s`.
- Compute mean airflow vectors and magnitudes for the three heights at each position by averaging components first, then calculating magnitude.
- Accumulate multiple processed position files in the current session.
- Generate equivalent SI and Imperial 3D airflow vector plots for the same physical room.
- Plot arrows with relative room-scale display length so small air speeds remain visible without using raw velocity as geometric displacement.
- Use a continuous `turbo` velocity-magnitude colormap, unit-aware colorbars, orthographic projection, black markers at actual arrow origins, and matching view angles in both plots.
- Use a native Streamlit Plotly preview for exploration plus Azimuth/Elevation/Roll controls for the publication and export camera view.
- Show SI and Imperial summary tables with coordinates, mean velocity components, and magnitude in the correct units.
- Download SI and Imperial EPS/PNG figures, CSV/XLSX summaries, or one combined ZIP.

#### Vertical Profile Plot

- Add one experimental position dataset at a time and accumulate processed datasets in the current session.
- Select Dataset Type, Profile Type, variables, room height, position number, and three measurement heights.
- Upload separate source CSV files for velocity, temperature, and contaminant concentration when those measurements are stored separately.
- Reuse the Velocity CSV for temperature or contaminant concentration when those variables are stored in the same file.
- Experimental velocity CSV files can use `Vx1`, `Vy1`, `Vz1`, `Vx2`, `Vy2`, `Vz2`, `Vx3`, `Vy3`, and `Vz3` for air velocity.
- Air velocity profiles compute instantaneous speed for each row first, then average speed at each height.
- Temperature uses `Temp1`, `Temp2`, and `Temp3` when available, or manually selected numeric columns.
- Contaminant concentration uses manually selected numeric columns plus a contaminant name and unit.
- Source files do not need timestamp alignment because the vertical profile workflow uses per-height summary statistics.
- Optionally upload replicate files per variable. Replicate error bars use between-replicate SD of per-file profile means, not within-file time-series SD.
- Averaged replicate files from Merge CSV are also recognized because mean columns keep the original variable names, with adjacent `_std` columns available for reference.
- The summary table combines values by Position and Height, with missing variables left blank.
- Vertical Profile plots use raw physical values only: measured variable means against actual height in feet.
- The app generates equivalent SI and IP publication figures from the same canonical raw records.
- SI plots use height in meters, air velocity in m/s, temperature in °C, and contaminant concentration in the user-selected unit.
- IP plots use height in feet, air velocity in fpm, temperature in °F, and the same contaminant concentration unit.
- Fixed comparable profile ranges are used for velocity and temperature across cases.
- CFD and Experimental vs CFD input support is reserved for a future update.
- Download EPS and 300 dpi PNG figures, CSV/XLSX summary tables, or one combined ZIP.

## Run Tests

The split logic and regression checks can be run with:

```bash
python tests/test_time_splitter.py
```
