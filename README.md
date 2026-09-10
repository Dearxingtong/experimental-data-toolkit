# Experimental Data Toolkit

Experimental Data Toolkit is a lightweight Streamlit web application for routine experimental data processing.

## What It Does

The MVP includes three tools:

- **DAT → XLSX**: Process one experimental `.dat` file at a time, assign Case No and Case Date, optionally split into 2 to 6 user-defined Positions, and download XLSX and CSV outputs.
- **Merge CSV Files**: Upload two or more `.csv` files, preview each file, merge them vertically by rows, and download the merged result as CSV or XLSX.
- **Data Analysis**: Choose between a 3D airflow vector plot workflow and a vertical profile plotting workflow for experimental room data.

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
- If Split is Yes, choose 2 through 6 Positions, select a Start Time from the detected experiment range, and select a Duration from 0 to 45 minutes.
- Duration `0 min` is treated as unconfigured and must be changed to a value greater than zero before processing.
- Position intervals use half-open slicing: `start <= TIMESTAMP < start + duration`.
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

### Data Analysis

The Data Analysis tab starts with a simple method selector:

- **3D Vector Plot** for combined 3D airflow vector maps.
- **Vertical Profile Plot** for vertical distributions of air velocity, temperature, and contaminant concentration.

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
- The summary table combines values by Position and Height, with missing variables left blank.
- Raw Profiles plot measured means against height in feet.
- Normalized Profiles plot `Z = z / H`, `U* = U / Us`, `theta = (T - Tin) / (Tout - Tin)`, and `C* = (C - Cin) / (Cout - Cin)`.
- CFD and Experimental vs CFD input support is reserved for a future update.
- Download EPS and 300 dpi PNG figures, CSV/XLSX summary tables, or one combined ZIP.

## Run Tests

The split logic and regression checks can be run with:

```bash
python tests/test_time_splitter.py
```
