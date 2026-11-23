# Flood Watch

Flood Watch is a research prototype for exploring rainfall forecasts, river conditions, reported flood events, and satellite-based flood mapping across India. It combines a Streamlit dashboard, geospatial and rainfall analysis helpers, Jupyter workflows, and a U-Net model workflow built around the Sen1Floods11 flood-mapping dataset.

> **Safety:** This is a research prototype, not an official warning service. Rainfall values are model-derived estimates, not rain-gauge readings. Do not use this project to make emergency decisions.

## What it does

The Streamlit app has five main views:

- **Overview:** national rainfall exposure and research summaries.
- **Monitor:** recorded rainfall forecasts, river-flow observations, and nearby GDACS flood reports.
- **State Report:** rainfall history, forecast context, river conditions, neighbouring states, and a downloadable PDF report.
- **Mapping:** analyse Sentinel-1 GeoTIFF imagery with the flood segmentation model, compare before/after scenes, and view sample chips when the required files are present.
- **Alerts:** build a rule-based advisory outbox from the recorded data. It is a review workflow; it does not send messages.

Rainfall categories follow the thresholds used in the app: heavy at 64.5 mm/day, very heavy at 115.6 mm/day, and extremely heavy at 204.5 mm/day. River flow is flagged relative to the monitored point's recent history. State monitoring currently focuses on Assam, Kerala, Bihar, Maharashtra, and West Bengal.

## How the project fits together

- `streamlit_app.py` defines the pages, navigation, and display components.
- `app_core.py` contains the data-loading and analysis functions, mapping and chart helpers, PDF report generation, satellite inference helpers, and the rule-based advisory queue.
- `01_flood_model_training.ipynb` trains and evaluates flood segmentation models. The experiments include thresholding baselines and U-Net variants, including a leave-India-out evaluation.
- `02_record_fixtures.ipynb` records rainfall forecasts, river data, and flood reports into local fixture files. This needs internet access when run; the Streamlit app reads the saved fixtures so the demo can run offline.
- `03_app_in_notebook.ipynb` demonstrates mapping, uploaded-scene analysis, monitoring, state reports, and the advisory queue inside Jupyter.
- `05_streamlit_launcher (1).ipynb` documents the notebook-based Streamlit launcher workflow.

Notebook cell outputs have been cleared for publication. The notebooks contain workflow code and explanations; they do not include the large source datasets or trained weights.

## Data and model files

The repository intentionally contains the lightweight source code and notebooks. The original archive also included a Python virtual environment, raw geospatial data, model weights, experiment checkpoints, and generated results. Those are excluded from GitHub to keep the repository usable and avoid oversized files; some checkpoints exceed GitHub's 100 MB per-file limit.

To run the full project, provide the assets separately:

- **Sen1Floods11 dataset** under `Data/sen1floods11_v1.1/` for training and sample-chip mapping.
- **Natural Earth state/province boundaries** under `Data/natural_earth/` for the state maps and reports.
- **A trained model checkpoint** at the location expected by `app_core.py` / the inference helper.
- **Recorded fixtures** under `work/fixtures/` for the offline monitor, state reports, and advisory workflow.
- **Experiment outputs** under `work/results/` if you want the research-results views and comparison charts.

The source supports path overrides through `FLOOD_BASE`, `FLOOD_ROOT` (dataset), and `FLOOD_WORK` (fixtures and outputs). `FLOOD_BASE` defaults to the repository folder.

## Setup

Use a Python environment compatible with the geospatial libraries, then install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the dashboard from the repository root:

```bash
streamlit run streamlit_app.py
```

The dashboard will start, but pages that depend on omitted data, fixtures, experiment outputs, or model weights will need those files in place. Run `02_record_fixtures.ipynb` while online to create the live-data fixture files. Run `01_flood_model_training.ipynb` with the dataset available to train or evaluate a model.

## Research notes

The notebook workflow compares Otsu thresholding baselines with U-Net experiments trained on Sentinel-1 imagery and flood labels. The app's research summaries read metrics and prediction artifacts from `work/results/`; the artifacts are not checked into this repository. Data sources credited by the project include Sen1Floods11 / Cloud to Street, Open-Meteo (including GloFAS / Copernicus river data), GDACS, and Natural Earth. Check each provider's terms and attribution requirements when obtaining data.

## Limitations

- Results depend on the quality, date, coverage, and configuration of the separately supplied data and checkpoints.
- Recorded forecast fixtures are snapshots; they are not a continuously refreshed feed.
- Satellite results depend on compatible Sentinel-1 GeoTIFF inputs and the appropriate trained model.
- The rule-based advisory queue is for demonstration and review. Nothing is sent automatically.
- This code has not been validated as an operational flood-warning system.
