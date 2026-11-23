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

## Model architecture, training, and evaluation

### Architecture and inputs

The flood segmentation workflow uses a U-Net from `segmentation-models-pytorch` with a ResNet-34 encoder. The input has two Sentinel-1 channels, and the model produces one logit for each pixel. Each input band is clipped to −30 to 0 dB and scaled to the [0, 1] range using `(x + 30) / 30`.

Training labels use `1` for water, `0` for non-water, and `-1` for pixels excluded from loss and metric calculations.

### Loss and activations

- **Primary loss:** masked binary cross-entropy with logits plus soft Dice loss. The ignored (`-1`) pixels are excluded from both terms.
- **BCE-only ablation:** the notebook also evaluates a variant using masked binary cross-entropy alone.
- **Output activation:** the network returns raw logits during training. `BCEWithLogitsLoss` combines the sigmoid operation and binary cross-entropy in a numerically stable way. Inference applies sigmoid to logits to obtain per-pixel water probabilities.
- **Decision threshold:** the notebook selects a probability threshold on the validation split from 0.30 to 0.70 in increments of 0.05, choosing the threshold with the best validation IoU.

### Training configuration

The notebook's standard configuration uses AdamW with learning rate `1e-4`, weight decay `1e-4`, batch size `16`, 256 × 256 crops, and up to 60 epochs. It samples four crops per image chip and uses cosine annealing for the learning rate, with early stopping patience of 15 epochs. A quick-run configuration reduces the model to ResNet-18, uses batch size 4 and 128 × 128 crops, and runs for two epochs with patience 2; it is intended for workflow checks, not comparable final results.

### Evaluation metrics

Metrics are calculated over valid pixels by summing true positives (TP), false positives (FP), and false negatives (FN) across chips, then computing:

| Metric | Definition | What it measures |
| --- | --- | --- |
| **Precision** | `TP / (TP + FP)` | Of pixels predicted as water, the fraction that are water. |
| **Recall** | `TP / (TP + FN)` | Of actual water pixels, the fraction detected. |
| **F1 score** | `2 × precision × recall / (precision + recall)` | Harmonic mean of precision and recall. |
| **IoU (Jaccard)** | `TP / (TP + FP + FN)` | Overlap between predicted and reference water regions. |

## Experimental results

The following scores are taken from the project evaluation outputs supplied for this update. Values are shown as mean ± standard deviation across three seeds when available. The two Otsu baselines each use one run, so no standard deviation is available. Scores are computed over valid pixels; `india_all` is the held-out India evaluation for the leave-India-out experiment.

| Experiment | Evaluation set | Seeds | IoU | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| e0_otsu_per_chip | test | 1 | 0.242 | 0.390 | 0.262 | 0.766 |
| e0_otsu_per_chip | bolivia | 1 | 0.429 | 0.601 | 0.470 | 0.832 |
| e0_otsu_per_chip | test_india | 1 | 0.390 | 0.561 | 0.414 | 0.873 |
| e0_otsu_provided | test | 1 | 0.546 | 0.706 | 0.681 | 0.733 |
| e0_otsu_provided | bolivia | 1 | 0.697 | 0.822 | 0.781 | 0.867 |
| e0_otsu_provided | test_india | 1 | 0.704 | 0.827 | 0.880 | 0.779 |
| e1_baseline | test | 3 | 0.665 ± 0.018 | 0.799 ± 0.013 | 0.805 ± 0.004 | 0.793 ± 0.028 |
| e1_baseline | bolivia | 3 | 0.595 ± 0.051 | 0.745 ± 0.040 | 0.767 ± 0.018 | 0.725 ± 0.060 |
| e1_baseline | test_india | 3 | 0.692 ± 0.036 | 0.817 ± 0.025 | 0.816 ± 0.014 | 0.820 ± 0.046 |
| e2a_bce_only | test | 3 | 0.648 ± 0.012 | 0.786 ± 0.009 | 0.818 ± 0.002 | 0.758 ± 0.015 |
| e2a_bce_only | bolivia | 3 | 0.608 ± 0.031 | 0.756 ± 0.024 | 0.780 ± 0.016 | 0.734 ± 0.038 |
| e2a_bce_only | test_india | 3 | 0.657 ± 0.011 | 0.793 ± 0.008 | 0.827 ± 0.010 | 0.762 ± 0.014 |
| e3_leave_india_out | india_all | 3 | 0.490 ± 0.011 | 0.658 ± 0.010 | 0.735 ± 0.026 | 0.596 ± 0.020 |
| e3_leave_india_out | test | 3 | 0.649 ± 0.004 | 0.787 ± 0.003 | 0.822 ± 0.008 | 0.755 ± 0.012 |
| e3_leave_india_out | bolivia | 3 | 0.584 ± 0.032 | 0.737 ± 0.026 | 0.775 ± 0.005 | 0.703 ± 0.043 |
| e3_leave_india_out | test_india | 3 | 0.645 ± 0.005 | 0.784 ± 0.004 | 0.827 ± 0.006 | 0.745 ± 0.010 |
| e5_final | test | 3 | 0.666 ± 0.004 | 0.799 ± 0.003 | 0.806 ± 0.006 | 0.793 ± 0.010 |
| e5_final | bolivia | 3 | 0.625 ± 0.022 | 0.769 ± 0.016 | 0.782 ± 0.003 | 0.757 ± 0.030 |
| e5_final | test_india | 3 | 0.673 ± 0.016 | 0.804 ± 0.012 | 0.827 ± 0.009 | 0.783 ± 0.019 |

For the final three-seed experiment, the highest mean IoU is 0.673 ± 0.016 on `test_india`; on the `test` split, mean IoU is 0.666 ± 0.004 and mean F1 is 0.799 ± 0.003. The BCE-only experiment scores slightly lower test IoU (0.648 ± 0.012), while the three-seed baseline is 0.665 ± 0.018. The leave-India-out model obtains 0.490 ± 0.011 IoU on `india_all`, which reflects the more challenging held-out geography.

The training plots supplied with these results show training loss decreasing over epochs while validation water IoU fluctuates around its best range. Validation IoU is used for checkpoint selection and early stopping; the test-set metrics above are the reported final evaluation.

## Running the app

Install dependencies from `requirements.txt`, provide the data and model assets listed above, then start the dashboard with:

```bash
streamlit run streamlit_app.py
```

Refer to the notebooks for dataset preparation, fixture recording, model training, and evaluation workflows.
