# NeuroSeize-Deep Learning Based EEG Signal Processing For Seizure Detection

Deep-learning pipeline for detecting epileptic seizures from scalp EEG recordings using the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/). Multi-channel EEG signals are sliced into short overlapping windows, converted into image-like tensors, and classified as **seizure** / **non-seizure** by a convolutional neural network built in PyTorch.

## Overview

- **Dataset:** CHB-MIT Scalp EEG Database (pediatric patients, `.edf` recordings with seizure annotations)
- **Channels used:** 18 standard bipolar montage channels (`FP1-F7`, `F7-T7`, ..., `FZ-CZ`, `CZ-PZ`)
- **Windowing:** 8-second windows with a 4-second step (50% overlap)
- **Model:** Custom 2D CNN (`ConvNet`) operating on `(channel, time)` windows, trained with binary cross-entropy
- **Evaluation:** Classification report, ROC curve, AUC score, and a moving-average peak-detection scheme to flag seizure onset points on held-out recordings

## Pipeline

1. **Patient split** — Patients are split 80/20 into train/test sets (`random.seed(2023)`), so evaluation is done on unseen patients rather than unseen windows.
2. **Signal extraction** — Each `.edf` file is read with `mne`, channels are matched/renamed to the 18-channel bipolar montage, and seizure annotations are read with `wfdb` to build a per-sample binary seizure mask.
3. **Windowing & sampling** — Recordings are cut into 8s windows (4s stride). Since seizure segments are rare, non-seizure windows are subsampled (~1%) while all seizure windows are kept, to reduce class imbalance.
4. **Dataset construction** — Windows are stacked into a `(N, channels, time)` array and split into train/validation sets with `sklearn.model_selection.train_test_split` (stratified on seizure presence).
5. **Model training** — A `ConvNet` (6 conv layers + max-pooling + global average pooling + fully-connected head with dropout, sigmoid output) is trained with `Adam` and `BCELoss`, tracking train/validation loss with early stopping.
6. **Testing** — The trained model is run on held-out test recordings, producing a classification report, ROC curve, and AUC score.
7. **Seizure-onset detection** — Model predictions over a full recording are smoothed with a moving average, and peaks above a probability threshold are treated as detected seizure events. Detected windows are plotted against the raw EEG traces for visual inspection.

## Model Architecture

```
Conv2d(1→64) → ReLU → Conv2d(64→64, stride) → ReLU → MaxPool
→ Conv2d(64→128) → ReLU → Conv2d(128→128, stride) → ReLU → MaxPool
→ Conv2d(128→256) → ReLU → Conv2d(256→256, stride) → ReLU → MaxPool
→ Global Average Pool
→ FC(256→256) → Dropout → FC(256→128) → FC(128→64) → Dropout → FC(64→1) → Sigmoid
```

## Repository Structure

```
.
├── siezure-detection-eeg.ipynb   # Main notebook: data loading, training, evaluation
└── README.md
```

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`
- `scipy`
- `mne`
- `wfdb`
- `scikit-learn`
- `torch`
- `tqdm`

```bash
pip install numpy matplotlib scipy mne wfdb scikit-learn torch tqdm
```

## Data Setup

1. Download the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (`.edf` files with corresponding `.seizures` annotation files).
2. Update the `path2pt` variable in the notebook to point to your local copy of the dataset.
3. (Optional) If you have pre-extracted `signal_samples.npy` / `is_sz.npy` arrays (e.g. from a prior run), place them where the notebook expects them to skip the raw-signal extraction step.

## Usage

Open `siezure-detection-eeg.ipynb` and run the cells in order:

1. Set the dataset path and run the patient split + signal extraction cells.
2. Train the model (`num_epochs`, learning rate, and window/step sizes are configurable near the top of the training section).
3. Run the test-set evaluation cells to get the classification report, ROC curve, and AUC.
4. Run the peak-detection cells on a chosen test recording to visualize predicted seizure onsets against the raw EEG.

## Notes / Limitations

- Only recordings whose channel set matches the 18-channel bipolar montage are used; others are skipped.
- Non-seizure windows are heavily subsampled during training to address class imbalance — this affects the interpretation of raw loss values.
- The model is trained and evaluated at the window level; the peak-detection step is a simple post-hoc heuristic (moving average + threshold) for event-level seizure detection, not part of the trained model itself.

## Acknowledgements

- CHB-MIT Scalp EEG Database, PhysioNet: Shoeb, A. (2009), and Goldberger et al. (2000) PhysioNet.
