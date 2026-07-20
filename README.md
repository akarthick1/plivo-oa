# End-of-Turn (EOT) Detection (CE23B080 - Plivo Assessment)

## Overview

This project builds a lightweight machine learning model from scratch to predict the probability that a speaker has finished talking (End-of-Turn Detection). The objective is to reduce the response latency of a Voice AI assistant while keeping the interruption rate below an acceptable threshold.

Unlike traditional Voice Activity Detection (VAD), which relies primarily on detecting silence, our approach analyzes causal acoustic features from the speech immediately preceding a pause. This enables the system to identify natural sentence endings significantly earlier, resulting in faster and more natural conversations.

---

## How the Model Works

Traditional Voice Activity Detection systems wait for a long period of silence (typically around **1.6 seconds**) before assuming that a speaker has finished talking. This conservative approach prevents interruptions but introduces noticeable response latency.

Our approach instead analyzes the **last 1.5 seconds of speech before a pause** and extracts causal audio features such as:

- Pitch trajectory
- Energy contour
- Pitch slope
- Energy slope
- Speaking dynamics before silence

Human speech naturally exhibits a gradual decrease in pitch and energy near the end of an utterance. By learning these patterns, the model predicts whether the current pause marks the end of a conversational turn instead of simply measuring silence duration.

The extracted features are used to train a **Random Forest Classifier** on both **English** and **Hindi** speech datasets, enabling the model to learn language-independent end-of-turn characteristics rather than language-specific patterns.

---

## Results

The primary objective was to minimize response delay while maintaining an interruption rate below **5%**.

| Metric | Baseline | Proposed Model |
|--------|---------:|---------------:|
| Response Delay | **1600 ms** | **521 ms** |
| Improvement | — | **≈67% reduction** |

Our model reduces Voice AI response latency by approximately **1 second** compared to a traditional silence-based approach while maintaining a low interruption rate.

---

## Project Structure

```
.
├── train.py              # Feature extraction and model training
├── predict.py            # Generate predictions using the trained model
├── score.py              # Evaluate latency and interruption rate
├── features.py           # Audio feature extraction utilities
├── requirements.txt
├── data/
│   ├── english/
│   └── hindi/
├── outputs/
└── README.md
```

---
## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Train the model

Train using both English and Hindi datasets.

```bash
python train.py --data_dirs data/english data/hindi
```

---

### 2. Generate predictions

Generate predictions for the English dataset.

```bash
python predict.py \
    --data_dir data/english \
    --out outputs/english_predictions.csv
```

Generate predictions for the Hindi dataset.

```bash
python predict.py \
    --data_dir data/hindi \
    --out outputs/hindi_predictions.csv
```

---

### 3. Evaluate the model

Evaluate on the English dataset.

```bash
python score.py \
    --data_dir data/english \
    --pred outputs/english_predictions.csv
```

Evaluate on the Hindi dataset.

```bash
python score.py \
    --data_dir data/hindi \
    --pred outputs/hindi_predictions.csv
```

---


Model Used: Random Forest Classifier

## Future Improvements

Potential enhancements include:

- Gradient Boosting (XGBoost / LightGBM)
- Temporal sequence models (LSTM or Transformer)
- Additional prosodic and spectral features
- Speaker adaptation
- Online real-time inference optimization

---
