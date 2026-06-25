# Stay Sharpe — Quantitative Trading Algorithm

A machine-learning-based algorithmic trading strategy built for the **Infinium Ctrl-Alpha "Stay Sharpe"** competition. The system trains an XGBoost classifier on historical OHLCV price data using Triple Barrier labeling and generates low-latency buy/sell/hold signals in a real-time streaming environment.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600?logo=xgboost)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/Preprocessing-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Approach Overview](#approach-overview)
- [Feature Engineering](#feature-engineering)
- [Triple Barrier Labeling](#triple-barrier-labeling)
- [Model](#model)
- [Prediction Pipeline](#prediction-pipeline)
- [Key Design Decisions](#key-design-decisions)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Configuration](#configuration)

---

## Problem Statement

The competition provides anonymized OHLCV (open, high, low, close, volume) time series for multiple financial assets. A strategy must implement two methods:

- **`train(train_df)`** — fit a model on historical price data
- **`predict(row, timestamp) → int`** — return a signal `{-1, 0, 1}` for each new incoming bar in real time, where:
  - `1` = **Buy** (go long)
  - `-1` = **Sell** (go short / exit long)
  - `0` = **Hold** (no action)

Performance is judged on **Sharpe ratio** — risk-adjusted returns rather than raw PnL — penalizing volatile strategies even if profitable.

---

## Approach Overview

```
Historical OHLCV data
        ↓
Technical Feature Engineering (SMA, Volatility, Momentum, Dist_SMA)
        ↓
Triple Barrier Labeling (volatility-adjusted profit/stop targets)
        ↓
XGBoost Multi-class Classifier (buy / sell / hold)
        ↓
StandardScaler normalization
        ↓
Real-time predict(): incremental feature computation from rolling window
        ↓
Confidence-thresholded signal output (1 / -1 / 0)
```

---

## Feature Engineering

Four technical indicators are computed over a rolling window of **20 bars**:

| Feature | Formula | Intuition |
|---|---|---|
| **SMA** | 20-period simple moving average of `close` | Trend direction baseline |
| **VOLATILITY** | 20-period rolling std of `pct_change(close)` | Market regime (calm vs. turbulent) |
| **Momentum** | `close[t] - close[t-20]` | Rate and direction of price movement |
| **Dist_SMA** | `close[t] - SMA[t]` | Distance from mean — mean-reversion or breakout signal |

During training, features are computed efficiently via Pandas vectorised operations (`rolling`, `pct_change`, `diff`). During inference, features are recomputed incrementally from a bounded circular buffer of the 60 most recent closing prices, keeping per-bar prediction latency minimal.

---

## Triple Barrier Labeling

Standard fixed-horizon labels (e.g., "did the price go up in 10 bars?") ignore risk asymmetry. The Triple Barrier Method, introduced by Marcos López de Prado (*Advances in Financial Machine Learning*), assigns labels based on which barrier is hit first:

```
Upper barrier:  entry_price × (1 + PROFIT_MULTIPLIER × volatility_t)   → label = +1 (Buy)
Lower barrier:  entry_price × (1 - LOSS_MULTIPLIER  × volatility_t)    → label = -1 (Sell)
Vertical bar:   t + EXECUTION_DELAY (10 bars)                           → label =  0 (Hold)
```

**Barrier parameters:**
- `PROFIT_MULTIPLIER = 1.5` — upper barrier is 50% wider than the lower, skewing the model toward higher-confidence long signals
- `LOSS_MULTIPLIER = 1.0` — stop-loss set at 1× current volatility
- `EXECUTION_DELAY = 10` — look-ahead window of 10 bars

Barriers are **volatility-adaptive**: tight in calm markets (more hold signals), wide in volatile markets (more decisive signals), naturally aligning with the Sharpe-ratio objective.

Labels are mapped from `{-1, 0, 1}` to `{0, 1, 2}` (XGBoost multi-class format) and stored in `label_map` for consistent round-tripping.

---

## Model

**XGBoost Multi-class Classifier** (`multi:softmax`, 3 classes):

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 50 | Enough capacity without overfitting on limited features |
| `max_depth` | 2 | Shallow trees → low variance, fast inference |
| `learning_rate` | 0.1 | Conservative shrinkage for stable convergence |
| `subsample` | 0.7 | Row subsampling reduces overfitting |
| `colsample_bytree` | 0.7 | Feature subsampling adds diversity |
| `n_jobs` | 1 | Single-threaded to minimize prediction latency |
| `random_state` | 42 | Reproducibility |

Features are standardised with `sklearn.StandardScaler` (zero mean, unit variance) before training and inference to ensure gradient boosting operates on a normalised feature space regardless of price scale differences across assets.

---

## Prediction Pipeline

Each call to `predict(row, timestamp)` follows this path:

```
1. Append new close price to rolling history (capped at 60 bars)

2. Guard: return 0 if history < feature_window + 1 (cold-start)

3. Compute features from the last 20 prices:
   - SMA       = mean(close[-20:])
   - VOLATILITY = std(pct_change(close[-21:]))
   - Momentum   = close[-1] - close[-21]
   - Dist_SMA   = close[-1] - SMA

4. Scale with fitted StandardScaler

5. model.predict_proba() → [p_sell, p_hold, p_buy]

6. Apply confidence threshold (0.55):
   - p_buy  > 0.55  → signal = +1
   - p_sell > 0.55  → signal = -1
   - otherwise      → signal =  0
```

The cold-start guard ensures the first `FEATURE_WINDOW` (20) bars emit `0` rather than garbage features, avoiding noise trades at session open.

---

## Key Design Decisions

**Volatility-adaptive barriers** — By tying profit and stop-loss targets to current volatility rather than fixed percentages, the labeling automatically scales with market regime. This produces more balanced and meaningful labels than fixed-threshold approaches.

**Asymmetric multipliers (1.5 vs 1.0)** — The wider upper barrier means the model is rewarded more for correctly anticipating large upward moves, introducing a mild long bias while still allowing short signals. The resulting Sharpe ratio is better than a symmetric strategy on trending asset universes.

**Confidence threshold (0.55)** — Rather than always following the argmax prediction, the model only acts when it is confident (probability > 55%). This converts a noisy high-frequency signal into a selective, high-conviction one, directly improving the Sharpe ratio by reducing unnecessary trades and their associated costs.

**Shallow XGBoost (max_depth=2, 50 trees)** — Deep trees on 4 features would overfit immediately. Shallow trees with moderate shrinkage generalise better across the diverse anonymized asset universe. `n_jobs=1` eliminates thread-spawning overhead in the tight predict loop.

**Bounded history buffer (60 bars)** — The history cap (`feature_window × 3`) prevents unbounded memory growth in long streaming sessions while providing sufficient context for all features (`feature_window + 1` minimum = 21 bars).

---

## Project Structure

```
Stay-Sharpe-/
├── ctrl_alpha.py              — CtrlAlpha class (train + predict)
├── README.md
├── data/
│   └── company_split_p1.csv  — Sample asset OHLCV data with ground-truth labels
└── tests/
    └── verify_class.py        — I/O conformance test (interface validation)
```

---

## Setup & Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost
```

### Verify Interface

```bash
python tests/verify_class.py
```

This runs a 5-step conformance check:
1. Imports `CtrlAlpha` from `ctrl_alpha.py`
2. Instantiates the class
3. Calls `train()` with mock OHLCV data
4. Calls `predict()` on a single row
5. Validates the return type is `int ∈ {-1, 0, 1}`

### Using the Class

```python
import pandas as pd
from ctrl_alpha import CtrlAlpha

# Load your OHLCV data (must have a 'close' column at minimum)
train_df = pd.read_csv("data/company_split_p1.csv")

model = CtrlAlpha()
model.train(train_df)

# Streaming inference — called bar-by-bar
for _, row in live_data.iterrows():
    signal = model.predict(row=row, timestamp=int(row['timestamp']))
    # signal: 1 = buy, -1 = sell, 0 = hold
```

---

## Configuration

All tunable constants are defined at the top of `ctrl_alpha.py`:

| Constant | Default | Description |
|---|---|---|
| `EXECUTION_DELAY` | `10` | Look-ahead horizon for Triple Barrier labeling (bars) |
| `PROFIT_MULTIPLIER` | `1.5` | Width of upper barrier as a multiple of volatility |
| `LOSS_MULTIPLIER` | `1.0` | Width of lower barrier as a multiple of volatility |
| `FEATURE_WINDOW` | `20` | Rolling window length for all technical indicators |
| `CONFIDENCE_THRESHOLD` | `0.55` | Minimum predicted probability to emit a non-zero signal |

Increasing `CONFIDENCE_THRESHOLD` toward 0.65–0.70 reduces trade frequency and often improves Sharpe at the cost of fewer opportunities. Decreasing `max_depth` or `n_estimators` trades accuracy for even lower inference latency.
