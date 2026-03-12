# 🏀 NBA Game Outcome Prediction — End-to-End Data Science Pipeline

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

An end-to-end pipeline for predicting NBA game outcomes before tip-off. The focus is on getting the data work right: clean inputs, no leakage in the features, and a reproducible process from raw CSVs to a trained model.

---

## 🎯 Objective

Predict `home_win` (1 = home team wins, 0 = away team wins) before tip-off, using historical performance features derived exclusively from prior games. All features are built using only data available before the game starts — no peeking at the game being predicted. Each notebook validates its own output before the next one reads it.

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.13 |
| Data manipulation | pandas 2.2, NumPy 2.1 |
| Storage | Apache Parquet via PyArrow |
| Notebooks | Jupyter |
| Database (planned) | PostgreSQL — star schema |
| ML (planned) | scikit-learn, XGBoost / LightGBM |

---

## 🔄 Pipeline Overview

```
Data_raw/  (CSV source files — never modified)
    │
    ▼
┌─────────────────────────────────────────────┐
│  01_data_Cleaning.ipynb                     │
│  Scope: NBA seasons 2017 – 2026             │
├─────────────────────────────────────────────┤
│  → games_clean.parquet          13,151 × 14 │
│  → team_stats_clean.parquet     26,302 × 41 │
│  → player_stats_advanced_clean  53,113 × 25 │
│  → player_stats_scoring_clean   53,113 × 20 │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  02_feature_engineering.ipynb               │
│  One row per game, features prefixed        │
│  home_ / away_, plus differentials          │
├─────────────────────────────────────────────┤
│  → game_features.parquet        13,151 × 60 │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  03_eda.ipynb  [In Progress]                │
│  Distribution analysis, correlation study,  │
│  multicollinearity audit, feature selection │
├─────────────────────────────────────────────┤
│  → feature_selected.parquet     (planned)   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  04_ml_baseline.ipynb  [Planned]            │
│  Logistic Regression, Random Forest,        │
│  XGBoost baselines — cross-validated        │
├─────────────────────────────────────────────┤
│  → trained model + evaluation report        │
└─────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
NBA/
├── Data_raw/                          # Source files (not tracked by git — size)
│   ├── Games.csv                          72,865 rows
│   ├── TeamStatistics.csv                145,731 rows
│   ├── PlayerStatisticsAdvanced.csv      116,899 rows
│   ├── PlayerStatisticsScoring.csv       116,899 rows
│   └── ...                            (additional raw files)
│
├── Data_processed/                    # Pipeline outputs — Parquet format
│   ├── games_clean.parquet
│   ├── team_stats_clean.parquet
│   ├── player_stats_advanced_clean.parquet
│   ├── player_stats_scoring_clean.parquet
│   └── game_features.parquet
│
└── Notebooks/
    ├── 01_data_Cleaning.ipynb         ✅ Complete
    ├── 02_feature_engineering.ipynb   ✅ Complete
    ├── 03_eda.ipynb                   🔄 In Progress
    └── 04_ml_baseline.ipynb           📋 Planned
```

---

## 📓 Completed Notebooks

### 🧹 01 — Data Cleaning

**Goal:** Transform raw CSVs into clean, typed, validated Parquet files ready for feature engineering.

- **Inputs:** `Games.csv`, `TeamStatistics.csv`, `PlayerStatisticsAdvanced.csv`, `PlayerStatisticsScoring.csv`
- **Output:** 4 Parquet files — one per source table

| File | Rows | Cols | Primary Key |
|------|------|------|-------------|
| `games_clean.parquet` | 13,151 | 14 | `game_id` |
| `team_stats_clean.parquet` | 26,302 | 41 | `(game_id, team_id)` |
| `player_stats_advanced_clean.parquet` | 53,113 | 25 | `(game_id, person_id)` |
| `player_stats_scoring_clean.parquet` | 53,113 | 20 | `(game_id, person_id)` |

**Key decisions:**

- **Scope filter:** seasons 2017–2026 only, applied to games first and cascaded to all child tables via FK.
- **`season_end_year`:** derived from `game_id` using the NBA's internal encoding, not from the calendar year.
- **`is_overtime`:** computed from quarter scores in `team_stats`, then propagated to `games`.
- **Nullable integers:** pandas `Int16` / `Int32` for columns with legitimate NaN (e.g. `turnovers`, `attendance`).
- **Outlier handling:** `efg_pct` and `ts_pct` values > 1.5 set to NaN — not clipped.
- **Validation:** PK uniqueness and FK coverage asserted at 100% for all four output tables.

---

### 🔧 02 — Feature Engineering

**Goal:** Build a game-level feature matrix for predicting `home_win`, using only information available before tip-off.

- **Inputs:** `games_clean.parquet` + `team_stats_clean.parquet`
- **Output:** `game_features.parquet` — 13,151 rows × 60 columns
- **Target variable:** `home_win` — 56.1% home wins, 43.9% away wins

#### Anti-Leakage Pattern

Every rolling feature follows the **shift-then-roll** approach:

```python
x.shift(1).rolling(window=10, min_periods=3).mean()
# shift(1)  → current game sees the previous game's value at position 0
# rolling() → looks back over the N shifted positions
# result    → rolling mean of the N games that occurred BEFORE this game
```

Rolling groups are defined as `(team_id, season_end_year)` — resetting at each new season prevents end-of-season form from contaminating the following year's early games.

#### Feature Groups

| Group | # Columns | Features |
|-------|-----------|---------|
| 📅 Schedule context | 6 | `rest_days`, `is_back_to_back`, `games_last_7d` — home and away |
| 🏆 Season context | 6 | `pre_game_wins`, `pre_game_losses`, `season_win_pct` — home and away |
| 📈 Rolling 10-game overall | 20 | win%, pts scored/allowed, pt diff, eFG%, FG3%, FT rate, rebounds, assist%, turnovers |
| 🏠 Rolling location splits | 4 | `split_roll_win_pct`, `split_roll_pt_diff` — same-location games only |
| ⚔️ Head-to-head history | 2 | `h2h_games_prior`, `h2h_home_win_pct` |
| ➕ Differentials | 14 | `diff_*` — home minus away for all key metrics |

> All feature columns appear twice: once prefixed `home_` and once `away_`. Differential columns (`diff_*`) represent `home_X − away_X`, making the advantage direction explicit.

---

## 📊 Key Findings (as of Notebook 02)

| Metric | Value |
|--------|-------|
| 🏠 Home court advantage | 56.1% of games won by the home team |
| 😓 Back-to-back game rate | 38.9% of team-game rows |
| ⚔️ Max H2H games between same pair | 33 (over 9 seasons) |
| 🥇 Strongest single predictor | `diff_split_pt_diff` (r = 0.271) |

**Top 5 feature correlations with `home_win`** (Pearson absolute value):

```
diff_split_pt_diff    0.271   ← location-specific rolling point differential
diff_pt_diff          0.269   ← overall rolling point differential
diff_season_win_pct   0.256   ← season win% gap (limited data — see below)
diff_split_win_pct    0.251   ← location-specific rolling win%
diff_win_pct          0.244   ← overall rolling win%
```

> Interestingly, how a team performs at home specifically (not just overall) turns out to be more predictive than their general rolling form. Comparing teams against each other (home minus away) also matters more than looking at each side in isolation.

---

## 🗺️ Project Roadmap

| # | Notebook | Status | Output |
|---|----------|--------|--------|
| 01 | Data Cleaning | ✅ Complete | 4 validated Parquet files |
| 02 | Feature Engineering | ✅ Complete | `game_features.parquet` (60 features) |
| 03 | Exploratory Data Analysis | 🔄 In Progress | `feature_selected.parquet` |
| 04 | ML Baseline | 📋 Planned | Trained model + evaluation report |

**Notebook 03 — EDA** digs into the distributions, checks correlations between features, and decides what goes into the model and what gets dropped.

**Notebook 04 — ML Baseline** trains a few baseline models (Logistic Regression, Random Forest, XGBoost) with proper time-series splits and compares them.

---

## 🚀 How to Run

> **Note:** Raw data files (`Data_raw/`) are not included in this repository due to file size. The processed Parquet files in `Data_processed/` are committed and can be used directly starting from Notebook 02.

**1. Clone the repository**
```bash
git clone https://github.com/FelipeFQ/NBA-win-prediction.git
cd NBA-win-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy pyarrow jupyter
```

**3. Run notebooks in order**
```bash
jupyter notebook Notebooks/01_data_Cleaning.ipynb
jupyter notebook Notebooks/02_feature_engineering.ipynb
```

Each notebook is self-contained and saves its output to `Data_processed/` before the next one reads it.
