# March Machine Learning Mania 2026

**Competition**: March Machine Learning Mania 2026 (Kaggle)  
[Competition Page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)  

This repository contains my solution for the March Machine Learning Mania 2026 competition, which challenges participants to predict the outcomes of the NCAA Basketball Tournament.

---

## 🏀 Problem Statement

The goal is to predict the outcomes of college basketball games in the 2026 NCAA Tournament (March Madness). This is a **binary classification** problem where we predict the probability that Team A beats Team B for each possible matchup.

Key challenges include:

- Handling historical game data spanning multiple seasons
- Engineering features from team statistics, rankings, and performance metrics
- Modeling tournament dynamics and momentum
- Dealing with the unpredictability of single-elimination tournaments
- Balancing model complexity with generalization

---

## 📁 Repository Structure

```
March_Machine_Learning_Mania_2026/
├── data/               # Competition datasets (teams, games, seeds, etc.)
├── src/                # Source code and notebooks
│   ├── model.ipynb         # Baseline model: XGBoost + LR blended ensemble
│   ├── model_advanced.ipynb# Advanced stacked ensemble (LightGBM + XGBoost + LR)
│   ├── exploration.ipynb   # Exploratory data analysis
│   ├── pipeline.py         # Reusable end-to-end pipeline module
│   ├── features.py         # Feature engineering utilities
│   └── clustering.py       # Grouped PCA and KMeans team clustering
├── submissions/        # Generated submission CSV files
├── prev_year_winner/   # Reference notebook from a prior year's winning approach
├── README.md           # This file
└── .gitattributes
```

---

## 📊 Dataset

The competition provides datasets covering both Men's (M) and Women's (W) tournaments:

- Regular season and tournament compact & detailed results
- Tournament seeds, brackets, and slot assignments
- Massey Ordinal ranking systems (multiple rating systems per season)
- Team conference affiliations and conference tournament results
- Coach data (tenures and historical records)
- Game cities (for neutral-site detection)
- Sample submission files for Stage 1 and Stage 2

---

## 🔧 Approach

### Feature Engineering (`features.py`, `clustering.py`)

- **Season statistics**: per-game averages for all box-score categories (points, FG%, 3P%, FT%, assists, turnovers, rebounds, blocks, steals, fouls) for both team and opponent
- **Derived metrics**: offensive/defensive rating, effective field goal %, turnover rate, true shooting %, possessions, strength of schedule (SOS)
- **ELO ratings**: custom ELO system built from all historical games, carried across seasons with decay
- **Massey Ordinals**: individual ranks from POM, SAG, MOR, COL, DOL, AP systems plus consensus mean/median/std
- **Tournament seed**: numerical seed extracted per team per season
- **Pythagorean win expectation** (advanced model): efficiency-based expected win rate
- **Coach features** (advanced model): tenure, career tournament win rate
- **Conference features** (advanced model): Power 6 membership, conference tournament wins/losses before NCAA tournament
- **Neutral-site record** (advanced model): win rate at neutral venues using game city data
- **Grouped PCA** (`clustering.py`): dimensionality reduction applied separately to Offense, Defense, Rebounding, and Overall feature groups
- **KMeans team clustering** (`clustering.py`): playstyle cluster labels derived from PCA components

### Models

#### Baseline (`model.ipynb`)
- **XGBoost** classifier + **Logistic Regression**, blended by weighted average
- Platt-scaling calibration on a held-out calibration season
- Walk-forward cross-validation (leak-free, PCA/KMeans refit each fold)
- Tournament games up-weighted 2× vs regular season games

#### Advanced (`model_advanced.ipynb`)
- **LightGBM** + **XGBoost** + **Logistic Regression** base learners
- Stacked meta-learner (Logistic Regression on out-of-fold predictions)
- Isotonic regression calibration for well-calibrated probability outputs
- Separate Men's and Women's models
- Optuna hyperparameter tuning (optional, ~15–25 min)

### Evaluation

- **Primary metric**: Log loss (competition metric)
- **Secondary metric**: Brier score (used for Optuna tuning in the advanced model)
- Walk-forward cross-validation across multiple seasons

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository and navigate to this project folder
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the competition data from Kaggle and place it in the `data/` directory
4. Run the notebooks in the `src/` directory:
   - Start with `exploration.ipynb` for EDA
   - Run `model.ipynb` for the baseline submission
   - Run `model_advanced.ipynb` for the full stacked ensemble

---

## 📈 Results

| Model | Submission File | Notes |
|---|---|---|
| Baseline (XGBoost + LR) | `submissions/submission_2026.csv` | Blended ensemble, Platt calibration |
| Advanced (LightGBM + XGBoost + LR) | `submissions/submission_advanced_2026.csv` | Stacked meta-learner, isotonic calibration |

---

## 📝 Notes

- Both Men's and Women's tournament matchups are included in the submission
- The competition follows the NCAA single-elimination tournament structure — upsets are common
- The evaluation metric is log loss, which heavily rewards well-calibrated probabilities
- Historical data patterns from the `prev_year_winner/` folder informed several feature ideas

---

## 🔗 Resources

- [Kaggle Competition Page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
- [NCAA Basketball Statistics](https://www.ncaa.com/stats/basketball-men/d1)
- [KenPom Ratings](https://kenpom.com/)

---

## 📄 License

Please refer to the LICENSE file in this directory for licensing information.
