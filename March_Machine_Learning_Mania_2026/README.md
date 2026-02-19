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
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── LICENSE             # License information
```

---

## 📊 Dataset

The competition provides several datasets including:

- Historical regular season and tournament game results
- Team box scores and statistics
- Tournament seeds and brackets
- Conference information
- Coach data
- Geographic data

---

## 🔧 Approach

The typical workflow involves:

1. **Data Exploration and Cleaning**
   - Understanding the data structure and relationships
   - Handling missing values
   - Merging multiple data sources

2. **Feature Engineering**
   - Team performance metrics (win percentage, strength of schedule)
   - Rating systems (ELO, RPI, etc.)
   - Momentum indicators
   - Historical matchup data
   - Tournament seed information

3. **Model Development**
   - Baseline models for comparison
   - Advanced models (Gradient Boosting, Neural Networks)
   - Ensemble methods
   - Cross-validation strategies

4. **Evaluation**
   - Log loss metric (competition metric)
   - Calibration analysis
   - Bracket simulation

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
4. Run the notebooks in the `src/` directory

---

## 📈 Results

Results and model performance will be documented here as the project progresses.

---

## 📝 Notes

- The competition follows the NCAA tournament structure with single-elimination games
- Historical data patterns may not always predict tournament outcomes (upsets are common!)
- The evaluation metric is log loss, which rewards well-calibrated probability predictions

---

## 🔗 Resources

- [Kaggle Competition Page](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
- [NCAA Basketball Statistics](https://www.ncaa.com/stats/basketball-men/d1)
- [KenPom Ratings](https://kenpom.com/)

---

## 📄 License

Please refer to the LICENSE file in this directory for licensing information.
