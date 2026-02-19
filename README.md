# Kaggle Competitions Repository

This repository contains multiple Kaggle competition projects, each in its own folder with complete code, data, and documentation.

## Projects

### 1. [Titanic Survival Prediction](./Titanic_Prediction/)
Exploratory machine learning project using the Titanic dataset from Kaggle. Focuses on data cleaning, feature engineering, and classification modeling to predict passenger survival.

**Models used:**
- Logistic Regression
- XGBoost
- LightGBM

**Key files:**
- `Titanic_Prediction/src/titanic_predictions.ipynb` - Main analysis notebook
- `Titanic_Prediction/src/best_model.ipynb` - Best model implementation
- `Titanic_Prediction/Data/` - Training and test datasets

### 2. [House Price Prediction](./House_Price/)
Exploratory machine learning project using the House Prices dataset (Ames, Iowa) from Kaggle. Includes feature engineering, model experimentation, and insights into regression techniques.

**Models used:**
- CatBoost with extensive feature engineering

**Key files:**
- `House_Price/src/house_price_pred.ipynb` - Main analysis notebook
- `House_Price/data/` - Training and test datasets

### 3. [Wine Quality Prediction](./Wine_Quality/)
Ordinal regression project using wine quality dataset from Kaggle. Predicts wine quality ratings based on physicochemical properties with optimized threshold classification.

**Models used:**
- XGBoost with threshold optimization

**Key files:**
- `Wine_Quality/src/Wine_quality_pred.ipynb` - Main analysis notebook
- `Wine_Quality/src/exploration.ipynb` - Exploratory data analysis
- `Wine_Quality/data/` - Training and test datasets

## Repository Structure

```
Kaggle_competitions/
├── Titanic_Prediction/
│   ├── Data/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
├── House_Price/
│   ├── data/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
├── Wine_Quality/
│   ├── data/
│   ├── src/
│   └── README.md
├── README.md
└── .gitignore
```

## Getting Started

Each project has its own `requirements.txt` file. To get started with a specific project:

1. Navigate to the project folder
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Refer to the project-specific README for more details

## License

Each project may have its own license. Please refer to the LICENSE file in each project directory.