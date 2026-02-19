# Wine Quality Prediction

This project uses the Wine Quality dataset from Kaggle to predict wine quality ratings based on physicochemical properties.

## Dataset
The dataset contains information about wines, including:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol content

The target variable is **quality** (ordinal rating, typically ranging from 3 to 9).

## Approach
This is an ordinal regression problem where the goal is to predict wine quality ratings. The project includes:

1. **XGBoost Regression**  
   - Gradient boosting model with threshold optimization
   - Uses Quadratic Weighted Kappa (QWK) for evaluation
   - Coordinate descent to optimize decision thresholds for ordinal classes

## Workflow
1. Data exploration and preprocessing
   - Handling missing values  
   - Feature scaling and normalization  
   - Exploratory data analysis  

2. Model training
   - XGBoost regressor (xgboost library)  
   - Custom threshold optimization for ordinal classification  

3. Model evaluation
   - Mean Squared Error (MSE)  
   - Quadratic Weighted Kappa (QWK)  
   - Cross-validation for robust performance assessment  

## Results
- XGBoost with optimized thresholds provides strong performance for ordinal quality prediction
- Threshold optimization improves ordinal classification accuracy compared to standard regression

## Key Files
- `src/Wine_quality_pred.ipynb` - Main analysis notebook with model training
- `src/exploration.ipynb` - Exploratory data analysis
- `src/pipeline.py` - Data preprocessing pipeline
- `data/` - Training and test datasets

## Requirements
- Python 3.8+  
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib / seaborn (for visualization)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Competition
This project is based on wine quality prediction tasks commonly found on Kaggle and similar platforms, focusing on predicting wine ratings from chemical properties.
