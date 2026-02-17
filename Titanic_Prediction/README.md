# Titanic Survival Prediction

This project explores the famous [Titanic dataset](https://www.kaggle.com/c/titanic), where the goal is to predict passenger survival using different machine learning models.

## Dataset
The dataset contains information about Titanic passengers, including:
- Demographics (age, sex, family aboard)
- Ticket details (class, fare, cabin)
- Embarkation port
- Survival outcome

The target variable is **Survived** (1 = survived, 0 = did not survive).

## Approach
I trained and evaluated three different models to compare performance:

1. **Logistic Regression**  
   - A baseline regression model for classification.
   - Interpretable coefficients help identify feature importance.

2. **XGBoost**  
   - Gradient boosting model optimized for speed and performance.
   - Handles non-linear relationships well and captures interactions between features.

3. **LightGBM**  
   - Gradient boosting framework similar to XGBoost but optimized for efficiency.
   - Performs especially well on larger datasets.

## Workflow
1. Data cleaning and preprocessing
   - Handling missing values  
   - Encoding categorical features  
   - Scaling/normalizing features where appropriate  

2. Model training
   - Logistic Regression (sklearn)  
   - XGBoost (xgboost library)  
   - LightGBM (lightgbm library)  

3. Model evaluation
   - Accuracy, precision, recall, F1-score  
   - Cross-validation to avoid overfitting  

## Results
- Logistic Regression: Baseline performance, interpretable results.  
- XGBoost: Stronger predictive performance compared to logistic regression.  
- LightGBM: Comparable or slightly better than XGBoost, with faster training times.  

## Requirements
- Python 3.8+  
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - matplotlib / seaborn (for visualization)

Install dependencies:
```bash
pip install -r requirements.txt