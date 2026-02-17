import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import set_config
set_config(transform_output="pandas")

drop_cols = ['PassengerId','Cabin', 'Name', 'Title', 'SibSp', 'Parch']
one_hot_cols = ['Sex', 'Embarked', 'cabin_level']
impute_cols = ['Age']
scaling_cols = ['Age', 'Fare', 'family_size']
title_cats = ['Master', 'Dr', 'Rev', 'Major', 'Col', 'Countess', 'Capt', 'Sir', 'Lady', 'Don', 'Jonkheer']
pass_through_bins = ['disting_title','is_child','is_alone']


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ticket string to number
    df['Ticket'] = df['Ticket'].str.extract(r'(\d+)\s*$')
    df['len_ticket_num'] = df['Ticket'].str.len()
    df['Ticket'] = df['Ticket'].astype(float)

    #  Name string to title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
    df['disting_title'] = df['Title'].isin(title_cats).astype(int)

    df['cabin_level'] = df['Cabin'].str[0]

    df['is_child'] = (df['Age'] < 18).astype(int)
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['is_alone'] = (df['family_size'] ==1).astype(int)

    return df

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    # ('age_impute', age_only_impute, impute_cols), 
    ('num', num_pipeline, scaling_cols),
    ('cat', cat_pipeline, one_hot_cols),
    ('pass', 'passthrough', pass_through_bins)
], remainder='drop', verbose_feature_names_out=False)

pipeline_base = Pipeline([
    ('feature_engineering', FunctionTransformer(feature_engineer, validate=False)),
    ('drop_cols', FunctionTransformer(lambda df: df.drop(columns=drop_cols, errors='ignore'), validate=False)),
    ('preprocessing', preprocessor)
])

def get_fitted_pipeline(X: pd.DataFrame):

    pipeline_base.fit(X)

    encoder = pipeline_base.named_steps['preprocessing']
    feature_names = encoder.get_feature_names_out()

    transformed = pipeline_base.transform(X)
    return pipeline_base, pd.DataFrame(transformed, columns=feature_names, index=X.index)