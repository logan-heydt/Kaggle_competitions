import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn import set_config
import numpy as np

set_config(transform_output="pandas")


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["house_age"] = df['YrSold'] - df['YearBuilt']

    df["DecadeBuilt"] = (df["YearBuilt"] // 10) * 10

    df["yrs_since_remodel"] = df['YrSold'] - df['YearRemodAdd']

    quality_mapping = {"Ex": 1, "Gd": 1, "TA": 2, "Fa": 3, "Po": 3}

    # zoning_mapping = {"A": "Non-Resi",
    #                   "C": "Non-Resi",
    #                   "I": "Non-Resi",
    #                   "FV": "RM",
    #                   "RH": "RH",
    #                   "RL": "RL",
    #                   "RP": "RL",
    #                   "RM": "RM"}
    
    # df['zoning_bucketed'] = df['MSZoning'].map(zoning_mapping).fillna("Unknown")


# THIS MAPPING ------------------------------------------------------------

    subclass_story_mapping = {20: 'one_story',
                              30: 'one_story',
                              40: 'one_story',
                              120: 'one_story',
                              45: 'one_half_story',
                              50: 'one_half_story',
                              150: 'one_half_story',
                              60: 'two_story',
                              70: 'two_story',
                              160: 'two_story',
                              75: 'two_half_story',
                              80: 'other',
                              85: 'other',
                              90: 'other',
                              180: 'other',
                              190: 'other'}
    
    pud_mapping = {20: 0,
                30: 0,
                40: 0,
                45: 0,
                50: 0,
                60: 0,
                70: 0,
                75: 0,
                80: 0,
                85: 0,
                90: 0,
                120: 1,
                150: 1,
                160: 1,
                180: 1,
                190: 1}
    
    # df['is_pud'] = df['MSSubClass'].map(pud_mapping).fillna(-1).astype("int").astype("str")

    # df['stories'] = df['MSSubClass'].map(subclass_story_mapping)


# TO HERE ------------------------------

    df["has_vaneer"] = np.where(df["MasVnrType"].isna(), 0, 1)

    # Exterior Quality
    df["ExterCond_bucketed"] = df["ExterCond"].map(quality_mapping).fillna(-1).astype("int").astype("str")
    df["ExterQual_bucketed"] = df["ExterQual"].map(quality_mapping).fillna(-1).astype("int").astype("str")

    # Basement
    df["BsmtQual"] =  df["BsmtQual"].astype("string").fillna("NA")
    df["BsmtCond"] =  df["BsmtCond"].astype("string").fillna("NA")

    df["has_basement"] = df["BsmtQual"].apply(lambda x: 0 if x == "NA" else 1)
    
    df["BsmtQual_bucketed"] = df["BsmtQual"].map(quality_mapping).fillna(-1).astype("int").astype("str")
    df["BsmtCond_bucketed"] = df["BsmtCond"].map(quality_mapping).fillna(-1).astype("int").astype("str")

    fin_type_mapping = {
    "GLQ": 1,   # Good Living Quarters
    "ALQ": 2,   # Average Living Quarters
    "Rec": 2,   # Average Rec Room
    "BLQ": 3,   # Below Average
    "LwQ": 3,   # Low Quality
    "Unf": 4,   # Unfinished
    "NA": 4     # No Basement
    }

    df["BsmtFinType1_bucketed"] = df["BsmtFinType1"].map(fin_type_mapping).fillna(4).astype("int").astype("str")

    # HVAC 
    df["HeatingQC_bucketed"] = df["HeatingQC"].map(quality_mapping).fillna(-1).astype("int").astype("str")
    df["CentralAir_bool"] = df["CentralAir"].map({"Y": True, "N": False})

    # Kitchen
    df["KitchenQual_bucketed"] = df["KitchenQual"].map(quality_mapping).fillna(-1).astype("int").astype("str")

    # Fireplace
    df["has_fireplace"] = df["FireplaceQu"].apply(lambda x: 0 if x == "NA" else 1)

    # Garage
    df['has_garage'] = df["GarageQual"].apply(lambda x: 0 if x == "NA" else 1)
    df["GarageQual_bucketed"] = df["GarageQual"].map(quality_mapping).fillna(4).astype("int").astype("str")
    df["GarageCond_bucketed"] = df["GarageCond"].map(quality_mapping).fillna(4).astype("int").astype("str")
    # df["garage_space_per_car"] = (df["GarageArea"].fillna(0) / df["GarageCars"].replace(0, np.nan))
    df['garage_overal'] = df['GarageQual_bucketed'].astype(int) * df['GarageCond_bucketed'].astype(int)
 
    # Pool
    df["has_pool"] = df["PoolQC"].apply(lambda x: 0 if x == "NA" else 1)

    # Fence
    df["has_fence"] = df["Fence"].apply(lambda x: 0 if x == "NA" else 1)

    # Sale type
    df["is_normal_sale"] = df["SaleCondition"].apply(lambda x: True if x == "Normal" else False)

    # Porch
    porch_cols = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]

    # Baths
    df['ttl_baths'] = df['FullBath'].fillna(0) + (df['HalfBath'].fillna(0) * .5)

    df["has_porch"] = (
        df[porch_cols]
        .fillna(0)
        .sum(axis=1)
        .gt(0)
        .astype(int))

    # Overall
    overall_mapping = {1: 1, 
                       2: 1, 
                       3: 2, 
                       4: 2, 
                       5: 2,
                       6: 3,
                       7: 3,
                       8: 3,
                       9: 4,
                       10: 4}

    df["OverallCond_q"] = df["OverallCond"].map(overall_mapping).fillna(-1).astype("int").astype("str")
    df["OverallQual_q"] = df["OverallQual"].map(overall_mapping).fillna(-1).astype("int").astype("str")

    df["MSSubClass"] = df["MSSubClass"].astype("int").astype("str")

    df["Electrical"] = df["Electrical"].fillna("SBrkr")

    df['has_misc_feat'] = df["MiscFeature"].apply(lambda x: 0 if x == "NA" else 1)

    df['LotFrontage'] = df['LotFrontage'].astype(int)

    df['frontage_area_mult'] = df['LotArea'] * df['LotFrontage']

    drop_cols = ["OverallCond", "OverallQual", "ExterCond", "ExterQual", "BsmtQual", "BsmtCond", "BsmtFinType1", "CentralAir", 
                     "HeatingQC", "KitchenQual","FireplaceQu", "GarageCond", "GarageQual", "PoolQC", "Fence", "SaleCondition",
                      "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "YearRemodAdd", "YearBuilt",
                      # don't know what to do with these columns
                       "MasVnrArea", 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                      'BsmtFullBath', 'BsmtHalfBath','BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces', 'GarageYrBlt', 
                      'GarageArea', 'PoolArea', 'MiscVal','Condition1', 'Condition2', 'RoofStyle', 'RoofMatl',
                      'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation','BsmtExposure', 'BsmtFinType2','GarageType', 'GarageCars',
                      'GarageFinish', 'PavedDrive','MiscFeature', "Alley", "BsmtFinSF1", "BsmtFinSF2", "1stFlrSF",
                       "2ndFlrSF", "LowQualFinSF", 'BsmtFullBath','BsmtHalfBath', "FullBath", "HalfBath",
                        'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF', "MasVnrArea", "Id",
                        # 'LotArea', 'LotFrontage', 'GrLivArea', 

                        ]
    
    df_eng = df.drop(columns=drop_cols)

    return df_eng

impute_cols = ["LotFrontage"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, impute_cols)
], remainder='passthrough', verbose_feature_names_out=False)

pipeline_base = Pipeline([
    ('preprocessing', preprocessor),
    ('feature_engineering', FunctionTransformer(feature_engineer, validate=False))
])


def get_fitted_pipeline(X: pd.DataFrame):

    pipeline_base.fit(X)

    # encoder = pipeline_base.named_steps['preprocessing']

    transformed = pipeline_base.transform(X)
    return pipeline_base, transformed
