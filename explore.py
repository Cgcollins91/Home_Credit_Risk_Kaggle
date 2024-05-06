# %%
import polars as pl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
import os
import dask.dataframe as dd

dataPath         = "C:/Users/Chris/OneDrive/Documents/Coding/Competitions/Home Credit Risk/Home Credit Risk/"
feature_def_file = 'feature_definitions.csv'
train_folder     = dataPath + 'csv_files/train/'
train_files      = [f for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))]


# %%
df_feature_def = pd.read_csv(dataPath + feature_def_file)
var_def        = dict()
for i in range (len(df_feature_def)):
    var_def[df_feature_def['Variable'][i]] = df_feature_def['Description'][i]


df         = pd.read_csv(train_folder + train_files[0])

for i in range(1, len(train_files)):
    ddf_i = pd.read_csv(train_folder + train_files[i])
    df    = pd.merge(df, ddf_i, on='case_id', how='left')








# %%

# %%
for col in train_cols:
    if col in var_def.keys():
        df.rename(columns={col: var_def[col]}, inplace=True)