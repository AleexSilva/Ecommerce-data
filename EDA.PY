import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/C01_l01_ecommerce_retail_data.csv')


## EDA

df.head()

df.info()

df.describe().transpose()


df_obj=df.select_dtypes('object')

for obj in df_obj.columns:
    print(df_obj[obj].unique())


df_int=df.select_dtypes('int64')


df.isnull().sum()

df.duplicated().sum()

## Data Cleaning




