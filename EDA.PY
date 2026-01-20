import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/C01_l01_ecommerce_retail_data.csv')


## EDA

df.shape

df.head()

df.info()

df.describe().transpose()


df_obj=df.select_dtypes('object')

for obj in df_obj.columns:
    print(df_obj[obj].unique())


df_int=df.select_dtypes('int64')

# null values
df.isnull().sum()
# Duplicated records
df.duplicated().sum()

# Step-by-step Python data cleaning plan (tailored to your dataset)
# Your CSV has 10,286 rows × 8 columns. A quick audit shows:

# 137 duplicate rows
# 308 missing values in order_amount_old
# Typos/inconsistent categories in customer_segment (e.g., premuim, platnum, standrad)
# Mixed date formats in date (e.g., 2024.12.05 and 29-06-2024) — but they’re fully parseable
# is_return is 0/1, hour_of_day is 9–22 (so it’s constrained to business hours in this dataset)


# Step by step
# 1) Load + baseline audit (shape, types, missingness, duplicates)
# 2) Standardize column names
# 3) Remove duplicate rows
# 4) Clean categorical text fields (customer_segment, payment_method)
# 5) Parse date safely (mixed formats)
# 6) Validate and enforce numeric columns
# 7) Handle missing values (specifically order_amount_old)
# 8) Add derived “quality” columns
# 9) Final assertions / sanity checks


# Step 2
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Step 3
df_clean = df.drop_duplicates(keep='first')

# Step 4

## Clean up customer_segment column

typo_mapping = {
    'premuim': 'premium',
    'platnum': 'platinum',
    'standrad': 'standard'
}

df_clean['customer_segment'] = df_clean['customer_segment'].astype(str).str.strip().str.lower()
df_clean['customer_segment'] = df_clean['customer_segment'].replace(typo_mapping)

df_clean.shape

# Step 5

date_str = df_clean['date'].astype(str).str.strip()

parsed_format1 = pd.to_datetime(date_str, format='%Y.%m.%d', errors='coerce')
parsed_format2 = pd.to_datetime(date_str, format='%d-%m-%Y', errors='coerce')
df_clean['date'] = parsed_format1.fillna(parsed_format2)

# Step 6: Validate and Enforce Numeric Columns

# 1 - order_amount_old Validation


col = "order_amount_old"

# Compute IQR bounds (ignoring NaNs)
q1 = df_clean[col].quantile(0.25)
q3 = df_clean[col].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Boolean mask: True where value is an outlier
outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)

# Rows that are outliers
df_outliers = df_clean.loc[outlier_mask].copy()

print("Outliers:", outlier_mask.sum())
print("Bounds:", lower_bound, upper_bound)
df_outliers

# 2 - cost Validation
pd.to_numeric(df_clean['cost'], errors='coerce')
(df_clean['cost'] < 0).sum()  # negative check
# Business rule validation
(df['cost'] > df['order_amount_old']).sum()


# 3 - hour_of_day Validation
pd.to_numeric(df_clean['hour_of_day'], errors='coerce')
df_clean['hour_of_day'].between(0, 23).sum()
df_clean['hour_of_day'].unique()

# 4 - is_return Validation
pd.to_numeric(df_clean['is_return'], errors='coerce')
df_clean['is_return'].isin([0, 1]).all()
df_clean['is_return'].value_counts()



