"""
E-commerce Retail Data Cleaning Pipeline
Apache Airflow DAG

This DAG performs data cleaning steps on e-commerce retail data:
- Step 2: Standardize column names
- Step 3: Remove duplicate rows
- Step 4: Clean categorical text fields
- Step 5: Parse date column (mixed formats)
- Step 7: Handle missing values
- Step 8: Add derived columns (profit & margin)
- Step 9: Final assertions & sanity checks

Data Quality Checks:
- Not null validation
- No duplicates warning
- Cost > 0 validation
- Customer segment count <= 3
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email': ['alee.silva94@gmaIL.COM'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize DAG
dag = DAG(
    'ecommerce_data_cleaning_pipeline',
    default_args=default_args,
    description='Clean and validate e-commerce retail data',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['data-cleaning', 'ecommerce', 'etl'],
)

# Global variables for file paths
INPUT_FILE = '../../../data/input/C01_l01_ecommerce_retail_data.csv'
OUTPUT_FILE = '../../../data/output/ecommerce_data_cleaned.csv'
TEMP_FILE = '/tmp/ecommerce_temp.csv'


def load_data(**context):
    """Task 1: Load raw data from CSV"""
    logging.info(f"Loading data from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    logging.info(f"Data loaded successfully. Shape: {df.shape}")

    # Save to temp location and push to XCom
    df.to_csv(TEMP_FILE, index=False)
    context['ti'].xcom_push(key='row_count_raw', value=len(df))

    return f"Loaded {len(df)} rows"


def step_2_standardize_columns(**context):
    """Task 2: Standardize column names"""
    logging.info("Step 2: Standardizing column names")

    df = pd.read_csv(TEMP_FILE)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    logging.info(f"Columns standardized: {list(df.columns)}")

    # Save progress
    df.to_csv(TEMP_FILE, index=False)

    return f"Columns: {list(df.columns)}"


def step_3_remove_duplicates(**context):
    """Task 3: Remove duplicate rows"""
    logging.info("Step 3: Removing duplicate rows")

    df = pd.read_csv(TEMP_FILE)

    initial_count = len(df)
    duplicate_count = df.duplicated().sum()

    # Remove duplicates
    df = df.drop_duplicates(keep='first')

    final_count = len(df)
    removed = initial_count - final_count

    logging.info(f"Duplicates found: {duplicate_count}")
    logging.info(f"Rows removed: {removed}")
    logging.info(f"Rows remaining: {final_count}")

    # WARNING: Check for duplicates
    if duplicate_count > 0:
        logging.warning(f"WARNING: {duplicate_count} duplicate rows were found and removed!")

    # Save progress
    df.to_csv(TEMP_FILE, index=False)
    context['ti'].xcom_push(key='duplicates_removed', value=removed)

    return f"Removed {removed} duplicates"


def step_4_clean_categorical(**context):
    """Task 4: Clean categorical text fields"""
    logging.info("Step 4: Cleaning categorical fields")

    df = pd.read_csv(TEMP_FILE)

    # Normalize categorical fields
    df['customer_segment'] = df['customer_segment'].astype(str).str.strip().str.lower()
    df['payment_method'] = df['payment_method'].astype(str).str.strip().str.lower()

    # Fix known typos
    typo_mapping = {
        'premuim': 'premium',
        'platnum': 'platinum',
        'standrad': 'standard'
    }

    df['customer_segment'] = df['customer_segment'].replace(typo_mapping)

    logging.info(f"Customer segments: {df['customer_segment'].unique()}")
    logging.info(f"Payment methods: {df['payment_method'].unique()}")

    # Save progress
    df.to_csv(TEMP_FILE, index=False)

    return "Categorical fields cleaned"


def step_5_parse_dates(**context):
    """Task 5: Parse date column with mixed formats"""
    logging.info("Step 5: Parsing date column")

    df = pd.read_csv(TEMP_FILE)

    # Parse mixed date formats
    date_str = df['date'].astype(str).str.strip()

    # Try format 1: YYYY.MM.DD
    parsed_format1 = pd.to_datetime(date_str, format='%Y.%m.%d', errors='coerce')

    # Try format 2: DD-MM-YYYY
    parsed_format2 = pd.to_datetime(date_str, format='%d-%m-%Y', errors='coerce')

    # Combine both
    df['date'] = parsed_format1.fillna(parsed_format2)

    failed_parses = df['date'].isna().sum()

    if failed_parses > 0:
        logging.warning(f"WARNING: {failed_parses} dates failed to parse!")
    else:
        logging.info("All dates parsed successfully")

    logging.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Save progress
    df.to_csv(TEMP_FILE, index=False)

    return f"Dates parsed. Range: {df['date'].min()} to {df['date'].max()}"


def step_7_handle_missing(**context):
    """Task 7: Handle missing values with segment-wise median imputation"""
    logging.info("Step 7: Handling missing values")

    df = pd.read_csv(TEMP_FILE)

    # Parse date back to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Convert numeric columns
    df['order_amount_old'] = pd.to_numeric(df['order_amount_old'], errors='coerce')
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    df['hour_of_day'] = pd.to_numeric(df['hour_of_day'], errors='coerce')
    df['is_return'] = pd.to_numeric(df['is_return'], errors='coerce')

    # Check missing values before
    missing_before = df.isna().sum()
    logging.info(f"Missing values before imputation: {missing_before[missing_before > 0].to_dict()}")

    # Impute order_amount_old with segment-wise median
    if df['order_amount_old'].isna().sum() > 0:
        df['order_amount_old'] = df.groupby('customer_segment')['order_amount_old'].transform(
            lambda x: x.fillna(x.median())
        )
        logging.info("Missing order_amount_old imputed with segment median")

    # Check missing values after
    missing_after = df.isna().sum()

    if missing_after.sum() > 0:
        logging.warning(f"WARNING: {missing_after.sum()} missing values remain!")
        logging.warning(f"{missing_after[missing_after > 0].to_dict()}")
    else:
        logging.info("All missing values handled")

    # Save progress
    df.to_csv(TEMP_FILE, index=False)

    return "Missing values handled"


def step_8_add_derived_columns(**context):
    """Task 8: Add derived columns (profit & margin)"""
    logging.info("Step 8: Adding derived columns")

    df = pd.read_csv(TEMP_FILE)

    # Parse date back to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Calculate profit
    df['profit'] = df['order_amount_old'] - df['cost']

    # Calculate margin (safe division)
    df['margin'] = np.where(
        df['order_amount_old'] > 0,
        (df['profit'] / df['order_amount_old']) * 100,
        0
    )

    logging.info(f"Profit - Mean: ${df['profit'].mean():.2f}, Median: ${df['profit'].median():.2f}")
    logging.info(f"Margin - Mean: {df['margin'].mean():.2f}%, Median: {df['margin'].median():.2f}%")

    # Check for negative profit
    negative_profit = (df['profit'] < 0).sum()
    if negative_profit > 0:
        logging.warning(f"WARNING: {negative_profit} rows have negative profit!")

    # Save progress
    df.to_csv(TEMP_FILE, index=False)

    return "Added profit and margin columns"


def step_9_data_quality_checks(**context):
    """Task 9: Final data quality checks and validations"""
    logging.info("Step 9: Running data quality checks")

    df = pd.read_csv(TEMP_FILE)

    # Parse date back to datetime
    df['date'] = pd.to_datetime(df['date'])

    validation_passed = True

    # CHECK 1: NOT NULL VALIDATION
    logging.info("=" * 70)
    logging.info("CHECK 1: NOT NULL VALIDATION")
    logging.info("=" * 70)

    null_counts = df.isna().sum()

    if null_counts.sum() > 0:
        validation_passed = False
        logging.error(f"FAILED: Found {null_counts.sum()} null values!")
        for col, count in null_counts[null_counts > 0].items():
            logging.error(f"  - {col}: {count} null values")
    else:
        logging.info("PASSED: No null values found")

    # CHECK 2: NO DUPLICATES WARNING
    logging.info("=" * 70)
    logging.info("CHECK 2: NO DUPLICATES WARNING")
    logging.info("=" * 70)

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        logging.warning(f"WARNING: {duplicate_count} duplicate rows found!")
        validation_passed = False
    else:
        logging.info("PASSED: No duplicate rows")

    # CHECK 3: COST > 0 VALIDATION
    logging.info("=" * 70)
    logging.info("CHECK 3: COST > 0 VALIDATION")
    logging.info("=" * 70)

    zero_or_negative_cost = (df['cost'] <= 0).sum()

    if zero_or_negative_cost > 0:
        validation_passed = False
        logging.error(f"FAILED: {zero_or_negative_cost} rows have cost <= 0!")
        logging.error(f"  Min cost: {df['cost'].min()}")
    else:
        logging.info(f"PASSED: All costs > 0 (min: ${df['cost'].min():.2f})")

    # CHECK 4: CUSTOMER SEGMENT COUNT <= 3
    logging.info("=" * 70)
    logging.info("CHECK 4: CUSTOMER SEGMENT COUNT <= 3")
    logging.info("=" * 70)

    segment_count = df['customer_segment'].nunique()
    segments = sorted(df['customer_segment'].unique())

    logging.info(f"Unique customer segments: {segment_count}")
    logging.info(f"Segments: {segments}")

    if segment_count > 3:
        validation_passed = False
        logging.error(f"FAILED: Found {segment_count} customer segments (expected <= 3)!")
        logging.error(f"  Segments: {segments}")
    else:
        logging.info("PASSED: Customer segments <= 3")

    # ADDITIONAL CHECKS
    logging.info("=" * 70)
    logging.info("ADDITIONAL VALIDATIONS")
    logging.info("=" * 70)

    # Check binary values
    is_return_valid = df['is_return'].isin([0, 1]).all()
    if not is_return_valid:
        validation_passed = False
        logging.error("FAILED: is_return contains invalid values!")
    else:
        logging.info("is_return is binary (0/1)")

    # FINAL SUMMARY
    logging.info("=" * 70)
    logging.info("DATA QUALITY SUMMARY")
    logging.info("=" * 70)

    logging.info(f"Final dataset shape: {df.shape}")
    logging.info(f"Total rows: {len(df)}")
    logging.info(f"Total columns: {len(df.columns)}")

    if validation_passed:
        logging.info("ALL DATA QUALITY CHECKS PASSED!")
    else:
        logging.error("SOME DATA QUALITY CHECKS FAILED!")
        raise ValueError("Data quality validation failed. Check logs for details.")

    # Save final cleaned data
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Cleaned data saved to: {OUTPUT_FILE}")

    context['ti'].xcom_push(key='final_row_count', value=len(df))
    context['ti'].xcom_push(key='validation_passed', value=validation_passed)

    return "Data quality checks completed"


def generate_summary_report(**context):
    """Task 10: Generate summary report"""
    logging.info("Generating pipeline summary report")

    ti = context['ti']

    # Pull metrics from XCom
    raw_count = ti.xcom_pull(key='row_count_raw', task_ids='load_data')
    duplicates_removed = ti.xcom_pull(key='duplicates_removed', task_ids='step_3_remove_duplicates')
    final_count = ti.xcom_pull(key='final_row_count', task_ids='step_9_data_quality_checks')
    validation_passed = ti.xcom_pull(key='validation_passed', task_ids='step_9_data_quality_checks')

    # Load final data for statistics
    df = pd.read_csv(OUTPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])

    status = 'PASSED' if validation_passed else 'FAILED'

    logging.info("=" * 70)
    logging.info("E-COMMERCE DATA CLEANING PIPELINE - SUMMARY REPORT")
    logging.info("=" * 70)
    logging.info(f"Pipeline Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("")
    logging.info("DATA PROCESSING SUMMARY:")
    logging.info(f"- Raw data rows: {raw_count}")
    logging.info(f"- Duplicates removed: {duplicates_removed}")
    logging.info(f"- Final cleaned rows: {final_count}")
    logging.info("")
    logging.info("DATA QUALITY STATUS:")
    logging.info(f"- Validation Status: {status}")
    logging.info(f"- Null values: {df.isna().sum().sum()}")
    logging.info(f"- Duplicate rows: {df.duplicated().sum()}")
    logging.info(f"- Customer segments: {df['customer_segment'].nunique()}")
    logging.info("")
    logging.info("BUSINESS METRICS:")
    logging.info(f"- Average order amount: ${df['order_amount_old'].mean():.2f}")
    logging.info(f"- Average profit: ${df['profit'].mean():.2f}")
    logging.info(f"- Average margin: {df['margin'].mean():.2f}%")
    logging.info(f"- Return rate: {(df['is_return'].sum() / len(df) * 100):.2f}%")
    logging.info("")
    logging.info(f"OUTPUT FILE: {OUTPUT_FILE}")
    logging.info("=" * 70)

    return "Summary report generated"


# DEFINE TASKS
task_load = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

task_step_2 = PythonOperator(
    task_id='step_2_standardize_columns',
    python_callable=step_2_standardize_columns,
    dag=dag,
)

task_step_3 = PythonOperator(
    task_id='step_3_remove_duplicates',
    python_callable=step_3_remove_duplicates,
    dag=dag,
)

task_step_4 = PythonOperator(
    task_id='step_4_clean_categorical',
    python_callable=step_4_clean_categorical,
    dag=dag,
)

task_step_5 = PythonOperator(
    task_id='step_5_parse_dates',
    python_callable=step_5_parse_dates,
    dag=dag,
)

task_step_7 = PythonOperator(
    task_id='step_7_handle_missing',
    python_callable=step_7_handle_missing,
    dag=dag,
)

task_step_8 = PythonOperator(
    task_id='step_8_add_derived_columns',
    python_callable=step_8_add_derived_columns,
    dag=dag,
)

task_step_9 = PythonOperator(
    task_id='step_9_data_quality_checks',
    python_callable=step_9_data_quality_checks,
    dag=dag,
)

task_summary = PythonOperator(
    task_id='generate_summary_report',
    python_callable=generate_summary_report,
    dag=dag,
)

# DEFINE TASK DEPENDENCIES
task_load >> task_step_2 >> task_step_3 >> task_step_4 >> task_step_5 >> task_step_7 >> task_step_8 >> task_step_9 >> task_summary
