from datetime import timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from etl_functions import get_dataframes, transform_dataframes, insert_data_into_tables

# Setting up a DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'marketing_data_etl',
    default_args=default_args,
    description='ETL pipeline to download the data from Kaggle, transform it, and insert into the database',
    schedule_interval=None,
    catchup=False
)

# Defining parameters to connect to the database
params = {
    'host': 'localhost',
    'user': 'postgres',
    'password': '1234',
    'port': 5432,
    'dbname': 'marketing_data'
}

# Temporary directory for csv/xlsx files from Kaggle
directory = '/tmp/kaggle_data'

# Extract step (part 1): downloading data from Kaggle using Kaggle API
download_from_kaggle = BashOperator(
    task_id='extract_step_1',
    bash_command=f'kaggle datasets download -d rishikumarrajvansh/marketing-insights-for-e-commerce-company --path {directory} --unzip',
    dag=dag
)

# Extract step (part 2): creating dataframes from the csv/xlsx files in the temporary directory
get_data_from_files = PythonOperator(
    task_id='extract_step_2',
    python_callable=get_dataframes,
    op_kwargs={'directory': directory},
    dag=dag
    )

# Transform step: transforming data so that it fully matches the structure of tables in the database
transform_data = PythonOperator(
    task_id='transform_step',
    python_callable=transform_dataframes,
    dag=dag
)

# Load step: loading transformed data into the database
load_data = PythonOperator(
    task_id='load_step',
    python_callable=insert_data_into_tables,
    op_kwargs={'conn_params': params},
    dag=dag
)

# Defining a sequence of operations
download_from_kaggle >> get_data_from_files >> transform_data >> load_data