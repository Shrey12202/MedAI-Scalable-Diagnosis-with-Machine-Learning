from airflow import DAG
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

with DAG(
    dag_id="test_dag_hello_world",
    start_date=datetime.today() - timedelta(days=1),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    t1 = EmptyOperator(task_id="start")
