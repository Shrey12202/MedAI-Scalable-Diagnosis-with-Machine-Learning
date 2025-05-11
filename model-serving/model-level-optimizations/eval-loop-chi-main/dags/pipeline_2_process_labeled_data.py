from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
import os
import requests

# ENV vars
LABEL_STUDIO_URL = os.environ['LABEL_STUDIO_URL']
LABEL_STUDIO_TOKEN = os.environ['LABEL_STUDIO_USER_TOKEN']
MINIO_URL = os.environ['MINIO_URL']
MINIO_USER = os.environ['MINIO_USER']
MINIO_PASSWORD = os.environ['MINIO_PASSWORD']

PROJECT_NAME = "Food11 Continuous X"

# S3 Client
s3 = boto3.client(
    's3',
    endpoint_url=MINIO_URL,
    aws_access_key_id=MINIO_USER,
    aws_secret_access_key=MINIO_PASSWORD,
    region_name="us-east-1"
)

def ensure_bucket(bucket):
    existing = {b['Name'] for b in s3.list_buckets()['Buckets']}
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket)

def get_label_studio_results(**context):
    headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
    response = requests.get(f"{LABEL_STUDIO_URL}/api/projects", headers=headers)
    response.raise_for_status()
    projects = response.json().get("results", [])
    project_id = next((p['id'] for p in projects if p['title'] == PROJECT_NAME), None)

    if not project_id:
        raise Exception("Label Studio project not found.")

    tasks_response = requests.get(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks?completed=true",
        headers=headers
    )
    tasks_response.raise_for_status()
    completed_tasks = tasks_response.json()

    context['ti'].xcom_push(key='completed_tasks', value=completed_tasks)
    context['ti'].xcom_push(key='project_id', value=project_id)

def process_labeled_data_minio(**context):
    ensure_bucket("production-clean")
    ensure_bucket("production-noisy")

    completed_tasks = context['ti'].xcom_pull(key='completed_tasks', task_ids='get_label_studio_results')

    CLASSES = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax"
    ]


    for task in completed_tasks:
        try:
            original_key = task["meta"]["original_key"]
            label = task["annotations"][0]["result"][0]["value"]["choices"][0]
            class_index = CLASSES.index(label)
            class_dir = f"class_{class_index:02d}"
            filename = original_key.split("/")[-1]
            new_key = f"{class_dir}/{filename}"

            # Copy to production-clean
            copy_source = {'Bucket': 'production-label-wait', 'Key': original_key}
            s3.copy_object(
                Bucket='production-clean',
                CopySource=copy_source,
                Key=new_key
            )

            # Copy to production-noisy
            s3.copy_object(
                Bucket='production-noisy',
                CopySource=copy_source,
                Key=new_key
            )

            # Preserve tags
            tags = s3.get_object_tagging(
                Bucket='production-label-wait', Key=original_key
            )['TagSet']
            s3.put_object_tagging(
                Bucket='production-clean',
                Key=new_key,
                Tagging={'TagSet': tags}
            )

            # Delete from production-label-wait
            try:
                s3.head_object(Bucket='production-label-wait', Key=original_key)
                s3.delete_object(Bucket='production-label-wait', Key=original_key)
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print(f"Object not found for deletion: {original_key}")
                else:
                    raise

            print(f"Processed {original_key} → new key '{new_key}' with label '{label}'")

        except Exception as e:
            print(f"Error processing task {task.get('id')} (maybe this object was not labeled yet): {e}")

def report_batch_accuracy(**context):
    headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
    response = requests.get(f"{LABEL_STUDIO_URL}/api/projects", headers=headers)
    response.raise_for_status()
    projects = response.json().get("results", [])
    project_id = next((p['id'] for p in projects if p['title'] == PROJECT_NAME), None)
    if not project_id:
        print("Project not found for accuracy report.")
        return

    tasks_response = requests.get(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks?completed=true",
        headers=headers
    )
    tasks_response.raise_for_status()
    tasks = tasks_response.json()

    total = len(tasks)
    correct = 0
    for task in tasks:
        try:
            predicted = task['data']['predicted_class']
            corrected = task['annotations'][0]['result'][0]['value']['choices'][0]
            if predicted == corrected:
                correct += 1
        except Exception as e:
            print(f"Skipping task {task.get('id')}: {e}")

    if total > 0:
        accuracy = correct / total
        print(f"Accuracy on batch: {accuracy:.2%} ({correct}/{total})")
    else:
        print("No completed tasks to evaluate.")

# DAG definition
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="pipeline_2_process_labeled_data",
    default_args=default_args,
    start_date=datetime.today() - timedelta(days=1),
    schedule_interval="@daily",
    catchup=False,
    description="Process human-labeled data and calculate accuracy on random sample",
) as dag:

    get_results_task = PythonOperator(
        task_id="get_label_studio_results",
        python_callable=get_label_studio_results
    )

    process_task = PythonOperator(
        task_id="process_labeled_data_minio",
        python_callable=process_labeled_data_minio
    )

    report_task = PythonOperator(
        task_id="report_batch_accuracy",
        python_callable=report_batch_accuracy
    )

    get_results_task >> process_task >> report_task
