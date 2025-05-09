from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import boto3
import os
from botocore.exceptions import ClientError
import random
import requests

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

SAMPLE_SIZE = 5
LOW_CONFIDENCE_THRESHOLD = 0.7
PROJECT_NAME = "Food11 Continuous X"

def ensure_buckets_exist(bucket_names):
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_URL'],
        aws_access_key_id=os.environ['MINIO_USER'],
        aws_secret_access_key=os.environ['MINIO_PASSWORD'],
        region_name="us-east-1"
    )
    existing_buckets = {b['Name'] for b in s3.list_buckets()['Buckets']}
    for bucket in bucket_names:
        if bucket not in existing_buckets:
            print(f"Creating bucket: {bucket}")
            s3.create_bucket(Bucket=bucket)
        else:
            print(f"Bucket already exists: {bucket}")

def init_buckets_task(**context):
    ensure_buckets_exist(['production-label-wait', 'production-noisy'])

def sample_production_images(**context):
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_URL'],
        aws_access_key_id=os.environ['MINIO_USER'],
        aws_secret_access_key=os.environ['MINIO_PASSWORD'],
        region_name="us-east-1"
    )

    if context['dag_run'].external_trigger:
        # Manual run — use current time window
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=30)
        print("Manual trigger, using real-time window:", start, "to", end)
    else:
        # Scheduled run — use the defined data interval
        start = context['data_interval_start'].astimezone(timezone.utc)
        end = context['data_interval_end'].astimezone(timezone.utc)
        print("Scheduled run, using data interval:", start, "to", end)

    low_conf, flagged, corrected, others = [], [], [], []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket='production'):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            tags = s3.get_object_tagging(Bucket='production', Key=key)['TagSet']
            tag_dict = {t['Key']: t['Value'] for t in tags}

            timestamp = tag_dict.get("timestamp")
            if not timestamp:
                continue

            ts = datetime.fromisoformat(timestamp)
            if not (start <= ts < end):
                continue

            confidence = tag_dict.get("confidence")
            predicted_class = tag_dict.get("predicted_class", "")
            corrected_class = tag_dict.get("corrected_class", "")
            flagged_bool = tag_dict.get("flagged") == "true"

            item = {
                "key": key,
                "confidence": float(confidence) if confidence else None,
                "predicted_class": predicted_class,
                "corrected_class": corrected_class,
                "flagged": flagged_bool
            }

            if item["confidence"] is not None and item["confidence"] < LOW_CONFIDENCE_THRESHOLD:
                low_conf.append(item)
            elif flagged_bool:
                flagged.append(item)
            elif corrected_class:
                corrected.append(item)
            else:
                others.append(item)

    # Step 1: Get existing keys in production-label-wait
    existing_keys = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket='production-label-wait'):
        for obj in page.get("Contents", []):
            existing_keys.add(obj["Key"])

    # Step 2: Filter all lists
    low_conf = [item for item in low_conf if item["key"] not in existing_keys]
    flagged = [item for item in flagged if item["key"] not in existing_keys]
    corrected = [item for item in corrected if item["key"] not in existing_keys]
    others = [item for item in others if item["key"] not in existing_keys]

    print("Low confidence:", len(low_conf))
    print("Flagged:", len(flagged))
    print("Corrected:", len(corrected))
    print("Others:", len(others))
    print("Existing in label-wait:", len(existing_keys))

    # Step 3: Build selected list
    selected = low_conf + flagged + corrected
    remaining = [obj for obj in others if obj["key"] not in {i["key"] for i in selected}]
    random_others = random.sample(remaining, min(SAMPLE_SIZE, len(remaining)))
    selected += random_others

    all_items = selected + remaining

    all_items = selected + remaining

    context['ti'].xcom_push(key='selected_images', value=selected)
    context['ti'].xcom_push(key='all_images', value=all_items)

def move_sampled_images(**context):
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['MINIO_URL'],
        aws_access_key_id=os.environ['MINIO_USER'],
        aws_secret_access_key=os.environ['MINIO_PASSWORD'],
        region_name="us-east-1"
    )

    selected = context['ti'].xcom_pull(key='selected_images', task_ids='sample_production_images')
    all_items = context['ti'].xcom_pull(key='all_images', task_ids='sample_production_images')
    selected_keys = {item['key'] for item in selected}

    for item in all_items:
        source_key = item['key']
        target_bucket = 'production-label-wait' if source_key in selected_keys else 'production-noisy'
        s3.copy_object(
            Bucket=target_bucket,
            CopySource={'Bucket': 'production', 'Key': source_key},
            Key=source_key
        )
        #s3.delete_object(Bucket='production', Key=source_key)

def create_label_studio_project(**context):
    label_studio_url = os.environ['LABEL_STUDIO_URL']
    token = os.environ['LABEL_STUDIO_USER_TOKEN']
    headers = {"Authorization": f"Token {token}"}

    response = requests.get(f"{label_studio_url}/api/projects", headers=headers)
    response.raise_for_status()
    projects = response.json().get('results', [])

    for p in projects:
        if p['title'] == PROJECT_NAME:
            context['ti'].xcom_push(key='project_id', value=p['id'])
            return

    label_config = """
    <View>
      <Image name="image" value="$image" maxWidth="500px"/>
      <Choices name="label" toName="image" choice="single" showInLine="true" >
        <Choice value="Bread"/>
        <Choice value="Dairy product"/>
        <Choice value="Dessert"/>
        <Choice value="Egg"/>
        <Choice value="Fried food"/>
        <Choice value="Meat"/>
        <Choice value="Noodles/Pasta"/>
        <Choice value="Rice"/>
        <Choice value="Seafood"/>
        <Choice value="Soup"/>
        <Choice value="Vegetable/Fruit"/>
      </Choices>
      <Header value="Model Confidence: $confidence"/>
      <Header value="Predicted Class: $predicted_class"/>
      <Header value="Corrected Class: $corrected_class"/>
    </View>
    """
    payload = {
        "title": PROJECT_NAME,
        "label_config": label_config
    }
    res = requests.post(f"{label_studio_url}/api/projects", headers=headers, json=payload)
    res.raise_for_status()
    project_id = res.json()["id"]
    context['ti'].xcom_push(key='project_id', value=project_id)

def send_tasks_to_label_studio(**context):
    selected = context['ti'].xcom_pull(key='selected_images', task_ids='sample_production_images')
    project_id = context['ti'].xcom_pull(task_ids='create_label_studio_project', key='project_id')

    if not selected:
        return
 
    public_ip = requests.get("http://169.254.169.254/latest/meta-data/public-ipv4").text.strip()
    s3 = boto3.client(
        's3',
        endpoint_url=f"http://{public_ip}:9000",
        aws_access_key_id=os.environ['MINIO_USER'],
        aws_secret_access_key=os.environ['MINIO_PASSWORD'],
        region_name="us-east-1"
    )

    label_studio_url = os.environ['LABEL_STUDIO_URL']
    token = os.environ['LABEL_STUDIO_USER_TOKEN']
    headers = {"Authorization": f"Token {token}"}

    tasks = []
    for item in selected:
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'production-label-wait', 'Key': item['key']},
            ExpiresIn=3600
        )
        tasks.append({
            "data": {
                "image": presigned_url,
                "confidence": str(item['confidence']) if item['confidence'] is not None else "",
                "predicted_class": item['predicted_class'],
                "corrected_class": item['corrected_class'],
                "flagged": item['flagged']
            },
            "meta": {"original_key": item['key']}
        })

    response = requests.post(
        f"{label_studio_url}/api/projects/{project_id}/import",
        json=tasks,
        headers=headers
    )

with DAG(
    dag_id='pipeline_1_get_and_label',
    default_args=default_args,
    description='Sample and tag production data for human review',
    start_date=datetime.today() - timedelta(days=1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    init_buckets = PythonOperator(
        task_id='init_buckets',
        python_callable=init_buckets_task
    )

    sample_production_images_task = PythonOperator(
        task_id='sample_production_images',
        python_callable=sample_production_images
    )

    move_sampled_images_task = PythonOperator(
        task_id='move_sampled_images',
        python_callable=move_sampled_images
    )

    create_project_task = PythonOperator(
        task_id='create_label_studio_project',
        python_callable=create_label_studio_project
    )

    label_studio_task = PythonOperator(
        task_id='send_tasks_to_label_studio',
        python_callable=send_tasks_to_label_studio
    )

    init_buckets >> sample_production_images_task >> move_sampled_images_task
    [move_sampled_images_task, create_project_task] >> label_studio_task
