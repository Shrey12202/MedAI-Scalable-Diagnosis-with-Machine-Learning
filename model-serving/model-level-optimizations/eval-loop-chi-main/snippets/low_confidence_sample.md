


::: {.cell .markdown}

### Label low-confidence production images

In the previous section, we used uniform random sampling to select production images for human annotation. In practice, however, we may want to preferentially label samples for which the model has low confidence. Combined with random sampling, this is a powerful strategy because:

* We can create a high-quality but small evaluation set using randomly sampled production images
* and in parallel, create a lower quality but large-volume re-training set, using model-labeled production images for which the confidence is high (assuming the model labeled them correctly) along with human-labeled production images for which the confidence is low

:::


::: {.cell .markdown}

We will use the Label Studio API to automate the setup of a new project and tasks. 

:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
import requests
import boto3 
import os
import random
```
:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
LABEL_STUDIO_URL = os.environ['LABEL_STUDIO_URL']
LABEL_STUDIO_TOKEN = os.environ['LABEL_STUDIO_USER_TOKEN']
```
:::

::: {.cell .markdown}

For this project, our labeling UI will be slightly different - we are going to also display the model's predicted class and confidence in its prediction:

:::



::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
LABEL_CONFIG = """
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
</View>
"""
```
:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
project_config = {
    "title": "Food11 Low Confidence",
    "label_config": LABEL_CONFIG
}
res = requests.post(f"{LABEL_STUDIO_URL}/api/projects", json=project_config, headers=headers)
if res.status_code == 201:
    PROJECT_ID = res.json()['id']
    print(f"Created new project: Food11 Low Confidence (ID {PROJECT_ID})")
else:
    raise Exception("Failed to create project:", res.text)
```
:::


::: {.cell .markdown}

Now, if we visit the Label Studio UI, we should see our "Food11 Low Confidence" project. However, it has no labeling tasks in it. We will create those via API as well.

:::

::: {.cell .markdown}

Let's authenticate to MinIO:

:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_USER']
MINIO_SECRET_KEY = os.environ['MINIO_PASSWORD']
BUCKET_NAME = "production"
```
:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
# note: we need to use the public IP of the MinIO service, not the hostname on the internal Docker network
# because we will use this S3 client to generate "pre-signed URLs" for images that we will label in Label Studio
# and these URLs must work in our own browser - outside of the Docker network
public_ip = requests.get("http://169.254.169.254/latest/meta-data/public-ipv4").text.strip()
s3 = boto3.client(
    "s3",
    endpoint_url=f"http://{public_ip}:9000",
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name="us-east-1"
)
```
:::

::: {.cell .markdown}

Now, we'll get a list of objects in the "production" bucket that are:

* recent (new since we last added labeling tasks, assuming this is a scheduled process)
* and have low confidence

:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
from datetime import datetime, timezone, timedelta

all_keys = []
recent_time_threshold = datetime.now(timezone.utc) - timedelta(hours=12)  
low_confidence_threshold = 0.7  # adjust threshold as needed so you get some samples!

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET_NAME):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        tags = s3.get_object_tagging(Bucket=BUCKET_NAME, Key=key)['TagSet']
        tag_dict = {t['Key']: t['Value'] for t in tags}
        timestamp_str = tag_dict.get("timestamp")
        predicted_class = tag_dict.get("predicted_class", "")
        confidence_str = tag_dict.get("confidence")
        if timestamp_str and confidence_str:
            ts = datetime.fromisoformat(timestamp_str)
            confidence = float(confidence_str)
            if ts > recent_time_threshold and confidence < low_confidence_threshold:
                all_keys.append({
                    "key": key,
                    "confidence": confidence_str,
                    "predicted_class": predicted_class
                })
```
:::

::: {.cell .markdown}

If you don't have any samples with "low confidence", adjust the threshold below until you have a couple:

:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
all_keys
```
:::

::: {.cell .markdown}

Depending on scale, we may label all of these, or a random sample of them. Here we will set up a task to label all.

Note that each "task" includes:

* the presigned URL for the image in the MinIO object store
* the predicted class according to our model
* and its confidence

and these will be visible to the human annotator.

:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
tasks = []
for item in all_keys:
    key = item["key"]
    confidence = item["confidence"]
    predicted_class = item["predicted_class"]

    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': key},
        ExpiresIn=3600
    )
    tasks.append({
        "data": {
            "image": presigned_url,
            "confidence": confidence,
            "predicted_class": predicted_class
        },
        "meta": {"original_key": key}
    })

res = requests.post(
    f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/import",
    json=tasks,
    headers=headers
)
if res.status_code == 201:
    print(f"Imported {len(tasks)} tasks into project {PROJECT_ID}")
else:
    raise Exception("Failed to import tasks:", res.text)
```
:::


::: {.cell .markdown}

In the Label Studio UI, validate that you can see the tasks in the "Food11 Low Confidence" project. The project overview will also now include "confidence" and "predicted class" columns, and you can sort and filter on these columns.

Complete the tasks in the "Food11 Low Confidence" project (i.e. label the images). Then, take a screenshot of the "Food11 Low Confidence" project dashboard, showing the list of images, the confidence of the model, and the first letters of your name next to each image in the "Annotated by" column.

:::
