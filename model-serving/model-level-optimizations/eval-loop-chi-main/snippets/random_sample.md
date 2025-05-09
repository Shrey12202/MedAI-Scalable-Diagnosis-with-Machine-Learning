


::: {.cell .markdown}

### Label random sample of production images

We had previously configured Label Studio so that human annotators would be asked to label *all* images in the production bucket. (Any time we "Sync storage" in Label Studio, new images in the production bucket are added as tasks in Label Studio.)

Of course, for a large scale production service, this is impractical.

Let's set up a new project in Label Studio, in which only a small (random) sample of production images are selected for labeling. 

:::


::: {.cell .markdown}

This time, we will use the Label Studio API to automate the setup of the new project and tasks. 

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
</View>
"""
```
:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}

# configure a project - set up its name and the appearance of the labeling interface
project_config = {
    "title": "Food11 Random Sample",
    "label_config": LABEL_CONFIG
}

# send it to Label Studio API
res = requests.post(f"{LABEL_STUDIO_URL}/api/projects", json=project_config, headers=headers)
if res.status_code == 201:
    PROJECT_ID = res.json()['id']
    print(f"Created new project: Food11 Random Sample (ID {PROJECT_ID})")
else:
    raise Exception("Failed to create project:", res.text)
```
:::


::: {.cell .markdown}

Now, if we visit the Label Studio UI, we should see our "Food11 Random Sample" project. However, it has no labeling tasks in it. We can create those via API as well.

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
SAMPLE_SIZE = 3  # Number of images to sample
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

get a list of objects in the "production" bucket, and randomly sample some:

:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
all_keys = []
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET_NAME):
    for obj in page.get("Contents", []):
        all_keys.append(obj["Key"])

sampled_keys = random.sample(all_keys, min(SAMPLE_SIZE, len(all_keys)))
```
:::

::: {.cell .markdown}

and then, send those as tasks to Label Studio:

:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
# generate a URL for each object we want to label, so that the annotator can view the image from their browser
tasks = []
for key in sampled_keys:
    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': key},
        ExpiresIn=3600
    )
    # and add to the list of tasks
    tasks.append({"data": {"image": presigned_url}, "meta": {"original_key": key}})

# then, send the lists of tasks to the Label Studio project
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


Now, we should see these tasks in the Label Studio UI, under the "Food11 Random Sample" project.


Complete the tasks in the "Food11 Random Sample" project (i.e. label the images). Then, take a screenshot of the "Food11 Random Sample" project dashboard, showing the list of images and the first letters of your name next to each image in the "Annotated by" column.

:::


::: {.cell .markdown}

This random sampling process could be automated, e.g. on a schedule as part of a continuous monitoring and re-training pipeline. 

Although we won't do it right now, it would also be reasonable to re-organize the data based on the new labels after annotation - if an image was originally placed in the "class_01" directory but the human label is class 3, it could be moved automatically to "class_03" to facilitate re-training on the production data.

:::

::: {.cell .markdown}

If we did automate this process, though, we would want to make sure to only sample from new production images that were not available as of the last random draw, so we might filter on the "timestamp" key first, like this:

:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
from datetime import datetime, timezone, timedelta

all_keys = []
recent_time_threshold = datetime.now(timezone.utc) - timedelta(hours=12) # try changing this to see the effect!

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET_NAME):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        tags = s3.get_object_tagging(Bucket=BUCKET_NAME, Key=key)['TagSet']
        tag_dict = {t['Key']: t['Value'] for t in tags}
        timestamp_str = tag_dict.get("timestamp")
        if timestamp_str:
            ts = datetime.fromisoformat(timestamp_str)
            if ts > recent_time_threshold:
                all_keys.append(key)
```
:::

::: {.cell .markdown}

The approach above - filtering by timestamp tag - works at a small to moderate scale. At a larger scale, though, it would be impractical because getting the list of tags to sample from requires many API calls to get individual object tags. To scale this up, we might:

* move data between a "raw production data" bucket and a "processed production data bucket" on a schedule, so that you only need to draw samples from a smaller bucket
* and/or save metadata about production samples externally (e.g. in a database or a [table](https://iceberg.apache.org/)), so we can query metadata more efficiently

:::

