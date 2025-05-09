
::: {.cell .markdown}

### Evaluate accuracy on production data

We are going to:

* connect to Label Studio and retrieve the details of all the "tasks" associated with our Food11 project
* connect to MinIO, and get the predicted class (from the tag!) of every object in the "production" bucket

and compare those, to evaluate the accuracy of our system on "production" data.

:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
import requests
import boto3 
from urllib.parse import urlparse
from collections import defaultdict, Counter
import os
```
:::

::: {.cell .markdown}

First, we need to get the details we will need to authenticate to MinIO and to Label Studio. We passed these as environment variables to the Jupyter container:

:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
LABEL_STUDIO_URL = os.environ['LABEL_STUDIO_URL']
LABEL_STUDIO_TOKEN = os.environ['LABEL_STUDIO_USER_TOKEN']
PROJECT_ID = 1  # use the first project set up in Label Studio

MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_USER']
MINIO_SECRET_KEY = os.environ['MINIO_PASSWORD']
BUCKET_NAME = "production"
```
:::


::: {.cell .markdown}

Now, we can authenticate to MinIO:

:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_URL,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name='us-east-1'
)
```
:::


::: {.cell .markdown}

And, we can authenticate to LabelStudio and get the details of all the "tasks". (Each image that requires human annotation is one task.)

:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
response = requests.get(
    f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/export?exportType=JSON",
    headers={"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
)

tasks = response.json()
```
:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
tasks
```
:::

::: {.cell .markdown}

Now, we can compute the accuracy of our model on the production data:

:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
total, correct = 0, 0

for task in tasks:
    # get human annotator's from Label Studio
    human_label = task['annotations'][0]['result'][0]['value']['choices'][0]
    key = urlparse(task['data']['image']).path.lstrip("/")
    key = key[len(f"{BUCKET_NAME}/"):] if key.startswith(f"{BUCKET_NAME}/") else key

    # get label assigned by model to that SAME IMAGE, from the object tag in MinIO
    tags = s3.get_object_tagging(Bucket=BUCKET_NAME, Key=key)['TagSet']
    model_label = {t['Key']: t['Value'] for t in tags}.get('predicted_class')

    if model_label and human_label:
        total += 1
        correct += int(model_label == human_label)
```
:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
print(f"Accuracy: {correct}/{total} = {correct / total:.2%}" if total else "No valid comparisons made.")
```
:::


::: {.cell .markdown}

In this example, we have computed simple accuracy over a static set of production images, as a demo. However, this could be integrated into a broader evaluation plan (including e.g. evaluation on different metrics, on specific slices of interest) and a broader monitoring plan (e.g. use the timestamp tag to monitor prediction accuracy over time using time windows.)

Similarly, our labeled production data can be used as part of a continuous training plan - after evaluating our model on the labeled production data, we can use it as part of the training set the next time we re-train our model.

:::

