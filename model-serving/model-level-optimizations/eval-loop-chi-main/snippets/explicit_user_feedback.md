

::: {.cell .markdown}

### Get explicit user feedback

In the previous sections, our own human annotators label production images. This allows us to evaluate and re-train our model on production data. However, it is not a very scalable approach. And, whether by random sampling or sampling from low-confidence predictions, we may miss cases where our model falls short.

To address this, in this section we will additionally create a mechanism by which the users of our service can explicitly signal whether or not the model's prediction is helpful.

Our modified Flask app, with support for explicit user feedback, is [in the feedback branch](https://github.com/teaching-on-testbeds/gourmetgram/tree/feedback) of the "gourmetgram" repository. 

For the modified GourmetGram application, we are going to return a flag icon along with the class label. 

```html
            flag_icon = f'''
                <form method="POST" action="/flag/{s3_key}" style="display:inline">
                    <button type="submit" class="btn btn-outline-warning btn-sm">🚩</button>
                </form>'''
            return f'<button type="button" class="btn btn-info btn-sm">{preds}</button> {flag_icon}'
```

Then, if the user clicks the flag icon, we will add a tag to the corresponding object (note that the key of the object to tag is passed to the function when the flag icon is clicked!):

```python
@app.route('/flag/<path:key>', methods=['POST'])
def flag_object(key):
    bucket = "production"
    current_tags = s3.get_object_tagging(Bucket=bucket, Key=key)['TagSet']
    tags = {t['Key']: t['Value'] for t in current_tags}

    if "flagged" not in tags:
        tags["flagged"] = "true"
        tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
        s3.put_object_tagging(Bucket=bucket, Key=key, Tagging={'TagSet': tag_set})
```

Let's try it now. Update the Docker compose file to switch the Flask application from the "production" branch to this new "feedback" branch:

```bash
# runs on node-eval-loop
nano eval-loop-chi/docker/docker-compose-production.yaml
```

and in the `flask` section, find

```
      context: https://github.com/teaching-on-testbeds/gourmetgram.git#production
```

and change it to 

```
      context: https://github.com/teaching-on-testbeds/gourmetgram.git#feedback
```

Use Ctrl+O and Enter to save, then Ctrl+X to exit `nano`. Then, rebuild the Flask app container image:

```bash
# runs on node-eval-loop
docker compose -f eval-loop-chi/docker/docker-compose-production.yaml build flask
```

and recreate the container 

```bash
# runs on node-eval-loop
docker compose -f eval-loop-chi/docker/docker-compose-production.yaml up -d
```

Try it now! In a browser, open

```
http://A.B.C.D:5000
```

substituting the floating IP assigned to your instance in place of `A.B.C.D`. Then, upload an image and make sure a label is returned. Note the flag icon next to the class label.

Upload at least ten images, and when a sample is misclassified, click the flag icon next to the class label to tag it.

Open the MinIO object store web UI -  in a browser, open

```
http://A.B.C.D:9001
```

substituting the floating IP assigned to your instance in place of `A.B.C.D`. Log in with `your-access-key` and password `your-secret-key`.

Using the Object Browser, find the images that you just submitted, and open the "Tags" view for one image that you had flagged. Verify that you can see the "flagged: true" tag. Take a screenshot of the browser window with the tags view, for later reference.


:::



::: {.cell .markdown}

Now, let's set up a Label Studio project and tasks for images that have been flagged - 

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
    "title": "Food11 User Flagged",
    "label_config": LABEL_CONFIG
}
res = requests.post(f"{LABEL_STUDIO_URL}/api/projects", json=project_config, headers=headers)
if res.status_code == 201:
    PROJECT_ID = res.json()['id']
    print(f"Created new project: Food11 User Flagged (ID {PROJECT_ID})")
else:
    raise Exception("Failed to create project:", res.text)
```
:::


::: {.cell .markdown}

Now, if we visit the Label Studio UI, we should see our "Food11 User Flagged" project. Next, we will create labeling tasks via API.

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
* and flagged

:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
from datetime import datetime, timezone, timedelta

all_keys = []
recent_time_threshold = datetime.now(timezone.utc) - timedelta(hours=12)

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET_NAME):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        tags = s3.get_object_tagging(Bucket=BUCKET_NAME, Key=key)['TagSet']
        tag_dict = {t['Key']: t['Value'] for t in tags}
        
        timestamp_str = tag_dict.get("timestamp")
        flagged = tag_dict.get("flagged") == "true"
        
        if timestamp_str and flagged:
            ts = datetime.fromisoformat(timestamp_str)
            if ts > recent_time_threshold:
                all_keys.append({
                    "key": key,
                    "confidence": tag_dict.get("confidence", ""),
                    "predicted_class": tag_dict.get("predicted_class", ""),
                    "flagged": tag_dict.get("flagged", "false")
                })
```
:::


::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
all_keys
```
:::

::: {.cell .markdown}

We will set up tasks to label each of these flagged images:

:::

::: {.cell .code}
```python
# runs inside Jupyter container on node-eval-loop
tasks = []
for item in all_keys:
    key = item["key"]
    confidence = item["confidence"]
    predicted_class = item["predicted_class"]
    flagged = item["flagged"]

    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': key},
        ExpiresIn=3600
    )
    tasks.append({
        "data": {
            "image": presigned_url,
            "confidence": confidence,
            "predicted_class": predicted_class,
            "flagged": flagged
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

In the Label Studio UI, validate that you can see the tasks in the "Food11 User Flagged" project. 

Complete the tasks in the "Food11 User Flagged" project (i.e. label the images). Then, take a screenshot of the "Food11 User Flagged" project dashboard, showing the list of images, the confidence of the model, and the first letters of your name next to each image in the "Annotated by" column.

:::



::: {.cell .markdown}

### Get user labels

We can further improve on this - instead of asking users to *flag* when a label is incorrect, we can allow them to change the label themselves.


Our modified Flask app, with support for explicit user feedback, is [in the userlabel branch](https://github.com/teaching-on-testbeds/gourmetgram/tree/userlabel) of the "gourmetgram" repository. 

For the modified GourmetGram application, instead of returning the class label in a button, we will return it in a form which allows the user to select another class label:

```html
            class_list = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
              "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]
            select_html = f'''
                <form method="POST" action="/correct-label/{s3_key}">
                    <select name="corrected_class" onchange="this.form.submit()" class="form-select form-select-sm" style="width: auto; display: inline-block;">
                    {''.join([f'<option value="{cls}" {"selected" if cls == preds else ""}>{cls}</option>' for cls in class_list])}
                </select>
                </form>
                '''
```


Then, if the user chanegs the label, we will add a tag to the corresponding object:


```python
@app.route('/correct-label/<path:key>', methods=['POST'])
def correct_label(key):
    new_label = request.form.get('corrected_class')
    current_tags = s3.get_object_tagging(Bucket='production', Key=key)['TagSet']
    tags = {t['Key']: t['Value'] for t in current_tags}
    tags['corrected_class'] = new_label
    tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
    s3.put_object_tagging(Bucket='production', Key=key, Tagging={'TagSet': tag_set})
    return '', 204
```

Let's try it now. Update the Docker compose file to switch the Flask application from the "feedback" branch to this new "userlabel" branch:

```bash
# runs on node-eval-loop
nano eval-loop-chi/docker/docker-compose-production.yaml
```

and in the `flask` section, find

```
      context: https://github.com/teaching-on-testbeds/gourmetgram.git#feedback
```

and change it to 

```
      context: https://github.com/teaching-on-testbeds/gourmetgram.git#userlabel
```

Use Ctrl+O and Enter to save, then Ctrl+X to exit `nano`. Then, rebuild the Flask app container image:

```bash
# runs on node-eval-loop
docker compose -f eval-loop-chi/docker/docker-compose-production.yaml build flask
```

and recreate the container 

```bash
# runs on node-eval-loop
docker compose -f eval-loop-chi/docker/docker-compose-production.yaml up -d
```

Try it now! In a browser, open

```
http://A.B.C.D:5000
```

substituting the floating IP assigned to your instance in place of `A.B.C.D`. Then, upload an image and make sure a label is returned. 

Use the form to change the class label for a misclassified image. 

Open the MinIO object store web UI -  in a browser, open

```
http://A.B.C.D:9001
```

substituting the floating IP assigned to your instance in place of `A.B.C.D`. Log in with `your-access-key` and password `your-secret-key`.

Using the Object Browser, find the images that you just submitted, and open the "Tags" view for one image that you had flagged. Verify that you can see the "corrected_class" tag. Take a screenshot of the browser window with the tags view, for later reference.


:::
