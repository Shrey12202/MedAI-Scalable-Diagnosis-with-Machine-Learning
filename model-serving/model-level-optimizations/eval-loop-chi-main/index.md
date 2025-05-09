

# Evaluation of ML systems by closing the feedback loop

In this tutorial, we will practice selected techniques for evaluating machine learning systems, and then monitoring them in production.

The lifecycle of a model may look something like this:

* **Training**: Initially, a model is trained on some training data
* **Testing** (offline): If training completes successfully, the model progresses to a testing - offline evaluation - stage. In this stage, it is evaluated using a held-out evaluation set not used in training, and potentially other special evaluation sets (as we'll see in this tutorial).
* **Staging**: Given satisfactory performance on the offline evaluation, the model may be *packaged* as part of a service, and then this package promoted to a staging environment that mimics the "production" service but without live users. In this staging environmenmt, we can perform integration tests against the service and also load tests to evaluate the inference performance of the system.
* **Canary** (or blue/green, or other "preliminary" live environment): From the staging environment, the service can be promoted to a canary or other preliminary environment, where it gets requests from a small fraction of live users. In this environment, we are closely monitoring the service, its predictions, and the infrastructure for any signs of problems. We will try to "close the feedback loop" so that we can evaluate how effective our model is on production data, and potentially, evaluate the system on business metrics.
* **Production**: Finally, after a thorough offline and online evaluation, we may promote the model to the live production environment, where it serves most users. We will continue monitoring the system for signs of degradation or poor performance.

In this particular section, we will practice evaluation and monitoring in the *online* stage - when a system is serving some or all real users - and specifically the part where we "close the feedback loop" in order to evaluate how well our system performs on production data.

![This tutorial focuses on the online testing stage.](images/stages-online.svg)

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You should also have added your SSH key to the KVM@TACC site.



## Experiment resources 

For this experiment, we will provision one virtual machine on KVM@TACC.

Our initial online system, with monitoring of the live service, will include the following components:

* a FastAPI endpoint for our model
* a Flask app that sends requests to our FastAPI endpoint

These comprise the operational system we want to evaluate and monitor! To this, we'll add:

* MinIO object store, to save "production" data - images that are submitted by "real" users - and other artifacts
* Label Studio, an open source labeling tool used by human annotators to label data for ML training
* and Airflow, which we'll use to orchestrate a continuous monitoring and re-training workflow after we have "closed the loop"

We will also host a Jupyter container for interacting with the "production" data.




## Open this experiment on Trovi


When you are ready to begin, you will continue with the next step, in which you bring up and configure a VM instance! To begin this step, open this experiment on Trovi:

* Use this link: [Evaluation of ML systems by closing the feedback loop](https://chameleoncloud.org/experiment/share/285f3758-3df2-4226-99ab-c243aa715b8e) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it, including the notebok to bring up the VM instance.






## Launch and set up a VM instance- with python-chi

We will use the `python-chi` Python API to Chameleon to provision our VM server. 

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. 


```python
from chi import server, context
import chi, os, time, datetime

context.version = "1.0" 
context.choose_project()
context.choose_site(default="KVM@TACC")
```


We will bring up a `m1.medium` flavor VM instance with the `CC-Ubuntu24.04` disk image. 

> **Note**: the following cell brings up a server only if you don't already have one with the same name! (Regardless of its error state.) If you have a server in ERROR state already, delete it first in the Horizon GUI before you run this cell.



```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-eval-loop-{username}", 
    image_name="CC-Ubuntu24.04",
    flavor_name="m1.medium"
)
s.submit(idempotent=True)
```



Then, we'll associate a floating IP with the instance:


```python
s.associate_floating_ip()
```

```python
s.refresh()
s.check_connectivity()
```


In the output below, make a note of the floating IP that has been assigned to your instance (in the "Addresses" row).


```python
s.refresh()
s.show(type="widget")
```


By default, all connections to VM resources are blocked, as a security measure.  We need to attach one or more "security groups" to our VM resource, to permit access over the Internet to specified ports.

The following security groups will be created (if they do not already exist in our project) and then added to our server:



```python
security_groups = [
  {'name': "allow-ssh", 'port': 22, 'description': "Enable SSH traffic on TCP port 22"},
  {'name': "allow-5000", 'port': 5000, 'description': "Enable TCP port 5000 (used by Flask)"},
  {'name': "allow-8000", 'port': 8000, 'description': "Enable TCP port 8000 (used by FastAPI)"},
  {'name': "allow-8888", 'port': 8888, 'description': "Enable TCP port 8888 (used by Jupyter)"},
  {'name': "allow-9000", 'port': 9000, 'description': "Enable TCP port 9000 (used by MinIO API)"},
  {'name': "allow-9001", 'port': 9001, 'description': "Enable TCP port 9001 (used by MinIO Web UI)"},
  {'name': "allow-8080", 'port': 8080, 'description': "Enable TCP port 8080 (used by cAdvisor, Label Studio, Airflow)"},
  {'name': "allow-8081", 'port': 8081, 'description': "Enable TCP port 8081 (alt for 8080)"}

]
```


```python
# configure openstacksdk for actions unsupported by python-chi
os_conn = chi.clients.connection()
nova_server = chi.nova().servers.get(s.id)

for sg in security_groups:

  if not os_conn.get_security_group(sg['name']):
      os_conn.create_security_group(sg['name'], sg['description'])
      os_conn.create_security_group_rule(sg['name'], port_range_min=sg['port'], port_range_max=sg['port'], protocol='tcp', remote_ip_prefix='0.0.0.0/0')

  nova_server.add_security_group(sg['name'])

print(f"updated security groups: {[group.name for group in nova_server.list_security_group()]}")
```







### Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.


```python
s.execute("git clone https://github.com/teaching-on-testbeds/eval-loop-chi")
```



### Set up Docker

Here, we will set up the container framework.


```python
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```



## Open an SSH session

Finally, open an SSH sesson on your server. From your local terminal, run

```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where

* in place of `~/.ssh/id_rsa_chameleon`, substitute the path to your own key that you had uploaded to KVM@TACC
* in place of `A.B.C.D`, use the floating IP address you just associated to your instance.





## Save production data

Our first step in making sure that we "close the feedback loop" is to save the data that is submitted to our service in production, so that we can later evaluate the performance of our model on "production" data.




### Bring up services

Inside the SSH session, we'll bring up the Flask, FastAPI, & MinIO services. 

First, we are going to create a shared network, so that future services defined in other Docker compose files will also be able to access these services using container names - 


```bash
# runs on node-eval-loop
docker network create production_net
```

Then, bring up the services:

```bash
# runs on node-eval-loop
docker compose -f eval-loop-chi/docker/docker-compose-production.yaml up -d
```




### Create a bucket


Open the MinIO object store web UI -  in a browser, open

```
http://A.B.C.D:9001
```

substituting the floating IP assigned to your instance in place of `A.B.C.D`. Log in with `your-access-key` and password `your-secret-key`.

In the menu sidebar, click "Buckets". Note that there is already a bucket named "production". We created this bucket using a "sidecar" container in our Docker compose file - a container whose entire role was to create the bucket, then stop:

```
  minio-init:
    image: minio/mc
    container_name: minio_init
    depends_on:
      - minio
    restart: "no"
    entrypoint: >
      /bin/sh -c "
      sleep 5 &&
      mc alias set myminio http://minio:9000 your-access-key your-secret-key &&
      mc mb -p myminio/production || echo 'Bucket already exists'
      "
    networks:
      - production_net
```



### Modify service to send data to production bucket

Then, we need to modify our service to send data to the production bucket!

Our modified Flask app, with data sent to MinIO, is [in the production branch](https://github.com/teaching-on-testbeds/gourmetgram/tree/production) of the "gourmetgram" repository. 

For the modified GourmetGram application, we:

* specified `MINIO_URL`, `MINIO_USER`, `MINIO_PASSWORD` environment variables in the Docker compose file. These will be used to authenticate to the object store.
* added `boto3` to `requirements.txt` - this is a Python client for S3-compatible object store services, including MinIO.
* added imports to `app.py`:

```python
from mimetypes import guess_type # used to identify the type of image
from datetime import datetime # used to generate timestamp tag for image
import uuid # used to generate unique ID per image
import boto3 # client for s3-compatible object store, including MinIO
from concurrent.futures import ThreadPoolExecutor  # used for the thread pool that will upload images to MinIO
executor = ThreadPoolExecutor(max_workers=2)  # can adjust max_workers as needed
```

* added this near the beginning of `app.py`, to connect to the object store:

```python
# New! Authenticate to MinIO object store
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['MINIO_URL'],  # e.g. 'http://minio:9000'
    aws_access_key_id=os.environ['MINIO_USER'],
    aws_secret_access_key=os.environ['MINIO_PASSWORD'],
    region_name='us-east-1'  # required for the boto client but not used by MinIO
)
```

* added this function to `app.py`:

```python
# New! for uploading production images to MinIO bucket
def upload_production_bucket(img_path, preds, confidence, prediction_id):
    classes = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
	    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
	    "Vegetable/Fruit"])
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    pred_index = np.where(classes == preds)[0][0]
    class_dir = f"class_{pred_index:02d}"

    bucket_name = "production"
    root, ext = os.path.splitext(img_path)
    content_type = guess_type(img_path)[0] or 'application/octet-stream'
    s3_key = f"{class_dir}/{prediction_id}{ext}"
    
    with open(img_path, 'rb') as f:
        s3.upload_fileobj(f, 
            bucket_name, 
            s3_key, 
            ExtraArgs={'ContentType': content_type}
            )

    # tag the object with predicted class and confidence
    s3.put_object_tagging(
        Bucket=bucket_name,
        Key=s3_key,
        Tagging={
            'TagSet': [
                {'Key': 'predicted_class', 'Value': preds},
                {'Key': 'confidence', 'Value': f"{confidence:.3f}"},
                {'Key': 'timestamp', 'Value': timestamp}
            ]
        }
    )
```

* and finally, when a prediction is ready, we call it (asynchronously, so the user does not have to wait for it to return):

```python
# create a unique ID for the prediction - used in filename    
prediction_id = str(uuid.uuid4())
executor.submit(upload_production_bucket, img_path, preds, probs, prediction_id)
```



Note that in addition to uploading the image, we also tag the object with

* its predicted class label 
* the model's confidence in its prediction
* and the timestamp

This will help us organize the production data for later use.




Try out your GourmetGram service, and make sure it is running. In a browser, open

```
http://A.B.C.D:5000
```

substituting the floating IP assigned to your instance in place of `A.B.C.D`. Then, upload an image and make sure a label is returned.

Wait a few moments. Then, in the MinIO web UI, check the "production" bucket and make sure that your image has been uploaded. (It will have been placed in a folder corresponding to the class label is was assigned.)

Using the MinIO object browser:

* find the uploaded image
* click on "Preview" to see the image
* click on "Tags" to see the uploaded tags

Upload a few more images, from different classes, until you have at least 10 unique food images in the "production" bucket.






We are now saving data from our production service, including the image submitted by the user, the label assigned by our model, and the confidence of our model - but, that's not enough to tell us about performance in production, or help us improve it! We need some way to 

* evaluate whether or not our model is doing a good job assigning labels to production data
* and use production data to re-train our model using supervised learning (which requires labels)

In the next section, we'll explore some strategies for this.



## Practice "closing the feedback loop"

When there are no natural ground truth labels, we need to explicitly "close the feedback loop":

* in order to evaluate how well our model does in production, versus in offline evaluation on a held-out test set,
* and also to get new "production data" on which to re-train the model when its performance degrades.

For example, with this food type classifier, once it is deployed to "real" users:


* We could get human annotators to label production data. 
* We could set aside samples where the model has low confidence in its prediction, for a human to label. These extra-difficult samples are especially useful for re-training.
* We could allow users to give explicit feedback about whether the label assigned to their image is correct or not. This feedback may be sparse (some users won't bother giving feedback even if the label is wrong) and noisy (some users may give incorrect feedback). We can get human annotators to label this data, too.
* We could allow users to explicitly label their images, by changing the label that is assigned by the classifier. This feedback may be sparse (some users won't bother giving feedback even if the label is wrong) and noisy (some users may give incorrect feedback).

We're going to try out all of these options! 



### Start Label Studio

First, let's start Label Studio, our tool for managing the tasks assigned to human annotators. These humans will assign ground truth labels to production data - in this case, images that have been submitted by "real" users to our service - so that we can monitor the performance of our model in production. 


Run

```bash
# runs on node-eval-loop
docker compose -f eval-loop-chi/docker/docker-compose-labelstudio.yaml up -d
```

Note that Label Studio is now running *in addition to* the Flask, FastAPI, and MinIO services we started in the previous section.



### Label production images in Label Studio

In our initial implementation, we will use LabelStudio to label all of the "production" images. 

In a browser, open

```
http://A.B.C.D:8080
```

substituting the floating IP assigned to your instance in place of `A.B.C.D`. Log in with username `labelstudio@example.com` and password `labelstudio` (we have set these in our Docker compose file).

You should see a user interface that invites you to create a project. 

Click on the user icon in the top right, and choose "Account & Settings". In the "Personal Info" section, fill in your real first and last name, and save the changes.

Now, we are going to create a project in Label Studio! From the "Home" page, click "Create Project". Name it "Food11 Production", and for the description, use: 

> Review and correct food images submitted to production service.

Then, we'll set up the labeling interface. Click on the "Labeling setup" tab. From the "Computer Vision" section, choose "Image classification". 

Edit the template: in the "Choices" section, click "X" next to each of the pre-existing choices. Then, where it says "Add choices", paste the class labels:

```
Bread
Dairy product
Dessert
Egg
Fried food
Meat
Noodles/Pasta
Rice
Seafood
Soup
Vegetable/Fruit
```

and click "Add".

In the UI Preview area, you can see what the interface for the human annotators will look like. The long list of class labels is not very usable. To fix it, toggle from "Visual" to "Code" setting on the left side panel. Find the line

```html
  <Choices name="choice" toName="image" >
```

and change it to 

```html
  <Choices name="choice" toName="image"  showInLine="true" >
```

and verify that the UI preview looks better. 

Also change

```html
  <Image name="image" value="$image"/>
```

to 

```html
  <Image name="image" value="$image" maxWidth="500px"/>
```

If everything seems OK, click "Save".

Next, we need to configure other project details. From inside the project, click on the "Settings" button. 

Then, in the "Cloud Storage" section, click on "Add Source Storage". Fill in the details as follows (leave any that are unspecified blank):

* Storage type: AWS S3 (MinIO is an S3-compatible object store service)
* Storage title: MinIO
* Bucket name: production
* S3 endpoint: http://A.B.C.D:9000 (**substitute the floating IP address assigned to your instance**)
* Access key ID: your-access-key
* Secret access key: your-secret-key
* Treat every bucket object as a source file: checked (so that each object in the bucket is interpreted as an image to classify)
* Recursive scan: checked (so that it will look inside all of the class-specific directories)

Click "Check connection", then, if it is successful, "Add storage".

Then, click "Sync storage" and look for a "Completed" message.

Now, when you click on the project in the Label Studio interface, you will see a list of images to label! Use the Web UI to label the images. Then, take a screenshot of the project dashboard, showing the list of images and the first letters of your name next to each image in the "Annotated by" column.



Now that we have ground truth labels for the "production" data, we can evaluate the performance of our model on this production data.

We'll do this interactively inside a Jupyter notebook. Run

```bash
# runs on node-eval-loop
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of `127.0.0.1`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface.

In the file browser on the left side, open the `work` directory.



### Evaluate accuracy on production data

We are going to:

* connect to Label Studio and retrieve the details of all the "tasks" associated with our Food11 project
* connect to MinIO, and get the predicted class (from the tag!) of every object in the "production" bucket

and compare those, to evaluate the accuracy of our system on "production" data.



```python
# runs inside Jupyter container on node-eval-loop
import requests
import boto3 
from urllib.parse import urlparse
from collections import defaultdict, Counter
import os
```


First, we need to get the details we will need to authenticate to MinIO and to Label Studio. We passed these as environment variables to the Jupyter container:



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



Now, we can authenticate to MinIO:



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



And, we can authenticate to LabelStudio and get the details of all the "tasks". (Each image that requires human annotation is one task.)



```python
# runs inside Jupyter container on node-eval-loop
response = requests.get(
    f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/export?exportType=JSON",
    headers={"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
)

tasks = response.json()
```


```python
# runs inside Jupyter container on node-eval-loop
tasks
```


Now, we can compute the accuracy of our model on the production data:



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

```python
# runs inside Jupyter container on node-eval-loop
print(f"Accuracy: {correct}/{total} = {correct / total:.2%}" if total else "No valid comparisons made.")
```



In this example, we have computed simple accuracy over a static set of production images, as a demo. However, this could be integrated into a broader evaluation plan (including e.g. evaluation on different metrics, on specific slices of interest) and a broader monitoring plan (e.g. use the timestamp tag to monitor prediction accuracy over time using time windows.)

Similarly, our labeled production data can be used as part of a continuous training plan - after evaluating our model on the labeled production data, we can use it as part of the training set the next time we re-train our model.






### Label random sample of production images

We had previously configured Label Studio so that human annotators would be asked to label *all* images in the production bucket. (Any time we "Sync storage" in Label Studio, new images in the production bucket are added as tasks in Label Studio.)

Of course, for a large scale production service, this is impractical.

Let's set up a new project in Label Studio, in which only a small (random) sample of production images are selected for labeling. 




This time, we will use the Label Studio API to automate the setup of the new project and tasks. 


```python
# runs inside Jupyter container on node-eval-loop
import requests
import boto3 
import os
import random
```

```python
# runs inside Jupyter container on node-eval-loop
LABEL_STUDIO_URL = os.environ['LABEL_STUDIO_URL']
LABEL_STUDIO_TOKEN = os.environ['LABEL_STUDIO_USER_TOKEN']
```

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



Now, if we visit the Label Studio UI, we should see our "Food11 Random Sample" project. However, it has no labeling tasks in it. We can create those via API as well.



Let's authenticate to MinIO:


```python
# runs inside Jupyter container on node-eval-loop
MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_USER']
MINIO_SECRET_KEY = os.environ['MINIO_PASSWORD']
BUCKET_NAME = "production"
SAMPLE_SIZE = 3  # Number of images to sample
```

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


get a list of objects in the "production" bucket, and randomly sample some:


```python
# runs inside Jupyter container on node-eval-loop
all_keys = []
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET_NAME):
    for obj in page.get("Contents", []):
        all_keys.append(obj["Key"])

sampled_keys = random.sample(all_keys, min(SAMPLE_SIZE, len(all_keys)))
```


and then, send those as tasks to Label Studio:


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




Now, we should see these tasks in the Label Studio UI, under the "Food11 Random Sample" project.


Complete the tasks in the "Food11 Random Sample" project (i.e. label the images). Then, take a screenshot of the "Food11 Random Sample" project dashboard, showing the list of images and the first letters of your name next to each image in the "Annotated by" column.




This random sampling process could be automated, e.g. on a schedule as part of a continuous monitoring and re-training pipeline. 

Although we won't do it right now, it would also be reasonable to re-organize the data based on the new labels after annotation - if an image was originally placed in the "class_01" directory but the human label is class 3, it could be moved automatically to "class_03" to facilitate re-training on the production data.



If we did automate this process, though, we would want to make sure to only sample from new production images that were not available as of the last random draw, so we might filter on the "timestamp" key first, like this:


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


The approach above - filtering by timestamp tag - works at a small to moderate scale. At a larger scale, though, it would be impractical because getting the list of tags to sample from requires many API calls to get individual object tags. To scale this up, we might:

* move data between a "raw production data" bucket and a "processed production data bucket" on a schedule, so that you only need to draw samples from a smaller bucket
* and/or save metadata about production samples externally (e.g. in a database or a [table](https://iceberg.apache.org/)), so we can query metadata more efficiently






### Label low-confidence production images

In the previous section, we used uniform random sampling to select production images for human annotation. In practice, however, we may want to preferentially label samples for which the model has low confidence. Combined with random sampling, this is a powerful strategy because:

* We can create a high-quality but small evaluation set using randomly sampled production images
* and in parallel, create a lower quality but large-volume re-training set, using model-labeled production images for which the confidence is high (assuming the model labeled them correctly) along with human-labeled production images for which the confidence is low




We will use the Label Studio API to automate the setup of a new project and tasks. 


```python
# runs inside Jupyter container on node-eval-loop
import requests
import boto3 
import os
import random
```

```python
# runs inside Jupyter container on node-eval-loop
LABEL_STUDIO_URL = os.environ['LABEL_STUDIO_URL']
LABEL_STUDIO_TOKEN = os.environ['LABEL_STUDIO_USER_TOKEN']
```


For this project, our labeling UI will be slightly different - we are going to also display the model's predicted class and confidence in its prediction:




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



Now, if we visit the Label Studio UI, we should see our "Food11 Low Confidence" project. However, it has no labeling tasks in it. We will create those via API as well.



Let's authenticate to MinIO:


```python
# runs inside Jupyter container on node-eval-loop
MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_USER']
MINIO_SECRET_KEY = os.environ['MINIO_PASSWORD']
BUCKET_NAME = "production"
```

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


Now, we'll get a list of objects in the "production" bucket that are:

* recent (new since we last added labeling tasks, assuming this is a scheduled process)
* and have low confidence


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


If you don't have any samples with "low confidence", adjust the threshold below until you have a couple:



```python
# runs inside Jupyter container on node-eval-loop
all_keys
```


Depending on scale, we may label all of these, or a random sample of them. Here we will set up a task to label all.

Note that each "task" includes:

* the presigned URL for the image in the MinIO object store
* the predicted class according to our model
* and its confidence

and these will be visible to the human annotator.


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



In the Label Studio UI, validate that you can see the tasks in the "Food11 Low Confidence" project. The project overview will also now include "confidence" and "predicted class" columns, and you can sort and filter on these columns.

Complete the tasks in the "Food11 Low Confidence" project (i.e. label the images). Then, take a screenshot of the "Food11 Low Confidence" project dashboard, showing the list of images, the confidence of the model, and the first letters of your name next to each image in the "Annotated by" column.




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






Now, let's set up a Label Studio project and tasks for images that have been flagged - 


```python
# runs inside Jupyter container on node-eval-loop
import requests
import boto3 
import os
import random
```

```python
# runs inside Jupyter container on node-eval-loop
LABEL_STUDIO_URL = os.environ['LABEL_STUDIO_URL']
LABEL_STUDIO_TOKEN = os.environ['LABEL_STUDIO_USER_TOKEN']
```


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



Now, if we visit the Label Studio UI, we should see our "Food11 User Flagged" project. Next, we will create labeling tasks via API.



Let's authenticate to MinIO:


```python
# runs inside Jupyter container on node-eval-loop
MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_USER']
MINIO_SECRET_KEY = os.environ['MINIO_PASSWORD']
BUCKET_NAME = "production"
```

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


Now, we'll get a list of objects in the "production" bucket that are:

* recent (new since we last added labeling tasks, assuming this is a scheduled process)
* and flagged


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


```python
# runs inside Jupyter container on node-eval-loop
all_keys
```


We will set up tasks to label each of these flagged images:


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



In the Label Studio UI, validate that you can see the tasks in the "Food11 User Flagged" project. 

Complete the tasks in the "Food11 User Flagged" project (i.e. label the images). Then, take a screenshot of the "Food11 User Flagged" project dashboard, showing the list of images, the confidence of the model, and the first letters of your name next to each image in the "Annotated by" column.





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




### (Optional) Use in a continuous monitoring and re-training pipeline 

Finally, let's take a quick look at how we would finish "closing the loop" by using the labeled data in a continuous monitoring and re-training pipeline. We will want to automate the process of:

* creating tasks in Label Studio out of data from production
* evaluating performance on labeled data sampled from production
* and re-training on data from production

There are a variety of ways in which we can realize this goal. We will use Apache Airflow, a workflow orchestrator, to manage this pipeline on a schedule. (Airflow is a good fit for a Docker environment; in a Kubernetes environment, we might prefer to use Argo Events + Argo Workflow.)




First, let's bring up Airflow:

```bash
# runs on node-eval-loop
docker compose -f eval-loop-chi/docker/docker-compose-airflow.yaml up -d
```

When it comes up, a web UI will be on port 8081. (Airflow runs a web server on port 8080 by default, but since we already have Label Studio on port 8080, we used the Docker compose to map it to port 8081 instead.)


In a browser, open

```
http://A.B.C.D:8081
```

substituting the floating IP assigned to your instance in place of `A.B.C.D`. Log in with username `airflow@example.com` and password `airflow` (we have created an initial user with these credentials in our Docker compose file).




Airflow is a workflow orchestrator for running any pipeline that represented as a DAG - directed acyclic graph. Here's an example of basic DAG for Airflow:

```python
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

```

This Python script defines a DAG called "test_dag_hello_world" 

* that starts one day ago (e.g. it is allowed to run as of one day ago; it was not allowed to run before that date)
* that is scheduled to run daily
* and doesn't "catch up" on past runs, e.g. if I set the start date to one year ago, it wouldn't run 365 times to make up for the missing runs!)

The actual DAG just has one "node", `t1`, and that runs Airflow's built-in `EmptyOperator` which acts as a no-op placeholder task. 

When we place a Python file in the Airflow DAGs folder, Airflow scans that file at regular intervals to discover DAG definitions. If a file has top-level variables that are instances of the DAG class, Airflow adds it to its metadata database and displays it in the web UI.

In the Airflow web UI, the `test_dag_hello_world` DAG should be visible in DAGs tab. It runs on a schedule, but we can also trigger it manually - press the ▶ button to trigger it now. Confirm that it runs sucessfully.




Now, let's run a "real" pipeline. Our first pipeline will:

* Get objects from the "production" bucket that have been uploaded in the intervening interval since the last DAG run.
* Sample from them to get tasks to send to Label Studio, including: a random sample of all images, the low confidence images, the flagged images, and images that have been re-labeled by the user. 
* Move these to a "production-label-wait" bucket, then generate tasks for Label Studio. 
* and, move the remaining (not selected for labeling) high=confidence images from the list to a "production-noisy" bucket.

The "production-noisy" bucket will be used for model re-training. We consider it "noisy" because its labels are generated by the model itself, not by human; but we will add human-labeled data to it when available, in the next pipeline.

This DAG will take advantage of Airflow's built in "data interval" idea, which lets the DAG know what time interval of data it is responsible for processing according to its schedule:

```
start = context['data_interval_start']
end = context['data_interval_end']
```

although if we trigger it manually from the web UI, that won't apply, so then we would just use a recent half-hour window.

The DAG will also include a task that initializes the "production-label-wait" and "production-noisy" buckets if they do not yet exist, and it will create a Label Studio project for "Food11 Continuous X" if it does not yet exist.

Click through to this DAG, and look at it in both Code view and Graph view.

Then, upload 10-20 images to the Flask application. For at least a few images, correct the class label. Make sure you have uploaded a few images for which the model is known to have low confidence.

Trigger the DAG manually in the web interface. Observe the effect in MinIO and in Label Studio.



In Label Studio, label some of the images in the "Food11 Continuous X" project. Then, we can run the next stage, which:

* Gets the new labels from Label Studio
* Copy the newly labeled images to "production-clean" and "production-noisy" buckets, and remove them from "production-label-wait"
* Compute the accuracy on the batch of data

Click through to this second stage DAG, and look at it in both Code view and Graph view.

Trigger the DAG manually in the web interface. Observe the effect in MinIO. 




Airflow is an extremely capable platform, and we have barely scratched the surface of what we can do with it - but now, we have the basic pieces of a continuous monitoring and re-training pipeline in place!

We have a "production-clean" bucket suitable for evaluation (with reliable labels generated by our human annotator) and a "production-noisy" bucket suitable for re-training, with labels that many not be accurate (since most are labeled by our own model!)

We could extend this pipeline to - 

* push the batch accuracy to Prometheus, for continuous monitoring
* trigger re-training

but we'll stop here for now, since we have not set up our training and monitoring infrastructure in this experiment.







## Delete resources

When we are finished, we must delete the VM server instance to make the resources available to other users.

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. 


```python
from chi import server, context
import chi, os, time, datetime

context.version = "1.0" 
context.choose_project()
context.choose_site(default="KVM@TACC")
```


```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.get_server(f"node-eval-loop-{username}")
s.delete()
```



<hr>

<small>Questions about this material? Contact Fraida Fund</small>

<hr>

<small>This material is based upon work supported by the National Science Foundation under Grant No. 2230079.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.</small>


<hr>

<small>Questions about this material? Contact Fraida Fund</small>

<hr>

<small>This material is based upon work supported by the National Science Foundation under Grant No. 2230079.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.</small>