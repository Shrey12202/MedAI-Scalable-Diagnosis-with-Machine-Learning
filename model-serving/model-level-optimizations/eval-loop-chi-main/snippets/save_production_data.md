

::: {.cell .markdown}

## Save production data

Our first step in making sure that we "close the feedback loop" is to save the data that is submitted to our service in production, so that we can later evaluate the performance of our model on "production" data.


:::

::: {.cell .markdown}

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

:::


::: {.cell .markdown}

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

:::

::: {.cell .markdown}

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

:::

::: {.cell .markdown}

Note that in addition to uploading the image, we also tag the object with

* its predicted class label 
* the model's confidence in its prediction
* and the timestamp

This will help us organize the production data for later use.

:::

::: {.cell .markdown}


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


:::



::: {.cell .markdown}

We are now saving data from our production service, including the image submitted by the user, the label assigned by our model, and the confidence of our model - but, that's not enough to tell us about performance in production, or help us improve it! We need some way to 

* evaluate whether or not our model is doing a good job assigning labels to production data
* and use production data to re-train our model using supervised learning (which requires labels)

In the next section, we'll explore some strategies for this.

:::
