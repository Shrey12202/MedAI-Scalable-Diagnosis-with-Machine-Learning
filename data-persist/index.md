

# Persistent storage on Chameleon

In this tutorial, we will practice using two types of persistent storage options on Chameleon:

* object storage, which you may use to e.g. store large training data sets
* and block storage, which you may use for persistent storage for services that run on VM instances (e.g. MLFlow, Prometheus, etc.)

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You should also have added your SSH key to the KVM@TACC site and the CHI@TACC site.



## Experiment resources 

For this experiment, we will provision 

* one block storage volume on KVM@TACC
* one object storage bucket on CHI@TACC
* one virtual machine on KVM@TACC, with a floating IP, to practice using the persistent storage



## Open this experiment on Trovi


When you are ready to begin, you will continue with the next step! To begin this step, open this experiment on Trovi:

* Use this link: [Persistent storage on Chameleon](https://chameleoncloud.org/experiment/share/a1c68238-81f8-498d-8323-9d6c46cb0a78) on Trovi
* Then, click "Launch on Chameleon". This will start a new Jupyter server for you, with the experiment materials already in it.

You will see several notebooks inside the "data-persist-chi" directory - look for the one titled `0_intro.ipynb`. Open this notebook and continue there.




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


We will use bring up a `m1.large` flavor server with the `CC-Ubuntu24.04` disk image. 

> **Note**: the following cell brings up a server only if you don't already have one with the same name! (Regardless of its error state.) If you have a server in ERROR state already, delete it first in the Horizon GUI before you run this cell.



```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-persist-{username}", 
    image_name="CC-Ubuntu24.04",
    flavor_name="m1.large"
)
s.submit(idempotent=True)
```



Then, we'll associate a floating IP with the instance:


```python
s.associate_floating_ip()
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
  {'name': "allow-8888", 'port': 8888, 'description': "Enable TCP port 8888 (used by Jupyter)"},
  {'name': "allow-8000", 'port': 8000, 'description': "Enable TCP port 8000 (used by MLFlow)"},
  {'name': "allow-9000", 'port': 9000, 'description': "Enable TCP port 9000 (used by MinIO API)"},
  {'name': "allow-9001", 'port': 9001, 'description': "Enable TCP port 9001 (used by MinIO Web UI)"}
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

```python
s.refresh()
s.check_connectivity()
```




### Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.


```python
s.execute("git clone https://github.com/teaching-on-testbeds/data-persist-chi")
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




## Using object storage

Until now, in any experiment we have run on Chameleon, we had to re-download large training sets each time we launched a new compute instance to work on that data. For example, in our "GourmetGram" use case, we had to re-download the Food11 dataset each time we brought up a compute instance to train or evaluate a model on that data.

For a longer-term project, we will want to persist large data sets beyond the lifetime of the compute instance. That way, we can download a very large data set *once* and then re-use it many times with different compute instances, without having to keep a compute instance "alive" all the time, or re-download the data. We will use the object storage service in Chameleon to enable this.

Of the various types of storage available in a cloud computing environment (object, block, file), object storage is the most appropriate for large training data sets. Object storage is cheap, and optimized for storing and retrieving large volumes of data, where the data is not modified frequently. (In object storage, there is no in-place modification of objects - only replacement - so it is not the best solution for files that are frequently modified.)

After you run this experiment, you will know how to:

* create an object store container at CHI@TACC
* copy objects to it,
* and mount it as a filesystem in a compute instance.

The object storage service is available at CHI@TACC or CHI@UC. In this tutorial, we will use CHI@TACC. The CHI@TACC object store can be accessed from a KVM@TACC VM instance.



### Object storage using the Horizon GUI

First, let's try creating an object storage container from the OpenStack Horizon GUI. 

Open the GUI for CHI@TACC:

* from the [Chameleon website](https://chameleoncloud.org/hardware/)
* click "Experiment" > "CHI@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.

In the menu sidebar on the left side, click on "Object Store" > "Containers" and then, "Create Container". You will be prompted to set up your container step by step using a graphical "wizard".

* Specify the name as <code>object-persist-<b>netID</b></code> where in place of <code><b>netID</b></code> you substitute your own net ID (e.g. `ff524` in my case). 
* Leave other settings at their defaults, and click "Submit".



### Use `rclone` and authenticate to object store from a compute instance

We will want to connect to this object store from the compute instance we configured earlier, and copy some data to it!

For *write* access to the object store from the compute instance, we will need to authenticate with valid OpenStack credentials. To support this, we will create an *application credential*, which consists of an ID and a secret that allows a script or application to authenticate to the service. 

An application credential is a good way for something like a data pipeline to authenticate, since it can be used non-interactively, and can be revoked easily in case it is compromised without affecting the entire user account.

In the menu sidebar on the left side of the Horizon GUI, click "Identity" > "Application Credentials". Then, click "Create Application Credential".

* In the "Name", field, use "data-persist". 
* Set the "Expiration" date to the end date of the current semester. (Note that this will be in UTC time, not your local time zone.) This ensures that if your credential is leaked (e.g. you accidentially push it to a public Github repository), the damage is mitigated.
* Click "Create Application Credential".
* Copy the "ID" and "Secret" displayed in the dialog, and save them in a safe place. You will not be able to view the secret again from the Horizon GUI. Then, click "Download openrc file" to have another copy of the secret.

Now that we have an application credential, we can use it to allow an application to authenticate to the Chameleon object store service. There are several applications and utilities for working with OpenStack's Swift object store service; we will use one called [`rclone`](https://github.com/rclone/rclone).


On the compute instance, install `rclone`:


```bash
# run on node-persist
curl https://rclone.org/install.sh | sudo bash
```

We also need to modify the configuration file for FUSE (**F**ilesystem in **USE**rspace: the interface that allows user space applications to mount virtual filesystems), so that object store containers mounted by our user will be availabe to others, including Docker containers:

```bash
# run on node-persist
# this line makes sure user_allow_other is un-commented in /etc/fuse.conf
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
```

Next, create a configuration file for `rclone` with the ID and secret from the application credential you just generated:

```bash
# run on node-persist
mkdir -p ~/.config/rclone
nano  ~/.config/rclone/rclone.conf
```

Paste the following into the config file, but substitute your own application credential ID and secret. 

You will also need to substitute your own user ID. You can find it using "Identity" > "Users" in the Horizon GUI; it is an alphanumeric string (*not* the human-readable user name).


```
[chi_tacc]
type = swift
user_id = YOUR_USER_ID
application_credential_id = APP_CRED_ID
application_credential_secret = APP_CRED_SECRET
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
```


Use Ctrl+O and Enter to save the file, and Ctrl+X to exit `nano`.

To test it, run

```bash
# run on node-persist
rclone lsd chi_tacc:
```

and verify that you see your container listed. This confirms that `rclone` can authenticate to the object store.



### Create a pipeline to load training data into the object store

Next, we will prepare a simple ETL pipeline to get the Food11 dataset into the object store. It will:

* extract the data into a staging area (local filesystem on the instance)
* transform the data, organizing it into directories by class as required by PyTorch
* and then load the data into the object store

We are going to define the pipeline stages inside a Docker compose file. All of the services in the container will share a common `food11` volume. Then, we have:

1. A service to extract the Food11 data from the Internet. This service runs a Python container image, downloads the dataset, and unzips it.

```
  extract-data:
    container_name: etl_extract_data
    image: python:3.11
    user: root
    volumes:
      - food11:/data
    working_dir: /data
    command:
      - bash
      - -c
      - |
        set -e

        echo "Resetting dataset directory..."
        rm -rf Food-11
        mkdir -p Food-11
        cd Food-11

        echo "Downloading dataset zip..."
        curl -L https://nyu.box.com/shared/static/m5kvcl25hfpljul3elhu6xz8y2w61op4.zip \
          -o Food-11.zip

        echo "Unzipping dataset..."
        unzip -q Food-11.zip
        rm -f Food-11.zip

        echo "Listing contents of /data after extract stage:"
        ls -l /data
```

2. A service that runs a Python container image, and uses a Python script to organize the data into directories according to class label.

```
  transform-data:
    container_name: etl_transform_data
    image: python:3.11
    volumes:
      - food11:/data
    working_dir: /data/Food-11
    command:
      - bash
      - -c
      - |
        set -e

        python3 -c '
        import os
        import shutil

        dataset_base_dir = "/data/Food-11"
        subdirs = ["training", "validation", "evaluation"]
        classes = [
            "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
            "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"
        ]

        for subdir in subdirs:
            dir_path = os.path.join(dataset_base_dir, subdir)
            if not os.path.exists(dir_path):
                continue

            for i, class_name in enumerate(classes):
                class_dir = os.path.join(dir_path, f"class_{i:02d}")
                os.makedirs(class_dir, exist_ok=True)
                for f in os.listdir(dir_path):
                    if f.startswith(f"{i}_"):
                        shutil.move(
                            os.path.join(dir_path, f),
                            os.path.join(class_dir, f)
                        )
        '

        echo "Listing contents of /data/Food-11 after transform stage:"
        ls -l /data/Food-11
```

3. And finally, a service that uses `rclone copy` to load the organized data into the object store. Note that we pass some arguments to `rclone copy` to increase the parallelism, so that the data is loaded more quicly. Also note that since the name of the container includes your individual net ID, we have specified it using an environment variable that must be set before this stage can run.

```
    container_name: etl_load_data
    image: rclone/rclone:latest
    volumes:
      - food11:/data
      - ~/.config/rclone/rclone.conf:/root/.config/rclone/rclone.conf:ro
    entrypoint: /bin/sh
    command:
      - -c
      - |
        if [ -z "$RCLONE_CONTAINER" ]; then
          echo "ERROR: RCLONE_CONTAINER is not set"
          exit 1
        fi
        echo "Cleaning up existing contents of container..."
        rclone delete chi_tacc:$RCLONE_CONTAINER --rmdirs || true

        rclone copy /data/Food-11 chi_tacc:$RCLONE_CONTAINER \
        --progress \
        --transfers=32 \
        --checkers=16 \
        --multi-thread-streams=4 \
        --fast-list

        echo "Listing directories in container after load stage:"
        rclone lsd chi_tacc:$RCLONE_CONTAINER
```

These services are defined in `~/data-persist-chi/docker/docker-compose-etl.yaml`.

Now, we can run the stages using Docker. (If we had a workflow orchestrator, we could use it to run the pipeline stages - but we don't really need orchestration at this point.)


```bash
# run on node-persist
docker compose -f ~/data-persist-chi/docker/docker-compose-etl.yaml run extract-data
```

```bash
# run on node-persist
docker compose -f ~/data-persist-chi/docker/docker-compose-etl.yaml run transform-data
```

For the last stage, the container name is not specified in the Docker compose YAML (since it has your net ID in it) - so we have to pass it as an environment variable first. Substitute your own net ID in the line below:

```bash
# run on node-persist
export RCLONE_CONTAINER=object-persist-netID
docker compose -f ~/data-persist-chi/docker/docker-compose-etl.yaml run load-data
```

Now our training data is loaded into the object store and ready to use for training! We can clean up the Docker volume used as the temporary staging area:

```bash
# run on node-persist
docker volume rm food11-etl_food11
```

In the Horizon GUI, note that we can browse the object store and download any file from it. This container is independent of any compute instance - it will persist, and its data is still saved, even if we have no active compute instance. (In fact, we *already* have no active compute instance on CHI@TACC.)



### Mount an object store to local file system

Now that our data is safely inside the object store, we can use it anywhere - on a VM, on a bare metal site, on multiple compute instances at once, even outside of Chameleon - to train or evaluate a model. We would not have to repeat the ETL pipeline each time we want to use the data.

If we were working on a brand-new compute instance, we would need to download `rclone` and create the `rclone` configuration file at `~/.config/rclone.conf`, as we have above. Since we already done these steps in order to load data into the object store, we don't need to repeat them.

The next step is to create a mount point for the data in the local filesystem:

```bash
# run on node-persist
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object
```

Now finally, we can use `rclone mount` to mount the object store at the mount point (substituting your own **netID** in the command below).

```bash
# run on node-persist
rclone mount chi_tacc:object-persist-netID /mnt/object --read-only --allow-other --daemon
```

Here, 

* `chi_tacc` tells `rclone` which section of its configuration file to use for authentication information
* `object-persist-netID` tells it what object store container to mount
* `/mnt/object` says where to mount it

Since we only intend to read the data, we can mount it in read-only mode and it will be slightly faster; and we are also protected from accidental writes. We also specified `--allow-other` so that we can use the mount from Docker, and `--daemon` means the `rclone` process will be started in the background.


Run 

```bash
# run on node-persist
ls /mnt/object
```

and confirm that we can now see the Food11 data directories (`evaluation`, `training`, `validation`) there.

Now, we can start a Docker container with access to that virtual "filesystem", by passing that directory as a bind mount. Note that to mount a directory that is actually a FUSE filesystem inside a Docker container, we have to pass it using a slightly different `--mount` syntax, instead of the `-v` that we had used in previous examples.

```bash
# run on node-persist
docker run -d --rm \
  -p 8888:8888 \
  --shm-size 8G \
  -e FOOD11_DATA_DIR=/mnt/Food-11 \
  -v ~/data-persist-chi/workspace:/home/jovyan/work/ \
  --mount type=bind,source=/mnt/object,target=/mnt/Food-11,readonly \
  --name jupyter \
  quay.io/jupyter/pytorch-notebook:latest
```

Run

```bash
# run on node-persist
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running on your compute instance.

Then, find the `demo.ipynb` notebook. This notebook evaluates the `food11.pth` model on the evaluation set, which is **streamed from the object store**.

To validate this, on the host, run

```bash
# run on node-persist
sudo apt update
sudo apt -y install nload
nload ens3
```

to monitor the load on the network. Run the `demo.ipynb` notebook inside the Jupyter instance running on "node-persist", which also watching the `nload` output. 

Note the incoming data volume, which should be on the order of Mbits/second when a batch is being loaded.

Close the Jupyter container tab in your browser, and then stop the container with


```bash
# run on node-persist
docker stop jupyter
```

since we will bring up a different Jupyter instance in the next section.



### Un-mount an object store

We'll keep working with this object store in the next part, so you do not have to un-mount it now. But generally speaking to stop `rclone` running and un-mount the object store, you would run

```
fusermount -u /mnt/object
```

where you specify the path of the mount point.






## Using block storage

Until now, in any experiment we have run on Chameleon, the data in our experiment did not persist beyond the lifetime of our compute. That is, once the VM instance is deleted, any data we may have generated disappears with it. For example, if we were using MLFlow for experiment tracking, when the compute instance that the MLFlow service is running on stops, we would lose all of our tracking history.

For a longer-term project, we will of course want to be able to persist data beyond the lifetime of the compute instance. That way, we can provision a compute instance, do some work, delete the compute instance, and then resume later with a *new* compute instance but pick off where we left off with respect to *data*. 

To enable this, we can create a block storage volume, which can be attached to, detached from, and re-attached to a **VM instance**> Data stored on the block storage volume persists until the block storage volume itself is created.

After you run this experiment, you will know how to 

* create a block storage volume at KVM@TACC, 
* attach it to an instance,
* create a filesystem on it and mount it,
* create and use Docker volumes on the block storage volume.
* and re-attach the block storage volume to a new instance after the original compute instance ends.



### Block storage using the Horizon GUI

First, let's try creating a block storage volume from the OpenStack Horizon GUI. Open the GUI for KVM@TACC:

* from the [Chameleon website](https://chameleoncloud.org/hardware/)
* click "Experiment" > "KVM@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.

In the menu sidebar on the left side, click on "Volumes" > "Volumes" and then, "Create Volume". You will be prompted to set up your volume step by step using a graphical "wizard".

* Specify the name as <code>block-persist-<b>netID</b></code> where in place of <code><b>netID</b></code> you substitute your own net ID (e.g. `ff524` in my case). 
* Specify the size as 2 GiB.
* Leave other settings at their defaults, and click "Create Volume".

Next, it's time to to attach the block storage volume to the compute instance we created earlier. From  "Volumes" > "Volumes", next to *your* volume, click the ▼ in the menu on the right and choose "Manage Attachments". In the "Attach to Instance" menu, choose your compute instance. Then, click "Attach Volume".

Now, the "Volumes" overview page in the Horizon GUI should show something like for your volume:

```
| Name                | Description | Size | Status | Group | Type     | Attached To                     | Availability Zone | Bootable | Encrypted |
|---------------------|-------------|------|--------|-------|----------|---------------------------------|-------------------|----------|-----------|
| block-persist-netID | -           | 2GiB | In-use | -     | ceph-ssd | /dev/vdb on node-persist-netID  | nova              | No       | No        |
```

On the instance, let's confirm that we can see the block storage volume. Run

```bash
# run on node-persist
lsblk
```

and verify that `vdb` appears in the output.

The volume is essentially a raw disk. Before we can use it **for the first time** after creating it, we need to partition the disk, create a filesystem on the partition, and mount it. In subsequent uses, we will only need to mount it.

> **Note**: if the volume already had data on it, creating a filesystem on it would erase all its data! This procedure is *only* for the initial setup of a volume, before it has any data on it.

First, we create a partition with an `ext4` filesystem, occupying the entire volume:

```bash
# run on node-persist
sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%
```

Verify that we now have the partition `vdb1` in the output of 

```bash
# run on node-persist
lsblk
```

Next, we format the partition:

```bash
# run on node-persist
sudo mkfs.ext4 /dev/vdb1
```

Finally, we can create a directory in the local filesystem, mount the partition to that directory:

```bash
# run on node-persist
sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block
```

and change the owner of that directory to the `cc` user:

```bash
# run on node-persist
sudo chown -R cc /mnt/block
sudo chgrp -R cc /mnt/block
```

Run

```bash
# run on node-persist
df -h
```

and verify that the output includes a line with `/dev/vdb1` mounted on `/mnt/block`:

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/vdb1       2.0G   24K  1.9G   1% /mnt/block
```




### Create Docker volumes on persistent storage

Now that we have a block storage volume attached to our VM instance, let's see how persistent storage can be useful.

Suppose we are going to train some ML models. We will use MLFlow for experiment tracking. However, we won't necessarily be running MLFlow *all* the time. We will probably have to bring our "platform" VM(s) down and up as we iterate on our platform design. We don't want to lose all past experiment logs and models every time we bring the VMs down.

MLFLow uses two types of data systems: a relational database (Postgresql) for experiments, metrics, and parameters; and for unstructured data like artifacts and saved models, a MinIO object store. (We could hypothetically ask MinIO to use Chameleon's object store instead of running our own MinIO, but since we have already set it up for MinIO, we'll stick to that.) 

We can use a persistent block storage backend for both types of data storage to make sure that experiment logs and models persist even when the VM instance hosting MLFlow is not running.



We are now going to use Docker Compose to bring up a set of services on the VM instance:

* an MLFlow server.
* a Postgresql database with persistent storage: the host directory `/mnt/block/postgres_data`, which is on the block storage volume, is going to be mounted to `/var/lib/postgresql/data` inside the container.
* a MinIO object store with persistent storage: the host directory `/mnt/block/minio_data`, which is on the block storage volume, is going to be mounted to `/data` inside the container.
* and a Jupyter server. As before, we pass the object store mount to the Jupyter server, so that it can also access the Food11 dataset in the object store.

To bring up these services, run

```bash
# run on node-persist
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 ) docker compose -f ~/data-persist-chi/docker/docker-compose-block.yaml up -d
```

(we need to define `HOST_IP` so that we can set the MLFLow tracking URI in the Jupyter environment.)

Run

```bash
# run on node-persist
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running on your compute instance. In the "work" directory, find and open "demo.ipynb".

Also open the MLFlow service web UI: it is at


```
http://A.B.C.D:8000
```

where in place of `A.B.C.D`, you substitute the floating IP assigned to your instance.

Let's add some MLFlow tracking to our "demo.ipynb" notebook. (There's no model training in that notebook - it's just an evaluation - but it works for demo purposes!) At the end, add a cell:

```python
import mlflow
import mlflow.pytorch
mlflow.set_experiment("food11-classifier")
with mlflow.start_run():
    mlflow.log_metric(key="eval_accuracy", value=overall_accuracy)
    mlflow.pytorch.log_model(model, "food11")
```

and run the notebook.

Confirm in the MLFlow UI that both items are logged:

* the evaluation accuracy is logged as a metric, which will be stored in the Postgresql relational database
* the model is logged as an artifact, which will be stored in a MinIO bucket



Now, let's confirm that the MLFlow data persists beyond the lifetime of the compute instance! We will now delete the compute instance.

The following cells run in the **Chameleon** Jupyter environment (not in the Jupyter environment that you are hosting on your compute instance!)



```python
# run in Chameleon Jupyter environment
from chi import context, server
import chi
import os

context.version = "1.0" 
context.choose_project()  # Select the correct project
context.choose_site(default="KVM@TACC")
username = os.getenv('USER') # exp resources will have this suffix
```


```python
# run in Chameleon Jupyter environment
# delete the old server instance!
s_old = server.get_server(f"node-persist-{username}")
s_old.delete()
```

```python
# run in Chameleon Jupyter environment
s = server.Server(
    f"node-persist-{username}", 
    image_name="CC-Ubuntu24.04",
    flavor_name="m1.large"
)
s.submit(idempotent=True)
```

```python
# run in Chameleon Jupyter environment
s.associate_floating_ip()
```


```python
# run in Chameleon Jupyter environment
s.refresh()
s.check_connectivity()
```



```python
# run in Chameleon Jupyter environment
s.refresh()
s.show(type="widget")
```



```python
# run in Chameleon Jupyter environment
security_groups = [
  {'name': "allow-ssh", 'port': 22, 'description': "Enable SSH traffic on TCP port 22"},
  {'name': "allow-8888", 'port': 8888, 'description': "Enable TCP port 8888 (used by Jupyter)"},
  {'name': "allow-8000", 'port': 8000, 'description': "Enable TCP port 8000 (used by MLFlow)"},
  {'name': "allow-9000", 'port': 9000, 'description': "Enable TCP port 9000 (used by MinIO API)"},
  {'name': "allow-9001", 'port': 9001, 'description': "Enable TCP port 9001 (used by MinIO Web UI)"}
]

os_conn = chi.clients.connection()
nova_server = chi.nova().servers.get(s.id)

for sg in security_groups:
  nova_server.add_security_group(sg['name'])

print(f"updated security groups: {[group.name for group in nova_server.list_security_group()]}")
```


```python
# run in Chameleon Jupyter environment
s.execute("git clone https://github.com/teaching-on-testbeds/data-persist-chi")
```


```python
# run in Chameleon Jupyter environment
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```


This cell will attach the block storage volume named "block-persist-**netID**" to your compute instance - edit it to substitute your *own* net ID:


```python
# run in Chameleon Jupyter environment
cinder_client = chi.clients.cinder()
volume = [v for v in cinder_client.volumes.list() if v.name=='block-persist-netID'][0] # Substitute your own net ID

volume_manager = chi.nova().volumes
volume_manager.create_server_volume(server_id = s.id, volume_id = volume.id)
```



You can confirm in the Horizon GUI that your block storage volume is now attached to the new compute instance.





Let's confirm that data we put on the block storage volume earlier, is now available on the new compute instance. 


Connect to the new instance over SSH. Mount the block storage volume:


```bash
# run on node-persist
sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block
```

and confirm that it is not empty:

```bash
# run on node-persist
ls /mnt/block
```


for example, you can see previously logged artifacts saved by MinIO:


```bash
# run on node-persist
ls /mnt/block/minio_data/mlflow-artifacts/1/
```


Use Docker compose to bring up the services again:

```bash
# run on node-persist
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 ) docker compose -f ~/data-persist-chi/docker/docker-compose-block.yaml up -d
```

In your browser, open the MLFlow service web UI at


```
http://A.B.C.D:8000
```

where in place of `A.B.C.D`, you substitute the floating IP assigned to your instance. Verify that the experiment runs logged by the previous compute instance are persisted to the new MLFlow instance.




This MLFlow demo is just an example - the same principle applies to any other platform service we might use. Services like Prometheus that run directly on a VM can use an attached block storage volume. Services like Ray, which run on bare metal for GPU training, can use a MinIO storage backend that is hosted on a VM, and uses an attached block storage volume.



### Reference: creating block volumes storage using Python

We created our block storage volume using the Horizon GUI. However it is also worthwhile to learn how to create and manage block storage volumes directly in Python, if you are automating infrastructure setup using a Python notebook.



In OpenStack, the Cinder service provides block storage volumes. We can access the already-configured (authenticated) Cinder client from `python-chi` - 


```python
# run in Chameleon Jupyter environment
# get the Cinder Python client configured by python-chi
cinder_client = chi.clients.cinder()

```

```python
# run in Chameleon Jupyter environment
# list current volumes
cinder_client.volumes.list()
```


We can use the Cinder client to create a *new* block storage volume:




```python
# run in Chameleon Jupyter environment
# create a volume, specifying name and size in GiB
volume = cinder_client.volumes.create(name=f"block-persist-python-{username}", size=2)
volume._info
```


We can attach the volume to a compute instance:


```python
# run in Chameleon Jupyter environment
server_id = chi.server.get_server(f"node-persist-{username}").id
volume_manager = chi.nova().volumes
volume_manager.create_server_volume(server_id = s.id, volume_id = volume.id)
```


or detach the volume from a compute instance:



```python
# run in Chameleon Jupyter environment
volume_manager.delete_server_volume(server_id = s.id, volume_id = volume.id)
```


Or, to completely delete a volume (loses all the data!):



```python
# run in Chameleon Jupyter environment
cinder_client.volumes.delete(volume = volume)
```



## Delete resources

When we are finished, we must delete 

* the VM server instance 
* the block storage volume
* and the object store container

to make the resources available to other users.

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. 


```python
# run in Chameleon Jupyter environment
from chi import server, context
import chi, os, time, datetime

context.version = "1.0" 
context.choose_project()
context.choose_site(default="KVM@TACC")
```



Delete the compute instance:



```python
# run in Chameleon Jupyter environment
username = os.getenv('USER') # all exp resources will have this prefix
s = server.get_server(f"node-persist-{username}")
s.delete()
```


Wait a moment for this operation to be finished before you try to delete the block storage volume - you can't delete the volume when it is attached to a running instance.




Delete the block storage volume - in the following cell, substitute your own net ID in place of **netID**:



```python
# run in Chameleon Jupyter environment
cinder_client = chi.clients.cinder()
volume = [v for v in cinder_client.volumes.list() if v.name=='block-persist-netID'][0] # Substitute your own net ID
cinder_client.volumes.delete(volume = volume)
```



And finally, delete the object store container at CHI@TACC. We will use the OpenStack Swift client to delete all the objects, and then the container. 



```python
# run in Chameleon Jupyter environment
context.choose_project()
context.choose_site(default="CHI@TACC")
```

```python
# run in Chameleon Jupyter environment
os_conn = chi.clients.connection()
token = os_conn.authorize()
storage_url = os_conn.object_store.get_endpoint()

import swiftclient
swift_conn = swiftclient.Connection(preauthurl=storage_url,
                                    preauthtoken=token,
                                    retries=5)
```


In the following cell, replace **netID** with your own net ID: 


```python
# run in Chameleon Jupyter environment
container_name = "object-persist-netID"
while True:
    _, objects = swift_conn.get_container(container_name, full_listing=True)
    if not objects:
        break
    paths = "\n".join(f"{container_name}/{obj['name']}" for obj in objects)
    swift_conn.post_account(
        headers={"Content-Type": "text/plain"},
        data=paths,
        query_string="bulk-delete"
    )
swift_conn.delete_container(container_name)
print("Container deleted.")
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