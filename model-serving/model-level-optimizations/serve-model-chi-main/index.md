

# Model optimizations for serving

In this tutorial, we explore some model-level optimizations for model serving:

* graph optimizations
* quantization
* and hardware-specific execution providers, which switch out generic implementations of operations in the graph for hardware-specific optimized implementations

and we will see how these affect the throughput and inference time of a model.

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@UC and CHI@TACC sites.




## Context


The premise of this example is as follows: You are working as a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food. You have developed a convolutional neural network in Pytorch that automatically classifies photos of food into one of a set of categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.

Now that you have trained a model, you are preparing to serve predictions using this model. Your manager has advised that since GourmetGram is an early-stage startup, they can't afford much compute for serving models. Your manager wants you to prepare a few different options, that they will then price out among cloud providers and decide which to use:

* inference on a server-grade CPU (AMD EPYC 7763). Your manager wants to see an option that has less than 3ms median inference latency for a single input sample, and has a batch throughput of at least 1000 frames per second.
* inference on a server-grade GPU (A100). Since GourmetGram won't be able to afford to load balance across several GPUs, your manager said that the GPU option must have strong enough performance to handle the workload with a single GPU node: they are looking for less than 1ms median inference latency for a single input sample, and a batch throughput of at least 5000 frames per second.
* inference on end-user devices, as part of an app. For this option, the model itself should be less than 5MB on disk, because users are sensitive to storage space on mobile devices. Because the total prediction timme will not include any network delay when the model is on the end-user device, the "budget" for inference time is larger: your manager wants less than 15ms median inference latency for a single input sample on a low-resource edge device (ARM Cortex A76 processor).

You're already off to a good start, by using a MobileNetV2 as your foundation model; this is a small model that is especially designed for fast inference time. Now you need to measure the inference performance of the model and, if it doesn't meet the requirements above, investigate ways to improve it.



## Experiment resources 

For this experiment, we will provision one bare-metal node with a recent NVIDIA GPU (e.g. A100, A30). (Although most of the experiment will run on CPU, we'll also do a little bit of GPU.)

We'll use the `compute_liqid` node types at CHI@TACC, or `compute_gigaio` node types at CHI@UC. (We won't use `compute_gigaio` nodes at CHI@TACC, which have a different GPU and CPU.)

* The `compute_liqid` nodes at CHI@TACC have one or two NVIDIA A100 40GB GPUs, and an AMD EPYC 7763 CPU.
* The `compute_gigaio` nodes at CHI@UC have an NVIDIA A100 80GB GPU, and an AMD EPYC 7763 CPU.

You can decide which type to use based on availability.



## Create a lease for a GPU server



To use bare metal resources on Chameleon, we must reserve them in advance. For this experiment, we will reserve a 3-hour block on a bare metal node with GPU.

We can use the OpenStack graphical user interface, Horizon, to submit a lease. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/)
* click "Experiment" > "CHI@TACC" or "Experiment > CHI@UC", depending on which site you want to make reservation at
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.



Then, 

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `compute_liqid` or `compute_gigaio` as applicable to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone. 
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, make a note of:
  * the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
  * and the name of the node you want to reserve. (We will reserve nodes by name, not by type, to avoid getting a 1-GPU node when we wanted a 2-GPU node.)
* Then, on the left side, click on "Reservations" > "Leases", and then click on "Create Lease":
  * set the "Name" to <code>serve_model_<b>netID</b></code> where in place of <code><b>netID</b></code> you substitute your actual net ID.
  * set the start date and time in UTC. To make scheduling smoother, please start your lease on an hour boundary, e.g. `XX:00`.
  * modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time to three hours as directed. Also, to avoid a potential race condition that occurs when one lease starts immediately after another lease ends, you should end your lease five minutes before the end of an hour, e.g. at `YY:55`.
  * Click "Next".
* On the "Hosts" tab, 
  * check the "Reserve hosts" box
  * leave the "Minimum number of hosts" and "Maximum number of hosts" at 1
  * in "Resource properties", specify the node name that you identified earlier.
* Click "Next". Then, click "Create". (We won't include any network resources in this lease.)
  
Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.



Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.



## At the beginning of your GPU server lease


At the beginning of your GPU lease time, you will continue with the next step, in which you bring up and configure a bare metal instance! To begin this step, open this experiment on Trovi:

* Use this link: [Model optimizations for serving machine learning models](https://chameleoncloud.org/experiment/share/f5acccf8-f2cb-4d1e-8918-4c8fd97bfc32) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it, including the notebok to bring up the bare metal server.







## Launch and set up NVIDIA A100 or A30 server - with python-chi

At the beginning of the lease time for your bare metal server, we will bring up our GPU instance. We will use the `python-chi` Python API to Chameleon to provision our server. 

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. Also **change the site to CHI@TACC or CHI@UC**, depending on where your reservation is.


```python
from chi import server, context, lease
import os

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@TACC")
```


Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:


```python
l = lease.get_lease(f"serve_model_netID") 
l.show()
```


The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook can be executed without any interactions from you, so at this point, you can save time by clicking on this cell, then selecting "Run" > "Run Selected Cell and All Below" from the Jupyter menu.  

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!



We will use the lease to bring up a server with the `CC-Ubuntu24.04-CUDA` disk image. 

> **Note**: the following cell brings up a server only if you don't already have one with the same name! (Regardless of its error state.) If you have a server in ERROR state already, delete it first in the Horizon GUI before you run this cell.



```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-serve-model-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-CUDA"
)
s.submit(idempotent=True)
```


Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.



Then, we'll associate a floating IP with the instance, so that we can access it over SSH.


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





### Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.


```python
s.execute("git clone https://github.com/teaching-on-testbeds/serve-model-chi")
```



### Set up Docker

To use common deep learning frameworks like Tensorflow or PyTorch, and ML training platforms like MLFlow and Ray, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.


```python
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```


### Set up the NVIDIA container toolkit


We will also install the NVIDIA container toolkit, with which we can access GPUs from inside our containers.


```python
s.execute("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
s.execute("sudo apt update")
s.execute("sudo apt-get install -y nvidia-container-toolkit")
s.execute("sudo nvidia-ctk runtime configure --runtime=docker")
# for https://github.com/NVIDIA/nvidia-container-toolkit/issues/48
s.execute("sudo jq 'if has(\"exec-opts\") then . else . + {\"exec-opts\": [\"native.cgroupdriver=cgroupfs\"]} end' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp > /dev/null && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json")
s.execute("sudo systemctl restart docker")
```



## Open an SSH session

Finally, open an SSH sesson on your server. From your local terminal, run

```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where

* in place of `~/.ssh/id_rsa_chameleon`, substitute the path to your own key that you had uploaded to CHI@TACC
* in place of `A.B.C.D`, use the floating IP address you just associated to your instance.





## Prepare data

For the rest of this tutorial, we'll be training models on the [Food-11 dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/). We're going to prepare a Docker volume with this dataset already prepared on it, so that the containers we create later can attach to this volume and access the data. 




First, create the volume:

```bash
# runs on node-serve-model
docker volume create food11
```

Then, to populate it with data, run

```bash
# runs on node-serve-model
docker compose -f serve-model-chi/docker/docker-compose-data.yaml up -d
```

This will run a temporary container that downloads the Food-11 dataset, organizes it in the volume, and then stops. It may take a minute or two. You can verify with 

```bash
# runs on node-serve-model
docker ps
```

that it is done - when there are no running containers.

Finally, verify that the data looks as it should. Start a shell in a temporary container with this volume attached, and `ls` the contents of the volume:

```bash
# runs on node-mltrain
docker run --rm -it -v food11:/mnt alpine ls -l /mnt/Food-11/
```

it should show "evaluation", "validation", and "training" subfolders.



## Launch a Jupyter container

Inside the SSH session, build a container image for a Jupyter server with ONNX and related libraries for CPU inference installed:

```bash
# run on node-serve-model 
docker build -t jupyter-onnx -f serve-model-chi/docker/Dockerfile.jupyter-onnx-gpu .
```

Then, launch the container:

```bash
# run on node-serve-model 
docker run  -d --rm  -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/serve-model-chi/workspace:/home/jovyan/work/ \
    -v food11:/mnt/ \
    -e FOOD11_DATA_DIR=/mnt/Food-11 \
    --name jupyter \
    jupyter-onnx
```

Run

```bash
# run on node-serve-model 
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `4_measure_torch.ipynb` notebook to continue.




## Measure inference performance of PyTorch model on CPU 

First, we are going to measure the inference performance of an already-trained PyTorch model on CPU. After completing this section, you should understand:

* how to measure the inference latency of a PyTorch model
* how to measure the throughput of batch inference of a PyTorch model
* how to compare eager model execution vs a compiled model

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.


```python
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
import time
import numpy as np
```



First, let's load our saved model in evaluation mode, and print a summary of it. Note that for now, we will use the CPU for inference, not GPU.


```python
model_path = "models/food11.pth"  
device = torch.device("cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()  
summary(model)
```


and also prepare our test dataset:


```python
food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")
val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'), transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```



We will measure:

* the size of the model on disk
* the latency when doing inference on single samples
* the throughput when doing inference on batches of data
* and the test accuracy




#### Model size

We'll start with model size. Our default `food11.pth` is a finetuned MobileNetV2, which is a small model designed for deployment on edge devices, so it is fairly small.

```python
model_size = os.path.getsize(model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```


#### Test accuracy

Next, we'll measure the accuracy of this model on the test data


```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get predicted class index
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
```

```python
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
```




#### Inference latency

Now, we'll measure how long it takes the model to return a prediction for a single sample. We will run 100 trials, and then compute aggregate statistics.


```python
num_trials = 100  # Number of trials

# Get a single sample from the test data

single_sample, _ = next(iter(test_loader))  
single_sample = single_sample[0].unsqueeze(0)  

# Warm-up run 
with torch.no_grad():
    model(single_sample)

latencies = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        _ = model(single_sample)
        latencies.append(time.time() - start_time)
```



```python
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
```


#### Batch throughput 

Finally, we'll measure the rate at which the model can return predictions for batches of data. 


```python
num_batches = 50  # Number of trials

# Get a batch from the test data
batch_input, _ = next(iter(test_loader))  

# Warm-up run 
with torch.no_grad():
    model(batch_input)

batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        _ = model(batch_input)
        batch_times.append(time.time() - start_time)
```

```python
batch_fps = (batch_input.shape[0] * num_batches) / np.sum(batch_times) 
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```



#### Summary of results


```python
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```


When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)




### Eager mode execution vs compiled model

We had just evaluated a model in eager mode. However, in some (although, not all) cases we may get better performance from compiling the model into a graph, and executing it as a graph.

Go back to the cell where the model is loaded, and add

```python
model.compile()
```

just below the call to `torch.load`. Then, run the notebook again ("Run > Run All Cells"). 

When you are done, download the fully executed notebook **again** from the Jupyter container environment for later reference.





<!-- 

compute_gigaio 

  Model name:             AMD EPYC 7763 64-Core Processor
    CPU family:           25
    Model:                1
    Thread(s) per core:   2
    Core(s) per socket:   64

-->


<!-- summary for mobilenet model

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 60.16 ms
Inference Latency (single sample, 95th percentile): 77.22 ms
Inference Latency (single sample, 99th percentile): 77.37 ms
Inference Throughput (single sample): 15.82 FPS
Batch Throughput: 83.66 FPS


Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 73.97 ms
Inference Latency (single sample, 95th percentile): 83.16 ms
Inference Latency (single sample, 99th percentile): 83.94 ms
Inference Throughput (single sample): 13.34 FPS
Batch Throughput: 98.80 FPS

-->


<!-- summary for mobilenet compiled model

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 26.92 ms
Inference Latency (single sample, 95th percentile): 49.79 ms
Inference Latency (single sample, 99th percentile): 64.55 ms
Inference Throughput (single sample): 32.35 FPS
Batch Throughput: 249.08 FPS

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 34.14 ms
Inference Latency (single sample, 95th percentile): 53.85 ms
Inference Latency (single sample, 99th percentile): 60.23 ms
Inference Throughput (single sample): 27.39 FPS
Batch Throughput: 281.65 FPS

-->

<!-- 

(Intel CPU)

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 12.69 ms
Inference Latency (single sample, 95th percentile): 12.83 ms
Inference Latency (single sample, 99th percentile): 12.97 ms
Inference Throughput (single sample): 78.73 FPS
Batch Throughput: 161.27 FPS

With compiling

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 8.47 ms
Inference Latency (single sample, 95th percentile): 8.58 ms
Inference Latency (single sample, 99th percentile): 8.79 ms
Inference Throughput (single sample): 117.86 FPS
Batch Throughput: 474.67 FPS



-->




## Measure inference performance of ONNX model on CPU 

To squeeze even more inference performance out of our model, we are going to convert it to ONNX format, which allows models from different frameworks (PyTorch, Tensorflow, Keras), to be deployed on a variety of different hardware platforms (CPU, GPU, edge devices), using many optimizations (graph optimizations, quantization, target device-specific implementations, and more).

After finishing this section, you should know:

* how to convert a PyTorch model to ONNX
* how to measure the inference latency and batch throughput of the ONNX model

and then you will use it to evaluate the optimized models you develop in the next section.

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.


```python
import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

```python
# Prepare test dataset
food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")
val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'), transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```




First, let's load our saved PyTorch model, and convert it to ONNX using PyTorch's built-in `torch.onnx.export`:


```python
model_path = "models/food11.pth"  
device = torch.device("cpu")
model = torch.load(model_path, map_location=device, weights_only=False)

onnx_model_path = "models/food11.onnx"
# dummy input - used to clarify the input shape
dummy_input = torch.randn(1, 3, 224, 224)  
torch.onnx.export(model, dummy_input, onnx_model_path,
                  export_params=True, opset_version=20,
                  do_constant_folding=True, input_names=['input'],
                  output_names=['output'], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print(f"ONNX model saved to {onnx_model_path}")

onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
```


## Create an inference session

Now, we can evaluate our model! To use an ONNX model, we create an *inference session*, and then use the model within that session. Let's start an inference session:





```python
onnx_model_path = "models/food11.onnx"
```

```python
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
```




and let's double check the execution provider that will be used in this session:



```python
ort_session.get_providers()
```





#### Test accuracy


First, let's measure accuracy on the test set:


```python
correct = 0
total = 0
for images, labels in test_loader:
    images_np = images.numpy()
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: images_np})[0]
    predicted = np.argmax(outputs, axis=1)
    total += labels.size(0)
    correct += (predicted == labels.numpy()).sum()
accuracy = (correct / total) * 100
```

```python
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
```


#### Model size

We are also concerned with the size of the ONNX model on disk. It will be similar to the equivalent PyTorch model size (to start!)


```python
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```





#### Inference latency

Now, we'll measure how long it takes the model to return a prediction for a single sample. We will run 100 trials, and then compute aggregate statistics.


```python
num_trials = 100  # Number of trials

# Get a single sample from the test data

single_sample, _ = next(iter(test_loader))  
single_sample = single_sample[:1].numpy()

# Warm-up run
ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})

latencies = []
for _ in range(num_trials):
    start_time = time.time()
    ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})
    latencies.append(time.time() - start_time)
```



```python
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
```


#### Batch throughput 

Finally, we'll measure the rate at which the model can return predictions for batches of data. 


```python
num_batches = 50  # Number of trials

# Get a batch from the test data
batch_input, _ = next(iter(test_loader))  
batch_input = batch_input.numpy()

# Warm-up run
ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})

batch_times = []
for _ in range(num_batches):
    start_time = time.time()
    ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})
    batch_times.append(time.time() - start_time)
```

```python
batch_fps = (batch_input.shape[0] * num_batches) / np.sum(batch_times) 
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```




#### Summary of results


```python
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```

<!-- summary for mobilenet

Model Size on Disk: 8.92 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 8.92 ms
Inference Latency (single sample, 95th percentile): 9.15 ms
Inference Latency (single sample, 99th percentile): 9.41 ms
Inference Throughput (single sample): 112.06 FPS
Batch Throughput: 993.48 FPS

Model Size on Disk: 8.92 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.64 ms
Inference Latency (single sample, 95th percentile): 10.57 ms
Inference Latency (single sample, 99th percentile): 11.72 ms
Inference Latency (single sample, std error): 0.04 ms
Inference Throughput (single sample): 102.52 FPS
Batch Throughput: 1083.57 FPS

Accuracy: 90.59% (3032/3347 correct)
Model Size on Disk: 8.92 MB
Inference Latency (single sample, median): 16.24 ms
Inference Latency (single sample, 95th percentile): 18.06 ms
Inference Latency (single sample, 99th percentile): 18.72 ms
Inference Throughput (single sample): 63.51 FPS
Batch Throughput: 1103.28 FPS


-->


<!-- summary for mobilenet with graph optimization

Model Size on Disk: 8.91 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.31 ms
Inference Latency (single sample, 95th percentile): 9.47 ms
Inference Latency (single sample, 99th percentile): 9.71 ms
Inference Throughput (single sample): 107.22 FPS
Batch Throughput: 1091.58 FPS

Model Size on Disk: 8.91 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.95 ms
Inference Latency (single sample, 95th percentile): 10.14 ms
Inference Latency (single sample, 99th percentile): 10.70 ms
Inference Latency (single sample, std error): 0.02 ms
Inference Throughput (single sample): 100.18 FPS
Batch Throughput: 1022.77 FPS

Model Size on Disk: 8.91 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.55 ms
Inference Latency (single sample, 95th percentile): 10.58 ms
Inference Latency (single sample, 99th percentile): 11.14 ms
Inference Latency (single sample, std error): 0.04 ms
Inference Throughput (single sample): 102.97 FPS
Batch Throughput: 1079.81 FPS


-->


<!-- 

(Intel CPU)

Accuracy: 90.59% (3032/3347 correct)
Model Size on Disk: 8.92 MB
Inference Latency (single sample, median): 4.53 ms
Inference Latency (single sample, 95th percentile): 4.63 ms
Inference Latency (single sample, 99th percentile): 4.99 ms
Inference Throughput (single sample): 218.75 FPS
Batch Throughput: 2519.80 FPS


-->



When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

Also download the `food11.onnx` model from inside the `models` directory.





## Apply optimizations to ONNX model

Now that we have an ONNX model, we can apply some basic optimizations. After completing this section, you should be able to apply:

* graph optimizations, e.g. fusing operations
* post-training quantization (dynamic and static)
* and hardware-specific execution providers

to improve inference performance. 

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.




Since we are going to evaluate several models, we'll define a benchmark function here to help us compare them:



```python
import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

```python
# Prepare test dataset
food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")
val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'), transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```


```python
def benchmark_session(ort_session):

    print(f"Execution provider: {ort_session.get_providers()}")

    ## Benchmark accuracy

    correct = 0
    total = 0
    for images, labels in test_loader:
        images_np = images.numpy()
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: images_np})[0]
        predicted = np.argmax(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
    accuracy = (correct / total) * 100

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

    ## Benchmark inference latency for single sample

    num_trials = 100  # Number of trials

    # Get a single sample from the test data

    single_sample, _ = next(iter(test_loader))  
    single_sample = single_sample[:1].numpy()

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})

    latencies = []
    for _ in range(num_trials):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})
        latencies.append(time.time() - start_time)

    print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
    print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")

    ## Benchmark batch throughput

    num_batches = 50  # Number of trials

    # Get a batch from the test data
    batch_input, _ = next(iter(test_loader))  
    batch_input = batch_input.numpy()

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})

    batch_times = []
    for _ in range(num_batches):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})
        batch_times.append(time.time() - start_time)

    batch_fps = (batch_input.shape[0] * num_batches) / np.sum(batch_times) 
    print(f"Batch Throughput: {batch_fps:.2f} FPS")

```





### Apply basic graph optimizations

Let's start by applying some basic [graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode), e.g. fusing operations. 

We will save the model after applying graph optimizations to `models/food11_optimized.onnx`, then evaluate that model in a new session.


```python
onnx_model_path = "models/food11.onnx"
optimized_model_path = "models/food11_optimized.onnx"

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED # apply graph optimizations
session_options.optimized_model_filepath = optimized_model_path 

ort_session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=['CPUExecutionProvider'])
```



Download the `food11_optimized.onnx` model from inside the `models` directory. 


To see the effect of the graph optimizations, we can visualize the models using [Netron](https://netron.app/). Upload the original `food11.onnx` and review the graph. Then, upload the `food11_optimized.onnx` and see what has changed in the "optimized" graph.




Next, evaluate the optimized model. The graph optimizations may improve the inference performance, may have negligible effect, OR they can make it worse, depending on the model and the hardware environment in which the model is executed.



```python
onnx_model_path = "models/food11_optimized.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```

<!--

On gigaio AMD EPYC:


Execution provider: ['CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 8.70 ms
Inference Latency (single sample, 95th percentile): 8.88 ms
Inference Latency (single sample, 99th percentile): 9.24 ms
Inference Throughput (single sample): 114.63 FPS
Batch Throughput: 1153.63 FPS

On liqid Intel:

Execution provider: ['CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 4.63 ms
Inference Latency (single sample, 95th percentile): 4.67 ms
Inference Latency (single sample, 99th percentile): 4.75 ms
Inference Throughput (single sample): 214.45 FPS
Batch Throughput: 2488.54 FPS

-->


### Apply post training quantization

We will continue our quest to improve inference speed! The next optimization we will attempt is quantization.

There are many frameworks that offer quantization - for our Food11 model, we could:

* use [PyTorch quantization](https://pytorch.org/docs/stable/quantization.html#introduction-to-quantization)
* use [ONNX quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
* use [Intel Neural Compressor](https://intel.github.io/neural-compressor/latest/index.html) (which supports PyTorch and ONNX models)
* use [NNCF](https://github.com/openvinotoolkit/nncf) if we plan to use the OpenVINO execution provider
* etc...

These frameworks vary in the type of quantization they support, the range of operations that may be quantized, and many other details.

We will use Intel Neural Compressor, which in addition to supporting many ML frameworks and many types of quantization has an interesting feature: it supports quantization up to a specified evaluation threshold. In other words, we can specify "quantize as much as possible, but without losing more than 0.01 accuracy" and Intel Neural Compressor will find the best quantized version of the model that does not lose more than 0.01 accuracy.





Post-training quantization comes in two main types. In both types, FP32 values will be converted in INT8, using

$$X_{\text{INT8}} = \text{round} ( \text{scale}  \times X_{\text{FP32}} + \text{zero\_point} )$$

but they differ with respect to when and how the quantization parameters "scale" and "zero point" are computed:

* dynamic quantization: weights are quantized in advance and stored in INT8 representation. The quantization parameters for the activations are computed during inference. 
* static quantization: weights are quantized in advance and stored in INT8, and the quantization parameters are also set in advance for activations. This approach requires the use of a "calibration dataset" during quantization, to set the quantization parameters for the activations.

 




#### Dynamic quantization

We will start with dynamic quantization. No calibration dataset is required. 
 

```python
import neural_compressor
from neural_compressor import quantization
```

```python
# Load ONNX model into Intel Neural Compressor
model_path = "models/food11.onnx"
fp32_model = neural_compressor.model.onnx_model.ONNXModel(model_path)

# Configure the quantizer
config_ptq = neural_compressor.PostTrainingQuantConfig(
    approach="dynamic"
)

# Fit the quantized model
q_model = quantization.fit(
    model=fp32_model, 
    conf=config_ptq
)

# Save quantized model
q_model.save_model_to_file("models/food11_quantized_dynamic.onnx")
```



Download the `food11_quantized_dynamic.onnx` model from inside the `models` directory. 


To see the effect of the graph optimizations, we can visualize the models using [Netron](https://netron.app/). Upload the original `food11.onnx` and review the graph. Then, upload the `food11_quantized_dynamic.onnx` and see what has changed in the quantized graph.

Note that some of our operations have become integer operations, but we have added additional operations to quantize and dequantize activations throughout the graph. 



We are also concerned with the size of the quantized model on disk:


```python
onnx_model_path = "models/food11_quantized_dynamic.onnx"
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```




Next, evaluate the quantized model. Since we are saving weights in integer form, the model size is smaller. With respect to inference time, however, while the integer operations may be faster than their FP32 equivalents, the dynamic quantization and dequantization of activations may add more compute time than we save from integer operations.



```python
onnx_model_path = "models/food11_quantized_dynamic.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```


<!-- 

On liqid AMD EPYC

Model Size on Disk: 2.42 MB
Execution provider: ['CPUExecutionProvider']
Accuracy: 82.04% (2746/3347 correct)
Inference Latency (single sample, median): 22.32 ms
Inference Latency (single sample, 95th percentile): 22.97 ms
Inference Latency (single sample, 99th percentile): 23.14 ms
Inference Throughput (single sample): 44.71 FPS
Batch Throughput: 38.34 FPS

On liqid Intel

Execution provider: ['CPUExecutionProvider']
Accuracy: 84.58% (2831/3347 correct)
Inference Latency (single sample, median): 28.29 ms
Inference Latency (single sample, 95th percentile): 29.00 ms
Inference Latency (single sample, 99th percentile): 29.07 ms
Inference Throughput (single sample): 35.28 FPS

-->



#### Static quantization


Next, we will try static quantization with a calibration dataset. 

First, let's prepare the calibration dataset. This dataset will also be used to evaluate the quantized model, to see if it meets the accuracy criterion we will set.


```python
import neural_compressor
from neural_compressor import quantization
from torchvision import datasets, transforms
```


```python
food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")
val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
val_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'validation'), transform=val_test_transform)
eval_dataloader = neural_compressor.data.DataLoader(framework='onnxruntime', dataset=val_dataset)
```


Then, we'll configure the quantizer. We'll start with a more aggressive quantization strategy - we will prefer to quantize as much as possible, as long as the accuracy of the quantized model is not more than **0.05** less than the accuracy of the original FP32 model.




```python
# Load ONNX model into Intel Neural Compressor
model_path = "models/food11.onnx"
fp32_model = neural_compressor.model.onnx_model.ONNXModel(model_path)

# Configure the quantizer
config_ptq = neural_compressor.PostTrainingQuantConfig(
    accuracy_criterion = neural_compressor.config.AccuracyCriterion(
        criterion="absolute",  
        tolerable_loss=0.05  # We will tolerate up to 0.05 less accuracy in the quantized model
    ),
    approach="static", 
    device='cpu', 
    quant_level=1,
    quant_format="QOperator", 
    recipes={"graph_optimization_level": "ENABLE_EXTENDED"}, 
    calibration_sampling_size=128
)

# Find the best quantized model meeting the accuracy criterion
q_model = quantization.fit(
    model=fp32_model, 
    conf=config_ptq, 
    calib_dataloader=eval_dataloader,
    eval_dataloader=eval_dataloader, 
    eval_metric=neural_compressor.metric.Metric(name='topk')
)

# Save quantized model
q_model.save_model_to_file("models/food11_quantized_aggressive.onnx")
```


Download the `food11_quantized_aggressive.onnx` model from inside the `models` directory. 


To see the effect of the graph optimizations, we can visualize the models using [Netron](https://netron.app/). Upload the original `food11.onnx` and review the graph. Then, upload the `food11_quantized_aggressive.onnx` and see what has changed in the quantized graph.

Note that within the parameters for each quantized operation, we now have a "scale" and "zero point" - these are used to convert the FP32 values to INT8 values, as described above. The optimal scale and zero point for weights is determined by the fitted weights themselves, but the calibration dataset was required to find the optimal scale and zero point for activations.





Let's get the size of the quantized model on disk:


```python
onnx_model_path = "models/food11_quantized_aggressive.onnx"
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```




Next, evaluate the quantized model.


```python
onnx_model_path = "models/food11_quantized_aggressive.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```



<!-- 

On AMD EPYC

Model Size on Disk: 2.42 MB
Accuracy: 87.12% (2916/3347 correct)
Inference Latency (single sample, median): 7.52 ms
Inference Latency (single sample, 95th percentile): 7.78 ms
Inference Latency (single sample, 99th percentile): 7.84 ms
Inference Throughput (single sample): 132.40 FPS
Batch Throughput: 899.98 FPS

Model Size on Disk: 2.42 MB
Accuracy: 87.12% (2916/3347 correct)
Inference Latency (single sample, median): 7.85 ms
Inference Latency (single sample, 95th percentile): 8.14 ms
Inference Latency (single sample, 99th percentile): 8.26 ms
Inference Throughput (single sample): 126.58 FPS
Batch Throughput: 739.48 FPS

On Intel

Execution provider: ['CPUExecutionProvider']
Accuracy: 89.87% (3008/3347 correct)
Inference Latency (single sample, median): 2.51 ms
Inference Latency (single sample, 95th percentile): 2.60 ms
Inference Latency (single sample, 99th percentile): 2.71 ms
Inference Throughput (single sample): 396.18 FPS
Batch Throughput: 2057.18 FPS


-->


Let's try a more conservative approach to static quantization next - we'll allow an accuracy loss only up to **0.01**. 

This time, we will see that the quantizer tries a few different "recipes" - in many of them, only some of the operations are quantized, in order to try and reach the target accuracy. After each tuning attempt, it tests the quantized model on the evaluation dataset, to see if it meets the accuracy criterion; if not, it tries again.



```python
# Load ONNX model into Intel Neural Compressor
model_path = "models/food11.onnx"
fp32_model = neural_compressor.model.onnx_model.ONNXModel(model_path)

# Configure the quantizer
config_ptq = neural_compressor.PostTrainingQuantConfig(
    accuracy_criterion = neural_compressor.config.AccuracyCriterion(
        criterion="absolute",  
        tolerable_loss=0.01  # We will tolerate up to 0.01 less accuracy in the quantized model
    ),
    approach="static", 
    device='cpu', 
    quant_level=0,  # 0 is a less aggressive quantization level
    quant_format="QOperator", 
    recipes={"graph_optimization_level": "ENABLE_EXTENDED"}, 
    calibration_sampling_size=128
)

# Find the best quantized model meeting the accuracy criterion
q_model = quantization.fit(
    model=fp32_model, 
    conf=config_ptq, 
    calib_dataloader=eval_dataloader,
    eval_dataloader=eval_dataloader, 
    eval_metric=neural_compressor.metric.Metric(name='topk')
)

# Save quantized model
q_model.save_model_to_file("models/food11_quantized_conservative.onnx")
```


Download the `food11_quantized_conservative.onnx` model from inside the `models` directory. 


To see the effect of the quantization, we can visualize the models using [Netron](https://netron.app/). Upload the `food11_quantized_conservative.onnx` and see what has changed in the quantized graph, relative to the "aggressive quantization" graph.

In this graph, since only some operations are quantized, we have a "Quantize" node before each quantized operation in the graph, and a "Dequantize" node after.






Let's get the size of the quantized model on disk:


```python
onnx_model_path = "models/food11_quantized_conservative.onnx"
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```




Next, evaluate the quantized model. While we see some savings in model size relative to the unquantized model, the additional quantize and dequantize operations can make the inference time much slower.

However, these tradeoffs vary from one model to the next, and across implementations and hardware. In some cases, the quantize-dequantize model may still have faster inference times than the unquantized models.




```python
onnx_model_path = "models/food11_quantized_conservative.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```



<!--

on AMD EPYC

Model Size on Disk: 6.01 MB
Accuracy: 90.20% (3019/3347 correct)
Inference Latency (single sample, median): 10.20 ms
Inference Latency (single sample, 95th percentile): 10.39 ms
Inference Latency (single sample, 99th percentile): 10.66 ms
Inference Throughput (single sample): 97.87 FPS
Batch Throughput: 277.23 FPS

On intel

Execution provider: ['CPUExecutionProvider']
Accuracy: 90.44% (3027/3347 correct)
Inference Latency (single sample, median): 6.60 ms
Inference Latency (single sample, 95th percentile): 6.66 ms
Inference Latency (single sample, 99th percentile): 6.68 ms
Inference Throughput (single sample): 151.36 FPS
Batch Throughput: 540.19 FPS

-->

<!--



### Quantization aware training

To achieve the best of both worlds - high accuracy, but the small model size and faster inference time of a quantized model - we can try quantization aware training. In QAT, the effect of quantization is "simulated" during training, so that we learn weights that are more robust to quantization. Then, when we quantize the model, we can achieve better accuracy.


-->




When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

Also download the models from inside the `models` directory.





### Try a different execution provider

Once a model is in ONNX format, we can use it with many *execution providers*. In ONNX, an execution provider an interface that lets ONNX models run with special hardware-specific capabilities. Until now, we have been using the `CPUExecutionProvider`, but if we use hardware-specific capabilities, e.g. switch out generic implementations of graph operations for implementations that are optimized for specific hardware, we can execute exactly the same model, much faster.




```python
import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

```python
# Prepare test dataset
food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")
val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root=os.path.join(food_11_data_dir, 'evaluation'), transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```


```python
def benchmark_session(ort_session):

    print(f"Execution provider: {ort_session.get_providers()}")

    ## Benchmark accuracy

    correct = 0
    total = 0
    for images, labels in test_loader:
        images_np = images.numpy()
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: images_np})[0]
        predicted = np.argmax(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
    accuracy = (correct / total) * 100

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

    ## Benchmark inference latency for single sample

    num_trials = 100  # Number of trials

    # Get a single sample from the test data

    single_sample, _ = next(iter(test_loader))  
    single_sample = single_sample[:1].numpy()

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})

    latencies = []
    for _ in range(num_trials):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})
        latencies.append(time.time() - start_time)

    print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
    print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")

    ## Benchmark batch throughput

    num_batches = 50  # Number of trials

    # Get a batch from the test data
    batch_input, _ = next(iter(test_loader))  
    batch_input = batch_input.numpy()

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})

    batch_times = []
    for _ in range(num_batches):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})
        batch_times.append(time.time() - start_time)

    batch_fps = (batch_input.shape[0] * num_batches) / np.sum(batch_times) 
    print(f"Batch Throughput: {batch_fps:.2f} FPS")

```






#### CPU execution provider

First, for reference, we'll repeat our performance test for the (unquantized model with) `CPUExecutionProvider`:





```python
onnx_model_path = "models/food11.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```

<!--
Execution provider: ['CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.93 ms
Inference Latency (single sample, 95th percentile): 14.20 ms
Inference Latency (single sample, 99th percentile): 14.43 ms
Inference Throughput (single sample): 91.10 FPS
Batch Throughput: 1042.47 FPS
-->



#### CUDA execution provider


Next, we'll try it with the CUDA execution provider, which will execute the model on the GPU:





```python
onnx_model_path = "models/food11.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
benchmark_session(ort_session)
ort.get_device()
```

<!--
Execution provider: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 0.89 ms
Inference Latency (single sample, 95th percentile): 0.90 ms
Inference Latency (single sample, 99th percentile): 0.91 ms
Inference Throughput (single sample): 1117.06 FPS
Batch Throughput: 5181.99 FPS
-->



#### TensorRT execution provider


The TensorRT execution provider will optimize the model for inference on NVIDIA GPUs. It will take a long time to run this cell, because it spends a lot of time optimizing the model (finding the best subgraphs, etc.) - but once the model is loaded, its inference time will be much faster than any of our previous tests.




```python
onnx_model_path = "models/food11.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['TensorrtExecutionProvider'])
benchmark_session(ort_session)
ort.get_device()
```

<!--
Execution provider: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 0.63 ms
Inference Latency (single sample, 95th percentile): 0.64 ms
Inference Latency (single sample, 99th percentile): 0.70 ms
Inference Throughput (single sample): 1572.61 FPS
Batch Throughput: 9274.45 FPS
-->





#### OpenVINO execution provider

Even just on CPU, we can still use an optimized execution provider to improve inference performance. We will try out the Intel [OpenVINO](https://github.com/openvinotoolkit/openvino) execution provider. However, ONNX runtime can be built to support CUDA/TensorRT or OpenVINO, but not both at the same time, so we will need to bring up a new container.

Close this Jupyter server tab - you will reopen it shortly, with a new token.

Go back to your SSH session on "node-serve-model", and build a container image for a Jupyter server with ONNX and OpenVINO:

```bash
# run on node-serve-model 
docker build -t jupyter-onnx-openvino -f serve-model-chi/docker/Dockerfile.jupyter-onnx-cpu .
```

Stop the current Jupyter server:

```bash
# run on node-serve-model 
docker stop jupyter
```

Then, launch a container with the new image you just built:

```bash
# run on node-serve-model 
docker run  -d --rm  -p 8888:8888 \
    --shm-size 16G \
    -v ~/serve-model-chi/workspace:/home/jovyan/work/ \
    -v food11:/mnt/ \
    -e FOOD11_DATA_DIR=/mnt/Food-11 \
    --name jupyter \
    jupyter-onnx-openvino
```

Run

```bash
# run on node-serve-model 
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of 127.0.0.1, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `7_ep_onnx.ipynb` notebook to continue.

Run the three cells at the top, which `import` libraries, set up the data loaders, and define the `benchmark_session` function. Then, skip to the OpenVINO section and run:


```python
onnx_model_path = "models/food11.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['OpenVINOExecutionProvider'])
benchmark_session(ort_session)
ort.get_device()
```


<!--

On AMD EPYC

Execution provider: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 1.39 ms
Inference Latency (single sample, 95th percentile): 1.89 ms
Inference Latency (single sample, 99th percentile): 1.92 ms
Inference Throughput (single sample): 646.63 FPS
Batch Throughput: 1624.30 FPS

On Intel

Execution provider: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 1.55 ms
Inference Latency (single sample, 95th percentile): 1.76 ms
Inference Latency (single sample, 99th percentile): 1.81 ms
Inference Throughput (single sample): 663.72 FPS
Batch Throughput: 2453.48 FPS

-->


When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)



<hr>

<small>Questions about this material? Contact Fraida Fund</small>

<hr>

<small>This material is based upon work supported by the National Science Foundation under Grant No. 2230079.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.</small>