

::: {.cell .markdown}

## Measure inference performance of ONNX model on CPU 

To squeeze even more inference performance out of our model, we are going to convert it to ONNX format, which allows models from different frameworks (PyTorch, Tensorflow, Keras), to be deployed on a variety of different hardware platforms (CPU, GPU, edge devices), using many optimizations (graph optimizations, quantization, target device-specific implementations, and more).

After finishing this section, you should know:

* how to convert a PyTorch model to ONNX
* how to measure the inference latency and batch throughput of the ONNX model

and then you will use it to evaluate the optimized models you develop in the next section.

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.

:::

::: {.cell .code}
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
:::

::: {.cell .code}
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
:::



::: {.cell .markdown}

First, let's load our saved PyTorch model, and convert it to ONNX using PyTorch's built-in `torch.onnx.export`:

:::

::: {.cell .code}
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
:::

::: {.cell .markdown}

## Create an inference session

Now, we can evaluate our model! To use an ONNX model, we create an *inference session*, and then use the model within that session. Let's start an inference session:


:::



::: {.cell .code}
```python
onnx_model_path = "models/food11.onnx"
```
:::

::: {.cell .code}
```python
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
```
:::



::: {.cell .markdown}

and let's double check the execution provider that will be used in this session:

:::


::: {.cell .code}
```python
ort_session.get_providers()
```
:::




::: {.cell .markdown}

#### Test accuracy


First, let's measure accuracy on the test set:

:::

::: {.cell .code}
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
:::

::: {.cell .code}
```python
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
```
:::

::: {.cell .markdown}

#### Model size

We are also concerned with the size of the ONNX model on disk. It will be similar to the equivalent PyTorch model size (to start!)

:::

::: {.cell .code}
```python
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```
:::




::: {.cell .markdown}

#### Inference latency

Now, we'll measure how long it takes the model to return a prediction for a single sample. We will run 100 trials, and then compute aggregate statistics.

:::

::: {.cell .code}
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
:::



::: {.cell .code}
```python
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
```
:::

::: {.cell .markdown}

#### Batch throughput 

Finally, we'll measure the rate at which the model can return predictions for batches of data. 

:::

::: {.cell .code}
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
:::

::: {.cell .code}
```python
batch_fps = (batch_input.shape[0] * num_batches) / np.sum(batch_times) 
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```
:::



::: {.cell .markdown}

#### Summary of results

:::

::: {.cell .code}
```python
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```
:::

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


::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

Also download the `food11.onnx` model from inside the `models` directory.

:::

