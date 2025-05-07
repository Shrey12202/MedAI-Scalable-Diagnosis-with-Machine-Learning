
::: {.cell .markdown}

# Model optimizations for serving

In this tutorial, we explore some model-level optimizations for model serving:

* graph optimizations
* quantization
* and hardware-specific execution providers, which switch out generic implementations of operations in the graph for hardware-specific optimized implementations

and we will see how these affect the throughput and inference time of a model.

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@UC and CHI@TACC sites.

:::


::: {.cell .markdown}

## Context


The premise of this example is as follows: You are working as a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food. You have developed a convolutional neural network in Pytorch that automatically classifies photos of food into one of a set of categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.

Now that you have trained a model, you are preparing to serve predictions using this model. Your manager has advised that since GourmetGram is an early-stage startup, they can't afford much compute for serving models. Your manager wants you to prepare a few different options, that they will then price out among cloud providers and decide which to use:

* inference on a server-grade CPU (AMD EPYC 7763). Your manager wants to see an option that has less than 3ms median inference latency for a single input sample, and has a batch throughput of at least 1000 frames per second.
* inference on a server-grade GPU (A100). Since GourmetGram won't be able to afford to load balance across several GPUs, your manager said that the GPU option must have strong enough performance to handle the workload with a single GPU node: they are looking for less than 1ms median inference latency for a single input sample, and a batch throughput of at least 5000 frames per second.
* inference on end-user devices, as part of an app. For this option, the model itself should be less than 5MB on disk, because users are sensitive to storage space on mobile devices. Because the total prediction timme will not include any network delay when the model is on the end-user device, the "budget" for inference time is larger: your manager wants less than 15ms median inference latency for a single input sample on a low-resource edge device (ARM Cortex A76 processor).

You're already off to a good start, by using a MobileNetV2 as your foundation model; this is a small model that is especially designed for fast inference time. Now you need to measure the inference performance of the model and, if it doesn't meet the requirements above, investigate ways to improve it.

:::

::: {.cell .markdown}

## Experiment resources 

For this experiment, we will provision one bare-metal node with a recent NVIDIA GPU (e.g. A100, A30). (Although most of the experiment will run on CPU, we'll also do a little bit of GPU.)

We'll use the `compute_liqid` node types at CHI@TACC, or `compute_gigaio` node types at CHI@UC. (We won't use `compute_gigaio` nodes at CHI@TACC, which have a different GPU and CPU.)

* The `compute_liqid` nodes at CHI@TACC have one or two NVIDIA A100 40GB GPUs, and an AMD EPYC 7763 CPU.
* The `compute_gigaio` nodes at CHI@UC have an NVIDIA A100 80GB GPU, and an AMD EPYC 7763 CPU.

You can decide which type to use based on availability.

:::

::: {.cell .markdown}

## Create a lease for a GPU server

:::

::: {.cell .markdown}

To use bare metal resources on Chameleon, we must reserve them in advance. For this experiment, we will reserve a 3-hour block on a bare metal node with GPU.

We can use the OpenStack graphical user interface, Horizon, to submit a lease. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/)
* click "Experiment" > "CHI@TACC" or "Experiment > CHI@UC", depending on which site you want to make reservation at
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.

:::

::: {.cell .markdown}

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

:::

::: {.cell .markdown}

Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.

:::

::: {.cell .markdown}

## At the beginning of your GPU server lease


At the beginning of your GPU lease time, you will continue with the next step, in which you bring up and configure a bare metal instance! To begin this step, open this experiment on Trovi:

* Use this link: [Model optimizations for serving machine learning models](https://chameleoncloud.org/experiment/share/f5acccf8-f2cb-4d1e-8918-4c8fd97bfc32) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it, including the notebok to bring up the bare metal server.


:::

