

::: {.cell .markdown}

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

:::

