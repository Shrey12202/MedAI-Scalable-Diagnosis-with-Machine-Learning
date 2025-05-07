

::: {.cell .markdown}

## Prepare data

For the rest of this tutorial, we'll be training models on the [Food-11 dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/). We're going to prepare a Docker volume with this dataset already prepared on it, so that the containers we create later can attach to this volume and access the data. 

:::


::: {.cell .markdown}

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

:::