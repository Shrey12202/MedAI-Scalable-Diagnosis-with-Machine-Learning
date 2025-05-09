
::: {.cell .markdown}

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

:::

::: {.cell .markdown}

### Start Label Studio

First, let's start Label Studio, our tool for managing the tasks assigned to human annotators. These humans will assign ground truth labels to production data - in this case, images that have been submitted by "real" users to our service - so that we can monitor the performance of our model in production. 


Run

```bash
# runs on node-eval-loop
docker compose -f eval-loop-chi/docker/docker-compose-labelstudio.yaml up -d
```

Note that Label Studio is now running *in addition to* the Flask, FastAPI, and MinIO services we started in the previous section.

:::

::: {.cell .markdown}

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

:::

::: {.cell .markdown}

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


:::