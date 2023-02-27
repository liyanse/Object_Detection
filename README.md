## Object Detection with YOLOv5 
# Introduction
When we think of object detection, our mind automatically jumps to tesla, CCTV cameras, smart doors and facial recognition. But there is more that can be handled with object detection, and we used this as one of our projects here at DSAIL. The project kicked off around September and our first step was to select the classes we would use to train from the dataset that contained over 6000 images.After  going through the dataset we realized there were more photos of zebras and impalas than of any other animal. from the total dataset of over 600 images, 700 images were selected to train the model. 
YOLOv5 (You Only Look Once) is a popular object detection algorithm. It uses a single neural network to predict bounding boxes and class probabilities for objects in an image. It is known for its high accuracy and speed, making it a popular choice for applications such as surveillance, autonomous driving, and image recognition.
To use YOLOV5 for object detection, the first step is to train the model on a dataset of labeled images. This involves feeding the model with a large number of images and their corresponding labels, such as "impala" or "zebra", to teach it to recognize and classify different objects. Once the model is trained, it can be used to detect objects in new images or videos.
To detect objects, YOLOV5 uses a combination of convolutional layers and anchor boxes to identify the location and class of objects in an image. It first divides the image into a grid of cells, and each cell is responsible for detecting objects within a certain region of the image. The model then uses anchor boxes to predict the coordinates and class of the objects within each cell.
# Data Preprocessing
# Labeling Images
The main process when working with data preprocessing is labeling the images correctly based on their classes. During training, YOLO goes through every image and its corresponding label file. It also creates a new file with all the classes of the object you are training on. There are various ways to label the images with, in our case we used a program called LabelImg. The application allows you to draw a bounding box around the object present in the image and also define the class of the object. In our case, we did this for all 400 images we used to train the model with. It takes about 3 minutes to process an image and ensure the bounding box is well plotted around the image with no extra noise around it. Another important step is to ensure you select the type of labeling as YOLO’s labeling. This is because labelimg provides labeling for images for more object detection thus important to declare which one you are using. Below is a sample of the data processing done for all the images used to train our object detection model.
An important note to take home is that when preparing the dataset, your main folder must contain two other folders inside. For example, we could have a main folder called “Object-Detection” and inside it we have the “images” and “labels” folder. The “images” folder will contain all your training images and their corresponding labels will be stored in the “labels” folder. Your labels folder will have one extra file that is not present in your train folder.
# Dataset.yaml
For most steps in  computer vision, while writing the code we always tell the program the location of our training and testing dataset. If that’s too complicated think about it this way;
In our notebooks, we define the path to the dataset, the train set and the test set and then the notebook automatically picks up the classes present in the dataset.

However, when working with YOLO, we reduce all this code and create a new file that contains all this information. The dataset.yaml file contains the location of your dataset, your test and train path, the number of classes you have and the label classes. Here is a snippet of how your dataset.yaml file should look like;

# Training the Model
Ultralytics, the company responsible for writing the YOLOv5 model provides a step to step guide on training an object detection model with a custom dataset. This documentation is the same we used on our model. The documentation is present in their GitHub and on Training on Custom Object Dataset.
Inferencing the Model
After successfully training, we saved the model and used it to test on images and videos that were not in our dataset. Here is a snippet of our output when inferencing the model on new images.

The model we trained was not only used to classify the images of the animals from the conservancy, it was used to annotate the images automatically. Initially, it was tedious and time consuming to annotate the images with human effort. When used to annotate the whole dataset, the model took about an hour. This improves the output and reduces the manpower used in processing raw data.
We hope to use the model in our camera traps to detect and annotate the animals and also study their behavior before the dataset is stored in the servers.

# What models are we building?
For this specific repository, I'm going to be building a lot of models for object detection
We start with the Animal Class recognition, then the fruit class recognition



