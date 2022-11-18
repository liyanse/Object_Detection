This repository will cover my journey in handling the Object Detection model for a dataset provided by the Center of Data Science and Artificial Intelligence Lab (DSAIL) known as the DSAIL Porini. The dataset containss over 8000 images of animals from the Dedan Kimathu University conservancy of animals in 6 classes. Impala, zebras, waterbucks, bushbucks, monkeys and warthogs. The lab has already done an image classification model on the dataset. I am going to handle object detection with YOLOV5 and if there is enough time, Tensorflow.
We'll walk through collecting the images for the test and train sets, the set should be balanced, we'll walk through the instance of every class, remember the dataset should we have a lot of variety, the annotations of your labels must be consistent, we'll walk through random splitting, creating the yaml file and also dividing the set into the images and label folders.
We'll be working with three classes from the six in the conservancy
Introduction
In the field of artificial intelligence, object detection has established itself as a household term. Devices that unlock phones using facial recognition, self-driving cars, and picture search capabilities are just a few examples. Object detection is a subset of computer vision that is gradually enabling machines to get visual assistance in the outside world. Then, how do you begin using object detection? The first step is to classify the objects using image classification. This entails teaching a computer model to categorize particular real-world items into different groups. For instance, a person, animal, car, plane etc.

You might be wondering what the distinction is between object localization and object detection, which would be the second stage. Basically, object localization involves teaching a computer model to recognize the existence of a single object in a given image and to determine its location. In contrast, object detection involves teaching a model to identify and locate various items in an image or video.

How to practice object localization
Consider a situation where we are attempting to detect two different item types, a bicycle and a car. We will feed an image to either our ResNet, VGG, or CVV neural networks. The two classes are then predicted by the network using bounding boxes and a prediction confidence score.

Object Localization

A rectangular rectangle known as a bounding box surrounds an object and specifies its position class and confidence. There are two techniques for producing a bounding box;

The first step is to construct the numbers x1, x2, y1 and y2 where (x1, y1) and (x2, y2) represent the upper left and bottom right corner points, respectively.
Create two points (x, y) to indicate the image's corner points and two points (h, w) to indicate the object's height and breadth.
We'll discuss how to build bounding boxes for object detection models using the YOLOv5 format in a another session. As we can see, object localization just requires one class per image, therefore it does not require much.

Approaches to achieving object detection
With the advancement of technology, more and more methods were developed to make it easier to achieve object detection. These include;
- Sliding windows

Sliding Windows

Working with Sliding Windows is one of the original methods of object detection. Here we create a bounding box, which is normally a square and use this box to resize the image into numerous crops, trying to see if there is an instance of the classes being detected. In our above image, there is a sliding window that tries to see if the car appears in various crops of the image. This method is tedious, it involves a lot of computation and as seen, there are a lot of bounding boxes being created for the same object
- Regional based network

Regional Based network

Consider a scenario in which we are attempting to develop a model to determine if an object is a human or an animal and we have an image of a man riding a bicycle in a park. For each object, we will draw bounding boxes, extract all potential regions, compute CNN features, and then finally classify the regions.

- YOLO (You Only Look Once)

YOLOv5

YOLO is a pretrained model in pyTorch that is used for object detection. It is based on regression, that is instead of selecting the interesting part of an image, it predicts classes and bounding boxes for the whole image in one run of the algorithm. Yolo uses a single CNN to do the object detection as well as localization which makes it faster than R-CNN.
Yolo divides all the given input images into the S * S grid system and each grid is responsible for object detection. Each cell is going to be predicting if there’s a bounding box in that cell and what confidence the box has an predicting a certain class. For every box, we have 5 main attributes;
• X and y co-ordinates for the corner points
• The height and width of the object
• The confidence score for the probability that the box is containing a certain image
