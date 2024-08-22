# yolov5_VR
This repo does object detection on the MNIST Double Digit RGB (MNISTDD-RGB) dataset
As seen in the output, the best.pt model predicts with a very high accuracy over 99% with IOU (Intersection over Union) scores of 95 over%

# classifier.py
This python file loads the pre-trained model and the images, then using a loop, returns the predicted classes and boxes for the digit classification.

# main.py
This is the file that calculates all of the classification and IOU scores, also keeps track of all of the time taken to run, which shows the accuracies and iou scores of the model.
