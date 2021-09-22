# Dolphin-Eye for vistually impaired

## Inspiration
With just little aid to the visually impaired people existing currently, there is a need to implement a device that helps them in their daily activities. The existing systems such as Screen Reading software and Braille devices help visually impaired people to read and have access to various gadgets but these technologies become useless when the blind wants to carry out basic tasks, which involve detecting the scene in front of him, for instance, the people or objects. Our system will facilitate visually impaired people around the globe. Dolphin-Eye is made with an aim to help a person with detecting the object in front even with impaired eyesight and without the need of a guardian which leads to detecting objects and people in the frame of vision and notifying the user about the same. We present a method, which uses object detection on the live stream of videos. The resultant object or person is then transmitted to the impaired person in the form of signal. The goal is to provide inexpensive solution to the visually impaired and make their life better and self-sufficient.

## Dataset
We have used the [pretrained weights](https://github.com/ultralytics/yolov5) of the [Microsoft COCO dataset](https://cocodataset.org/)(Microsoft Common Objects in Context)  which is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.

## Model Components
The architecture used by us is YOLOv5s. YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, and includes simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, and export to ONNX, CoreML and TFLite. The github repository to understand the model is [here](https://github.com/ultralytics/yolov5). 

The three main parts of our architecture are:
- Model Backbone - CSPNet are used as a backbone to extract rich informative features from an input image
- Model Neck - Model Neck is mainly used to generate feature pyramids. PANet is used for as neck to get feature pyramids
- Model Head - The model Head is mainly used to perform the final detection part. It applied anchor boxes on features and generates final output  vectors with class probabilities, objectness scores, and bounding boxes.

![The-network-architecture-of-Yolov5-It-consists-of-three-parts-1-Backbone-CSPDarknet](https://user-images.githubusercontent.com/72155378/134271959-55ad63a4-ef1a-40fc-9c04-9e2369e19aa3.jpg)

The three main tasks of our project are:
- Object Detection using the YOLOv5 architecture
- Calculating the depth of the objects
- Sending warning messages in case object is too close

## Implementation details
- Object Detection using the YOLOv5 architecture
   - Model implemented is YoloV5s, which is the smallest version of YoloV5. 
   - Model architecture is defined in a [YAML file](models/yolov5s.yaml), which clearly mentions all the layers and their arguments. 
   - All the model blocks are defined in the files in model folder, along with the main yolo.py file which parses the yaml and creates the model.
   - This implementation loads in the pretrained weights from YoloV5 repo.
- Calculating the depth of the objects
   - The inspiration for depth calculation was taken from [here](https://ieeexplore.ieee.org/document/9234074)
   - It essentially utilises a focal distance relationship to calculate how far an object is.
- Sending warning messages in case object is too close
   - In case the distance falls below a threshhold, we've utilised the google tts API to make an mp4 of a simple warning (STOP) which is played back to the user. 
- [NOTE] This implementation works ONLY for a real time stream
 
## Installation and Quick Start
The code by default will only run on a video stream.
To use the repo and run inferences, please follow the guidelines below:

- Cloning the Repository: 

        $ git clone https://github.com/Ananya-Jha-code/dolphin-eye.git
        
- Entering the directory: 

        $ cd dolphin-eye/
        
- For running on CLI, use the inference file as follows:

        $ python inference.py
        

## Demo
We can see how the distance of the chair in bottom left corned goes from around 7000 to around 5000. Based on a threshhold, the application will issue a warning if the distance is less than the threshhold.

<img src="misc/demo.gif" width="800">

## Dependencies
- PyTorch
- Numpy
- Pandas 
- Opencv
- PIL

## Contributors 
- [Simran Agarwal](https://github.com/simran29aw)
- [Ananya Jha](https://github.com/Ananya-Jha-code)
- [Sashrika Surya](https://github.com/sashrika15)

