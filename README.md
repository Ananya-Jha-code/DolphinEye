# Dolphin-Eye for vistually impaired

## Inspiration
With just little aid to the visually impaired people existing currently, there is a need to implement a device that helps them in their daily activities. The existing systems such as Screen Reading software and Braille devices help visually impaired people to read and have access to various gadgets but these technologies become useless when the blind wants to carry out basic tasks, which involve detecting the scene in front of him, for instance, the people or objects. Our system will facilitate visually impaired people around the globe. Dolphin-Eye is made with an aim to help a person with detecting the object in front even with impaired eyesight and without the need of a guardian which leads to detecting objects and people in the frame of vision and notifying the user about the same. We present a method, which uses object detection on the live stream of videos. The resultant object or person is then transmitted to the impaired person in the form of signal. The goal is to provide inexpensive solution to the visually impaired and make their life better and self-sufficient.

## Dataset
We have used the pretrained weights of the Microsoft COCO (Microsoft Common Objects in Context) dataset which is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.

## Model Components
The architecture used by us is YOLOv5s. YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, and includes simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, and export to ONNX, CoreML and TFLite. The github repository to understand the model is [here](https://github.com/ultralytics/yolov5). The demo output can be viewd [here](https://colab.research.google.com/drive/1AiuBDleOUM5Vyq3Itq3edCjg5J8I719p?usp=sharing).

The three main parts of our architecture are:
- Model Backbone - CSPNet are used as a backbone to extract rich informative features from an input image
- Model Neck - Model Neck is mainly used to generate feature pyramids. PANet is used for as neck to get feature pyramids
- Model Head - The model Head is mainly used to perform the final detection part. It applied anchor boxes on features and generates final output  vectors with class probabilities, objectness scores, and bounding boxes.

The three main works of our project are:
- Object Detection using the YOLOv5 architecture
- Calculating the depth of the objects
- Sending messages or singals once encountered by object in a certain distance

## Implementation details
 - The zip files containing images and reports were mounted from Google Drive.
 - A dataset was prepared through a data cleaning process that consists of two images per report, one frontal and one lateral view.
 - Reports were extraced from .xml files and the frontal and lateral views were combined to prepare the above mentioned dataset and this was used to generate features.
 - glove.840B.300d was used for obtaining vector representations and generating the embedding matrix. It is available [here](https://nlp.stanford.edu/projects/glove/).
   - To run the model, download the glove file and add to MedGen folder.
 - Features were extracted using DenseNet121 model loaded with ChexNet weights (available [here](https://www.kaggle.com/theewok/chexnet-keras-weights)). The paper used a VGG-19 network.
   - The features are available in ./features directory. 
 - The features were fed into a model with the following structure
<p align="center">
<img src="https://github.com/01pooja10/Medical-Report-Generator/blob/main/misc/attn_mod.jpg" height="400" alt="Model structure">
 
 - To train the model, run *encoder_decoder.ipynb* in root directory.
 
 ## Generated Report
- The model was trained for 10 epochs. Due to computational difficulties, we were unable to train for more epochs and hence the model did not converge. 
- Final BLEU score was 0.643
 <p align="center">
<img src="https://github.com/01pooja10/Medical-Report-Generator/blob/main/misc/generated_report.png" height="300" alt="Generated Report">

## Installation and Quick Start
The code by default will only run on a video stream.
To use the repo and run inferences, please follow the guidelines below:

- Cloning the Repository: 

        $ git clone https://github.com/simran29aw/dolphin-eye.git
        
- Entering the directory: 

        $ cd dolphin-eye/
        
- Setting up the Python Environment with dependencies:

        $ pip install -r requirements.txt

- Running the file for inference:

        $ python inference.py

## Dependencies
- PyTorch
- Numpy
- Pandas 
- PIL 8.0.1
- Nltk 3.5
- Matplotlib 3.3.2
- Opencv 4.5.1
- Tqdm 4.50.2
- OS

## Contributors 
- [Simran Agarwal](https://github.com/simran29aw)
- [Ananya Jha](https://github.com/Ananya-Jha-code)
- [Sashrika Surya](https://github.com/sashrika15)

