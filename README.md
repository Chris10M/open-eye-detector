# open-eye-detector
A simple opencv script to detect if the eyes are open or closed

### Introduction
A resnet inspired CNN model to predict if a person's eye is open or closed.

The main use case for this, is while in face recognition we have to ensure the person remains active and not asleep while doing the auth process, so we have trained the model to ouput closed eyes and some-what drowsy eyes.


### Contents
1. [Installation](#installation)
2. [Demo](#demo)

### Installation

**Requirements**
  1) Keras
  2) Tensoflow
  3) OpenCV
  4) dlib
  
  just use pip3 

### Download
A **Pre-trained test model** is available at this Link [GoogleDrive](https://drive.google.com/open?id=1EC98US1ck0wF0lDE4HFjySxzlWCv75jh), dowload it and place it in models/ path.

### Demo
If you've cloned the repo and then downloaded the pre-trained model, run 
```
./dependencies/install_dependencies.sh
./startup.sh
```

Now you could acess the websever using **POST** Reqeust using the endpoint, 
```
http://0.0.0.0:52118/test-tensorflow-serving-api
```
with content type as "application/json"
```
{
  "url": "some-image-url"
}
```
or if you want to give a base64Image input then,
```
{
  "base64Image": "base64_image_data"
}
```
to facilitate a response.

### Issues
If you encounter any issues, please create an issue tracker.


