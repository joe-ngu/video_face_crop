# Video Face Crop

This Python script utilizes OpenCV to automatically detect and zoom in on faces within a video.

## Description

The script is designed to analyze a video file, detect faces using the Haar cascade classifier provided by OpenCV, and zoom in on each detected face. It uses computer vision techniques to identify facial features and adjust the video frame to focus on the detected faces.

## Features

- Automatic face detection using Haar cascade classifier.
- Dynamic zooming to focus on detected faces within the video.

## Usage

1. Set up your environment and install dependencies. 

For example:

```bash
$ python3 -m venv venv/
$ source
$ pip install -r requirements.txt
```

2. In the local repo, create a folder named `videos` if you don't have one and add a video file there. The script uses the first video there by default.

<!-- Empty line -->
3. Run the script

```bash
$ python3 main.py
```

