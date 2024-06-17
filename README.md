# DeepClarity
This project tackles the challenge of **detecting and enhancing** pixelated images at exceptional speeds.


# Requirements
Install the necessary libraries
```sh
pip install -r requirements.txt
``` 

# Table Of Contents
-  [Results](#results)
-  [Inference: Detection](#inference-detection)
-  [Inference: Correction](#inference-correction)
-  [In detail](#in-detail)
-  [Contributing](#contributing)

# Results

**The performance of these models was tested on RTX 3060 Mobile GPU**

### MobileNet_v3_small + Canny Edges

**Datasets used of testing:**

* Div2K (Full dataset - 900 images)
* Flickr2K (Test split - 284 images)

**Performance:**

| Metric        | Div2K         | Flickr2K        |
|---------------|----------------|-----------------|
| Precision     | 0.9084967     | 0.944            |
| Recall        | 0.9084967     | 0.9007633        |
| F1 Score       | 0.9084967     | 0.921875         |
| Accuracy      | 0.9046053     | 0.9300595        |
| False Positives | 9.52%          | 4.58%           |
| Speed         | 3489 FPS       | 3489 FPS                |
| Model Size     | 5.844 MB       | 5.844 MB                |

### Super Resolution with Modified SRGAN

This section describes a super-resolution approach using a modified SRGAN architecture.

**Method:**

* SRGAN with a modified generator incorporating depthwise separable convolutions in residual blocks.
* Bilinear upsampling for improved performance at the final stage.

**Performance:**

* PSNR: 0.27
* Speed: 28 FPS (4x super-resolution to FHD (1920 x 1080) image)

**Sample Output:**


Top Row is input, 
Middle Row is output, 
Bottom Row is target


![alt text](images/sr_result.png)
![alt text](images/sr_closeup.png)



# Inference: Detection

To run the detection app run the following command:
```sh
python detect_app.py
``` 

You will see the following output:

![alt text](images/detect_terminal.png)

Open the link on any web browser and you will see the following interface:

![alt text](images/detect.png)

Select method and proceed with uploading an image for detecting if it is pixelated.


**Method1: MobileNetV3_Small**

**Method2 (Proposed method): MobileNetV3_Small + Canny Edge Detection**

Example:

![alt text](./images/detect_method_select.png)

You will see the result as soon as you upload the image

![alt text](./images/detect_result.png)


# Inference: Correction 

To run the correction app run the following command:
```sh
python correct_app.py
``` 

You will see the following output:

![alt text](images/correct_terminal.png)

Open the link on any web browser and you will see the following interface:

![alt text](images/correct.png)

Upload an image and hit submit!!!

Example:

![alt text](./images/correct_result.png)





# In Details
```
├──  detection_method 1 [Pixelated].ipynb  - Training notebook file for detection method 1 (baseline)
│ 
│ 
│ 
├──  detection_method 2 [Pixelated].ipynb  - Training notebook file for detection method 2 (proposed method)
│    
│
│
├──  test.ipynb  - Test notebook file for detection method 2 (proposed method)
|
|
├──  time_calculation.py    - Time measurement notebook for the detection method and the correction method
│ 
│
├──  experiment_detection  
│    └── comparision_n  - Contains comparision between low_res and high_res image
│    └── solo_n         - Contains low_res version of sample images
│    
│
├──  images             - Contains readme.md images
│  
│
│
├──  detect_app.py       - Detection inference app.py file
│
│
├──  correct_app.py       - Correction inference app.py file
```


# Future Work

Obtaining Mean opinion score and calculating other metrics for super resolution task

# Contributing
Any kind of enhancement or contribution is welcomed.





