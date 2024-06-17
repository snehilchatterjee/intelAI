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
-  [Solution Architecture](#solution-architecture)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

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
| Speed         | 2244 FPS       | 2244 FPS                |
| Model Size     | 5.844 MB       | 5.844 MB                |

### Super Resolution with Modified SRGAN

This section describes a super-resolution approach using a modified SRGAN architecture.

**Method:**

* SRGAN with a modified generator incorporating depthwise separable convolutions in residual blocks.
* Bilinear upsampling for improved performance at the final stage.

**Performance:**

* PSNR: 0.27
* Speed: 20 FPS (4x super-resolution to FHD image)

**Sample Output:**


Top Row is input, 
Middle Row is output, 
Bottom Row is target


![alt text](images/sr_result.png)


**Note:** The image link points to a sample output showcasing the input, generated image, and target image for comparison.


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
├──  config
│    └── defaults.py  - here's the default config file.
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
│
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
│   
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
```


# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.


# Acknowledgments



