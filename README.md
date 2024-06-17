# DeepClarity

This project tackles the challenge of **detecting and enhancing** pixelated images at exceptional speeds.

## Requirements

Install the necessary libraries:
```sh
pip install -r requirements.txt
```

## Table Of Contents
- [Results](#results)
- [Inference: Detection](#inference-detection)
- [Inference: Correction](#inference-correction)
- [In Detail](#in-detail)
- [Contributing](#contributing)

## Results

**The performance of these models was tested on RTX 3060 Mobile GPU.**

### MobileNet_v3_small + Canny Edges

**Datasets used for testing:**

- Div2K (Full dataset - 900 images)
- Flickr2K (Test split - 284 images)

**Performance:**

I didn't evaluate the baseline model on the Div2K dataset because it was already performing very poorly in the validation set/test set of Flickr2K.

**Proposed Method:**

| Metric           | Div2K         | Flickr2K       |
|------------------|---------------|----------------|
| Precision        | 0.9084967     | 0.944          |
| Recall           | 0.9084967     | 0.9007633      |
| F1 Score         | 0.9084967     | 0.921875       |
| Accuracy         | 0.9046053     | 0.9300595      |
| False Positives  | 9.52%         | 4.58%          |
| Speed            | 3489 FPS      | 3489 FPS       |
| Model Size       | 5.844 MB      | 5.844 MB       |

**Baseline:**

| Metric           | Baseline on Flickr2K  |
|------------------|-----------------------|
| Precision        | 0.5648                |
| Recall           | 0.4326                |
| F1 Score         | 0.4899                |
| Accuracy         | 0.5556                |
| False Positives  | 32.867%               |
| Speed            | 3951 FPS              |
| Model Size       | 5.844 MB              |

### Super Resolution with Modified SRGAN

This section describes a super-resolution approach using a modified SRGAN architecture.

**Method:**

- SRGAN with a modified generator incorporating depthwise separable convolutions in residual blocks.
- Bilinear upsampling for improved performance in the final stage.

**Performance:**

| Metric        | Proposed Method         | Bicubic (Baseline)      |
|---------------|-------------------------|-------------------------|
| PSNR          | 28.95 dB                | 27.23 dB                |
| SSIM          | 0.7582                  | 0.6684                  |
| LPIPS         | 0.3794                  | 0.2878                  |
| Speed         | 28 FPS                  | -                       |

**Sample Output:**

Top Row: Input, Middle Row: Output, Bottom Row: Target

![Super Resolution Result](images/sr_result.png)
![Close-up Super Resolution Result](images/sr_closeup.png)

## Inference: Detection

To run the detection app, use the following command:
```sh
python detect_app.py
```

You will see the following output:

![Terminal Output](images/detect_terminal.png)

Open the provided link in any web browser to access the interface:

![Detection Interface](images/detect.png)

Select the method and proceed with uploading an image to detect if it is pixelated.

**Methods:**
- Method 1: MobileNetV3_Small
- Method 2 (Proposed method): MobileNetV3_Small + Canny Edge Detection

Example:
![Select Method](./images/detect_method_select.png)

The result will appear as soon as you upload the image:
![Detection Result](./images/detect_result.png)

## Inference: Correction 

To run the correction app, use the following command:
```sh
python correct_app.py
```

You will see the following output:

![Terminal Output](images/correct_terminal.png)

Open the provided link in any web browser to access the interface:

![Correction Interface](images/correct.png)

Upload an image and click submit!

Example:
![Correction Result](./images/correct_result.png)

## In Detail

```
├── detection_method 1 [Pixelated].ipynb  - Training notebook file for detection method 1 (baseline)
│ 
│ 
│ 
├── detection_method 2 [Pixelated].ipynb  - Training notebook file for detection method 2 (proposed method)
│    
│
│
├── test_detect.ipynb  - Test notebook file for detection method 2 (proposed method)
│
│
├── test_correct.ipynb  - Test notebook file for correction 
│
│
├── time_calculation.py  - Time measurement notebook for the detection method and the correction method
│ 
│
├── experiment_detection  
│   └── comparision_n  - Contains comparison between low_res and high_res image
│   └── solo_n         - Contains low_res version of sample images
│   
│
├── images             - Contains readme.md images
│  
│
│
├── detect_app.py       - Detection inference app.py file
│
│
├── correct_app.py       - Correction inference app.py file
```

## Future Work

Obtaining Mean Opinion Score and calculating other metrics for super-resolution tasks.

## Contributing

Any kind of enhancement or contribution is welcomed.
