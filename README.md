# DeepClarity 🔍


<p align="center">
    <img src="images/logo.gif" alt="Logo" style="border-radius: 35px;">
</p>


**🚀 DeepClarity** introduces novel models designed from scratch to address the dual challenges of **detecting and enhancing pixelated** images with **exceptional speeds**. The proposed detection model, leveraging **MobileNet_v3_small used with Canny edge detection**, demonstrates significant improvements over baseline methods on datasets like **Div2K** and **Flickr2K**, achieving higher precision, recall, F1 score, and accuracy while maintaining a lower false positive rate. Despite a slightly reduced speed compared to the baseline, the model remains efficient, operating at **3489 FPS**. ⚡

For image correction, the project introduces **MobileSR**, a super resolution model based on SRGAN with modified generator architecture, showcasing enhanced performance in terms of PSNR, SSIM, and LPIPS metrics compared to traditional methods like Bicubic and other deep learning approaches. MobileSR strikes a balance between visual quality and computational efficiency, operating at **28 FPS** with a compact model size of **0.482 MB**. 🖼️✨

Both models were implemented entirely from scratch and are hosted on HuggingFace for accessibility, although optimal performance is observed with GPU-based execution. The project's success underscores the potential for practical deployment in real-time applications where rapid image enhancement and high fidelity are paramount. 🔧🎉

<p align="center">
    <img src="images/sr_closeup_transition.gif" alt="Logo" style="border-radius: 5px;">
</p>


## :fire: Try it yourself!!!! :fire:

The models are hosted on HuggingFace :hugs:, but please note that their speed might not be optimal since they are running on CPUs (free tier).

**Detection Model:** [IntelAI Detection](https://huggingface.co/spaces/snehilchatterjee/intelAI_detection)

**Correction Model:** [IntelAI Correction](https://huggingface.co/spaces/snehilchatterjee/intelAIcorrect)

## :stop_sign: IMPORTANT!!!!!!!

When running **locally**, ensure you use only the versions of libraries mentioned in `requirements.txt`. I've observed significant and incorrect differences in results when using different versions.

## :bookmark_tabs: Table of Contents
- [Results](#results)
- [Model Architecture](#architecture)
- [Requirements](#requirements)
- [Training and Testing Details](#training-and-testing-details)
- [Inference: Detection](#inference-detection)
- [Inference: Correction](#inference-correction)
- [In Detail](#in-detail)
- [Future Work](#future-work)
- [Contributing](#contributing)


## :star: Results

**The performance of these models was tested on RTX 3060 Mobile GPU.**

###  <ins>Detection Results: </ins>

### MobileNet_v3_small + Canny Edge Detection

**Datasets used for testing:**

- Div2K (Full dataset - 900 images)
- Flickr2K (Test split - 284 images)

**Performance:**

The baseline model was not evaluated on the Div2K dataset due to its poor performance on the Flickr2K validation/test set.

### Comparison of Proposed Method vs Baseline


#### Metrics on Flickr2K's test set

| Metric           | Proposed Method on Flickr2K | Baseline on Flickr2K  |
|------------------|-----------------------------|-----------------------|
| **Precision**    | 0.944                       | 0.5648                |
| **Recall**       | 0.9007633                   | 0.4326                |
| **F1 Score**     | 0.921875                    | 0.4899                |
| **Accuracy**     | 0.9300595                   | 0.5556                |
| **False Positives** | 4.58%                    | 32.867%               |
| **Speed**        | 3489 FPS                    | 3951 FPS              |
| **Model Size**   | 5.844 MB                    | 5.844 MB              |

### Confusion Matrices:


<p float="left">
   <h3>Proposed Method:</h3>
  <img src="images/cf_detection_method2.png" width="800" />
  <h3>Baseline:</h3>
  <img src="images/cf_detection_method1.png" width="800" />
</p>


#### Metrics on Div2K

| Metric           | Proposed Method on Div2K |
|------------------|--------------------------|
| **Precision**    | 0.9084967                |
| **Recall**       | 0.9084967                |
| **F1 Score**     | 0.9084967                |
| **Accuracy**     | 0.9046053                |
| **False Positives** | 9.52%                 |
| **Speed**        | 3489 FPS                 |
| **Model Size**   | 5.844 MB                 |

### Confusion Matrix:

![cf_method2_test](images/cf_detection_method2_test.png)

### Summary

The proposed method outperforms the baseline significantly across all evaluation metrics on the Flickr2K dataset. It achieves higher precision, recall, F1 score, and accuracy, while maintaining a much lower false positive rate. Although the proposed method has a slightly slower speed (3489 FPS vs. 3951 FPS), it is still extremely efficient and the trade-off is justified by the substantial improvements in other metrics. The model size remains consistent across both methods. 

Overall, the proposed method demonstrates superior performance and is a clear improvement over the baseline, especially in terms of accuracy and reliability.

###  <ins>Correction Results: </ins>

### Super Resolution with MobileSR

**Method:**

- SRGAN with a modified generator incorporating depthwise separable convolutions in residual blocks.
- Bilinear upsampling for improved performance in the final stage.

**Performance:**


### Comparison of Image Super-Resolution Methods (Evaluated on Sample2.jpeg)

| Metric            | Bicubic (Baseline)      | Proposed Method (MobileSR) | RealESRGAN               | FSRCNN                   | EDSR                     |
|-------------------|-------------------------|----------------------------|--------------------------|--------------------------|--------------------------|
| PSNR              | 27.23 dB                | 28.95 dB                   | 27.21 dB                 | 29.09 dB                 | 29.19 dB                 |
| SSIM              | 0.6684                  | 0.7582                     | 0.7359                   | 0.7604                   | 0.7657                   |
| LPIPS             | 0.2878                  | 0.3794                     | 0.4633                   | 0.5985                   | 0.6461                   |
| Speed             | -                       | 28 FPS                     | <1 FPS                   | 188 FPS                  | 16 FPS                   |
| Model Size        | -                       | 0.482 MB                   | 63.698 MB                | 0.049 MB                 | 5.789 MB                 |
| Mean Opinion Score | 3.19                    | 4.00                       | 4.31                     | 3.06                     | 3.39                     |


Total opinions taken: 16

Mean Opinion Score Plot:

![MOS](images/mean_ratings_plot_beautiful.png)

Mean Opinion Score Results: [Results](https://docs.google.com/spreadsheets/d/1gVwDEkZqQh_vei8I5QgR4UZ0rX_byFTNz3BPfMHUns8/edit?usp=sharing)

## Summary:

In summary, RealESRGAN excels in visual quality but suffers from artificial looks and impractically large model size. FSRCNN is extremely fast with a tiny model size but sacrifices visual quality. EDSR falls short in visual quality and speed requirements. The proposed method (MobileSR) strikes a balance by offering good visual quality close to RealESRGAN, with no artificial looks, optimal model size for embedded systems, and sufficient speed for real-time applications, making it a strong candidate for practical implementations where a blend of performance and efficiency is crucial.


**Sample Output:**

***1) (480 x 320) -----> (1920 x 1280)***

<p float="left">
  <img src="images/Labeled_Cropped/Upscaled%20Image%20914x609_Input.png" width="400" />
  <img src="images/Labeled_Cropped/Upscaled%20Image%20914x609_MobileSR.png" width="400" />
  <img src="images/Labeled_Cropped/Upscaled%20Image%20914x609_FSRCNN.png" width="400" />
  <img src="images/Labeled_Cropped/Upscaled%20Image%20914x609_EDSR.png" width="400" />
  <img src="images/Labeled_Cropped/Upscaled%20Image%20914x609_RealESRGAN.png" width="400" />
  <img src="images/Labeled_Cropped/GroundTruth.png" width="400" />
</p>

***2) Top Row: Input, Middle Row: Output, Bottom Row: Target***

![Super Resolution Result](images/sr_result.png)
Close up:
![Close-up Super Resolution Result](images/sr_closeup.png)


## Architecture

### Detector Architecture:

![detector](images/detector_architecture_white.jpeg)

### MobileSR Architecture:

![mobilesr](images/generator_architecture_white.jpeg)


## Requirements

Install the necessary libraries:
```sh
pip install -r requirements.txt
```

## Training and Testing Details

#### Detector Training
The detector was trained on the train split of the Flickr2K dataset, which consists of 2,200 images.

#### Detector Testing
The detector was tested in two phases:
1. Test split of the Flickr2K dataset, consisting of 284 images.
2. The full dataset of Div2K (train + val) to ensure the images were entirely independent of the trained dataset.

#### Super Resolution Model Training
The super resolution model was trained on a subset of the [COCO dataset](https://cocodataset.org/), using a total of 21,837 images.

#### Super Resolution Model Testing
The testing of the super resolution model was conducted using 166 randomly picked images from the Flickr2K dataset.


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
|
├── detection_method 2 [Pixelated].ipynb  - Training notebook file for detection method 2 (proposed method)
│ 
|   
├── Training_Correction.ipynb  - Training notebook file for correction method (proposed method)
│
|    
├── Testing_Correction_Result.ipynb  - Notebook file used to obtain images/sr_result.png and images/sr_closeup.png
│
|
├── test_detect.ipynb  - Test notebook file for detection method 2 (proposed method)
│
|
├── test_correct.ipynb  - Test notebook file for correction
│
|
├── time_calculation.ipynb  - Time measurement notebook for the detection method and the correction method
│ 
|
├── model_size.ipynb  - Model size measurement notebook for the detection method and the correction method
│
| 
├── experiment_detection  
│   └── comparison_n  - Contains comparison between low_res and high_res image
│   └── solo_n        - Contains low_res version of sample images
│  
| 
├── images             - Contains readme.md images
│ 
| 
├── detect_app.py       - Detection inference app.py file
│
|
├── correct_app.py       - Correction inference app.py file
```

## Future Work

June 21, 2024 : <s> Obtaining Mean Opinion Score and calculating other metrics for super-resolution tasks. </s>

Making a pipeline which will detect and correct in the same app

## Contributing

Any kind of enhancement or contribution is welcomed.