import sys
import argparse

import torch
import torchvision
import PIL
import os
from matplotlib import pyplot as plt
from torchvision.models import mobilenet_v3_small
import cv2
import random
import warnings
import timm
from torch import nn
import numpy as np

warnings.filterwarnings("ignore")

def method1_prep(image):
    transforms = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    image = torch.from_numpy(image).permute(2, 0, 1)
    return transforms(image)
    
def method2_prep(image):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop((224, 224))
    ])
    t_lower = 50  
    t_upper = 150  
    
    img = torch.from_numpy(cv2.Canny(image, t_lower, t_upper)[np.newaxis, ...])
    img = torch.vstack((img, img, img))
    
    return transforms(img.type(torch.float32))

def main():
    parser = argparse.ArgumentParser(description='Detects whether a file is pixelated or not')
    parser.add_argument('file', type=str, help='The location of file to be checked')
    parser.add_argument('method', type=str, help='The method to be used (either 1 or 2)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File {args.file} does not exist.")
        sys.exit(1)

    image = cv2.imread(args.file)
    if image is None:
        print(f"Failed to load image from {args.file}")
        sys.exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('Image loaded successfully')
    
    model = mobilenet_v3_small(weights='DEFAULT')
    model.classifier[3] = nn.Linear(in_features=1024, out_features=2, bias=True)   
    
    if args.method == '1':
        image = method1_prep(image).unsqueeze(dim=0)
        #print(image.shape)
        model.load_state_dict(torch.load('./method1(0.974).pt'))
        print("\nModel weights loaded successfully")
        
        model.eval()  # Set the model to evaluation mode
        
        with torch.inference_mode():
            model = model.to(device)
            image = image.to(device)
            output = torch.softmax(model(image), dim=1).detach().cpu()
            prediction = torch.argmax(output, dim=1).item()
            #print(f'Output: {output.numpy()}')
            #print(f'Prediction: {prediction}')
            if prediction == 0:
                print("The image is not pixelated")
            else:
                print("The image is pixelated")
        
    elif args.method == '2':
        image = method2_prep(image).unsqueeze(dim=0)
        #print(image[0].permute(1, 2, 0).shape)
        
        image_np = image[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)  # Ensure the image is of type uint8
        
        model.load_state_dict(torch.load('./method2(0.990).pt'))
        print("\nModel weights loaded successfully")
        
        model.eval()  # Set the model to evaluation mode
        
        with torch.inference_mode():
            model = model.to(device)
            image = image.to(device)
            output = torch.softmax(model(image), dim=1).detach().cpu()
            prediction = torch.argmax(output, dim=1).item()
            #print(f'Output: {output.numpy()}')
            #Sprint(f'Prediction: {prediction}')
            if prediction == 0:
                print("The image is not pixelated")
            else:
                print("The image is pixelated")
        
    else:
        print("Invalid method selected (only 1 or 2 allowed)")
    
if __name__ == '__main__':
    main()
