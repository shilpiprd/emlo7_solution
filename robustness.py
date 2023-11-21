#the purpose of this file is to apply following methods on all ur images and see if output changes
# Model Robustness with

# Pixel Dropout
# FGSM
# Random Noise
# Random Brightness

#contains pretrained timm model
import urllib
import os
# import gradio as gr
import torch
import timm
import numpy as np
import random 

from PIL import Image

import torchvision.transforms as T
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from typing import Dict

from captum.attr import visualization as viz

#imports specifically for gradcam: 
import matplotlib.pyplot as plt
from captum.robust import FGSM
from captum.robust import MinParamPerturbation

from captum.attr import FeatureAblation

from PIL import ImageEnhance

def random_brightness(image_tensor):
    # Convert tensor to PIL image
    image_pil = T.ToPILImage()(image_tensor.squeeze().cpu())

    # Apply random brightness
    enhancer = ImageEnhance.Brightness(image_pil)
    factor = np.random.uniform(0.5, 1.5)  # Random brightness factor
    img_bright = enhancer.enhance(factor)

    # Convert back to tensor
    bright_img_tensor = transform(img_bright).unsqueeze(0).to(device)
    return bright_img_tensor


device = torch.device("cuda:0")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_prediction(model, image: torch.Tensor):
    model = model.to(device)
    img_tensor = image.to(device)
    with torch.no_grad():
        output = model(img_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]

    return predicted_label, prediction_score.squeeze().item()

    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

def image_show(img, pred):
    npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.title("prediction: %s" % pred)
    plt.show()


#actual prediction starting here: 
MODEL: str = "resnet18"

model = timm.create_model(MODEL, pretrained=True)
model.eval()
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
)
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

transform_normalize = T.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
)

def pixel_dropout(image, dropout_pixels):
    pixel_attr_device = pixel_attr.to(image.device) 
    keep_pixels = image[0][0].numel() - int(dropout_pixels)
    vals, _ = torch.kthvalue(pixel_attr_device.flatten(), keep_pixels)
    return (pixel_attr_device < vals.item()) * image

def sp_noise(image, prob):
    # Convert PIL Image to NumPy array
    image = np.array(image)
    output = np.zeros(image.shape, np.uint8) 
    thres = 1 - prob 
    for i in range(image.shape[0]): 
        for j in range(image.shape[1]): 
            rdn = random.random() 
            if rdn < prob: 
                output[i][j] = 0 
            elif rdn > thres: 
                output[i][j] = 255 
            else: 
                output[i][j] = image[i][j] 
    return Image.fromarray(output)
        
feature_mask = torch.arange(64 *7*7).reshape(8*7, 8*7).repeat_interleave(repeats=4, dim=1).repeat_interleave(repeats=4, dim=0).reshape(1, 1, 224,224)
feature_mask = feature_mask.to(device)
images =  os.listdir("/content/drive/MyDrive/img_dir/")
for image in images: 
    img =  Image.open('/content/drive/MyDrive/img_dir/' + image)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    inv_transform= T.Compose([
        T.Normalize(
            mean = (-1 * np.array(mean) / np.array(std)).tolist(),
            std = (1 / np.array(std)).tolist()
        ),
    ])
    print('original image: ')
    pred, score  = get_prediction(model, img_tensor)
    image_show(img_tensor.cpu(), pred + " " + str(score))
    
    print('Image after Pixel Dropout')
    ablator = FeatureAblation(model)
    attr = ablator.attribute(img_tensor, target=285, feature_mask=feature_mask)
    pixel_attr = attr[:,0:1]
    min_pert_attr = MinParamPerturbation(forward_func=model, attack=pixel_dropout, arg_name="dropout_pixels", mode="linear",
                                     arg_min=10, arg_max=1024, arg_step=10,
                                     preproc_fn=None, apply_before_preproc=True)

    print('Image after FGSM: ')
    fgsm = FGSM(model, lower_bound=-1, upper_bound=1)
    perturbed_image_fgsm = fgsm.perturb(img_tensor, epsilon=0.16, target=285)
    new_pred_fgsm, score_fgsm = get_prediction(model, perturbed_image_fgsm)
    image_show(perturbed_image_fgsm.cpu(), new_pred_fgsm + " " + str(score_fgsm)) 

    print('Image after Random Noise: ')
    noise_img = sp_noise(img, 0.05)
    noise_img_tensor = transform(noise_img) # Assuming 'transform' is defined to convert PIL images to tensors
    noise_img_tensor = noise_img_tensor.unsqueeze(0).to(device)
    prediction, score = get_prediction(model, noise_img_tensor)
    image_show(noise_img_tensor.cpu(), prediction + " " + str(score))

    print("Image after Random Brightness: ")
    bright_img_tensor = random_brightness(img_tensor)
    bright_pred, bright_score = get_prediction(model, bright_img_tensor)
    image_show(bright_img_tensor.cpu(), bright_pred + " " + str(bright_score))
