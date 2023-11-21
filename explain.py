
import urllib
import os
# import gradio as gr
import torch
import timm
import numpy as np

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import torchvision.transforms as T
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from typing import Dict

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from captum.attr import Saliency
from captum.attr import DeepLift
#imports specifically for gradcam: 
import timm
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus
import shap


MODEL: str = "resnet18"

model = timm.create_model(MODEL, pretrained=True)
model.eval()
model = model.to(device)
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

images = os.listdir("/content/drive/MyDrive/img_dir/")
for image in images: 
    #create model then perform prediction
    print("Processing image: ", image)
    img = Image.open('/content/drive/MyDrive/img_dir/' + image)
    transformed_img = transform(img)
    img_tensor = transform_normalize(transformed_img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
    
    #code for IG 
    print('Integrated Gradients Model Explinability: ')
    integrated_gradients = IntegratedGradients(model)
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (1, '#000000')], N=256)
    attributions_ig = integrated_gradients.attribute(img_tensor, target=285, n_steps=400)

    sth = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                method='heat_map',
                                #  cmap=default_cmap,
                                show_colorbar=True,
                                sign='positive',
                                outlier_perc=1)
    
    sth[0].savefig('/content/s7/output/' + image + "_ig.png")
    #code for IG with Noise Tunnel 
    print('integrated gradients with noise tunnel')
    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(img_tensor, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
    sth = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        #   cmap=default_cmap,
                                        show_colorbar=True)
    
    sth[0].savefig('/content/s7/output/' + image + "_ig_WnoiseTunnel.png")

    #code for saliency:
    print('Saliency Model Explanation: ')
    saliency = Saliency(model)
    grads = saliency.attribute(img_tensor, target=285)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    original_image = np.transpose((img_tensor.squeeze(0).cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    sth = viz.visualize_image_attr(None, original_image,
                        method="original_image", title="Original Image")
    sth = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                            show_colorbar=True, title="Overlayed Gradient Magnitudes")

    sth[0].savefig('/content/s7/output/' + image + "_saliency.png")

    #code for Occlusion
    print('Occlusion Model Explainability: ')
    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(img_tensor,
                                        strides = (3, 2, 2),
                                        target=pred_label_idx,
                                        sliding_window_shapes=(3, 15, 15),
                                        baselines=0)
    sth = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2,
                                        )
    sth[0].savefig('/content/s7/output/' + image + "_occlusion.png")

    #code for SHAP (works well where number of classes are less)
    print('SHAP model explaiability: ')
    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([img_tensor * 0, img_tensor * 1])

    attributions_gs = gradient_shap.attribute(img_tensor,
                                            n_samples=50,
                                            stdevs=0.0001,
                                            baselines=rand_img_dist,
                                            target=pred_label_idx)
    sth = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "absolute_value"],
                                        #   cmap=default_cmap,
                                        show_colorbar=True)
    sth[0].savefig('/content/s7/output/' + image + "_SHAP.png")
    #code for GRAD
    print('GRAD model explainability: ')
    img_copy =  Image.open('/content/drive/MyDrive/img_dir/' + image)
    img_tensor = transform(img_copy)       #overriding image tensor. 
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    inv_transform = T.Compose([
        # T.Lambda(lambda x: x.permute(0, 2, 3, 1)),
        T.Normalize(
            mean = (-1 * np.array(mean) / np.array(std)).tolist(),
            std = (1 / np.array(std)).tolist()
        ),
        # T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
    ])
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(285)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    plt.imshow(visualization) 
    plt.show()
    Image.fromarray(visualization).save('/content/s7/output/' + image + "_gradcam.jpeg")
    #code for GRAD++
    print('code for GRAD++')
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.show()
    Image.fromarray(visualization).save('/content/s7/output/' + image + "_gradcam++.jpeg")