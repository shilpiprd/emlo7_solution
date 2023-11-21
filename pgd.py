#contains pretrained timm model
import urllib
import os
# import gradio as gr
import torch
import timm
import numpy as np

from PIL import Image

import torchvision.transforms as T
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from typing import Dict

from captum.attr import visualization as viz

#imports specifically for gradcam: 
import matplotlib.pyplot as plt
from captum.robust import PGD

#download timm  and captum with the following commands: 
# %pip install timm shap grad-cam
# %pip install git+https://github.com/pytorch/captum.git

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

    #code for PGD: 
    print('applying PGD: ') #286 is egyptian cat
    pgd = PGD(model, torch.nn.CrossEntropyLoss(reduction='none'), lower_bound=-1, upper_bound=1)  # construct the PGD attacker
    perturbed_image_pgd = pgd.perturb(inputs=img_tensor, radius=0.13, step_size=0.02,
                                    step_num=7, target=torch.tensor([285]).to(device), targeted=True)
    new_pred_pgd, score_pgd = get_prediction(model, perturbed_image_pgd)
    image_show(perturbed_image_pgd.cpu(), new_pred_pgd + " " + str(score_pgd))
    
