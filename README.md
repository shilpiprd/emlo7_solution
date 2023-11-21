# Task 
Use Pretrained Models from TIMM (take models with larger input)

Do ALL the following for any 10 images taken by you (must be a class from ImageNet)

Model Explanation with

IG
IG w/ Noise Tunnel
Saliency
Occlusion
SHAP
GradCAM
GradCAM++
Use PGD to make the model predict cat for all images

save the images that made it predict cat
add these images to the markdown file in your github repository
Model Robustness with

Pixel Dropout
FGSM
Random Noise
Random Brightness
HINT: you can use https://albumentations.ai/Links to an external site. for more image perturbations

Integrate above things into your pytorch lightning template

create explain.py that will do all the model explanations
create robustness.py to check for model robustness

## File Information 
- the 3 .py files contain code for 3 different purposes. 
- The output from running all 3 differnt python files have been saved in directory. 
