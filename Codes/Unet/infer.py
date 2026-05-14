import torch 
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.unet import Unet
import config

#transformations
transform = A.Compose([
        A.Resize(
        config.IMAGE_SIZE,
        config.IMAGE_SIZE
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])

#load model
model = Unet(num_classes = 1).to(config.DEVICE)

#load weights
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))

model.eval()

#augment
image = cv2.imread('Datasets/test_images/ck2bxlujamu330725szlc2jdu.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

augmented = transform(image = image)

input_image = augmented['image']
#add a batch dimension
input_image = input_image.unsqueeze(0)
#load image to device
input_image = input_image.to(config.DEVICE)

#get prediction
with torch.no_grad():
    #get prediction
    pred = model(input_image)
    #get sigmoid
    pred = torch.sigmoid(pred)
    
    pred = (pred > 0.5).float()
    
mask = pred.squeeze().cpu().numpy()

#reverse normalization
cv2.imwrite('prediction.png', mask * 255)


