import torch
from torch.utils.data import DataLoader
from models.unet import Unet
from utils.dataset import KVASIR
from utils.metrics import dice_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from tqdm import tqdm


#faster convolutions
torch.backends.cudnn.benchmark = True

#test transform
test_transform = A.Compose([
    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    A.Normalize(
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)
    ),
    ToTensorV2()
    
])

#test dataset
test_dataset = KVASIR(
    image_dir = config.TEST_IMG_DIR, 
    mask_dir = config.TEST_MASK_DIR, 
    transform = test_transform,
    )

#test loader
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

#load model
model = Unet(num_classes = 1).to(config.DEVICE)
model.load_state_dict(
    torch.load(config.MODEL_SAVE_PATH)
)

model.eval()


dice_total = 0

with torch.no_grad():
    for images, masks in tqdm(test_loader):
        images = images.to(config.DEVICE)
        masks = masks.to(config.DEVICE)
        #calculate outputs
        outputs = model(images)
        #calculate dice score
        dice = dice_score(outputs, masks)
        #dice total
        dice_total += dice

        
#average dice
avg_dice = dice_total / len(test_loader)    
    
#print the score
print(f"Average dice score is: {avg_dice}")