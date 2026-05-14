import os
import shutil
import random
from paths import (
    IMAGE_DIR, MASK_DIR, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TEST_IMAGE_DIR, TEST_MASK_DIR
)

random.seed(42)

#make directories if already not existing
os.makedirs(TRAIN_IMAGE_DIR, exist_ok = True)
os.makedirs(TRAIN_MASK_DIR, exist_ok = True)

os.makedirs(TEST_IMAGE_DIR, exist_ok = True)
os.makedirs(TEST_MASK_DIR, exist_ok = True)

#all images
images = os.listdir(IMAGE_DIR)

#shuffle the images
random.shuffle(images)

split_idx = int(0.85 * len(images))

#get the train images
train_images = images[: split_idx]
#get the test images
test_images = images[split_idx:]

#loop through the train images
for img in train_images:
    
    #copy to new path
    shutil.copy(
        os.path.join(IMAGE_DIR, img),
        os.path.join(TRAIN_IMAGE_DIR, img)
    )
    
    #mask too
    shutil.copy(
        os.path.join(MASK_DIR, img),
        os.path.join(TRAIN_MASK_DIR, img)
    )
    
    
#loop through the test images
for img in test_images:
    
    #copy to new path
    shutil.copy(
        os.path.join(IMAGE_DIR, img),
        os.path.join(TEST_IMAGE_DIR, img)
    )
    
    #mask too
    shutil.copy(
        os.path.join(MASK_DIR, img),
        os.path.join(TEST_MASK_DIR, img)
    )
    

print("Dataset split complete")



