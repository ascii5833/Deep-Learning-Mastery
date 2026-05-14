import torch

def dice_score(preds, targets, smooth = 1e-8):
    
    #get the sigmoid outputs [0-1]
    preds = torch.sigmoid(preds)
    #binarize the predictions
    preds  = (preds > 0.5).float()
    #intersection (overlapping region, where 1 both that overlaps)
    intersection = (preds * targets).sum()
    #union
    union = preds.sum() + targets.sum()
    
    #dice score (2* iou)
    dice = ( 2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()