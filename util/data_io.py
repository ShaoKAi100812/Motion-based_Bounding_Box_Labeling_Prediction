# import packages
import pandas as pd
from PIL import Image
import numpy as np
import os
import torch

# external path
PATH_DATA = os.path.join(os.getcwd(), "data")

# export ground truth
def gt_export(dataset: torch.Tensor, CSV_NAME: str) -> None:
    isExist = os.path.exists(PATH_DATA)
    if not isExist: os.mkdir(PATH_DATA)
    pd.DataFrame(dataset).to_csv(os.path.join(PATH_DATA, CSV_NAME), \
        index_label = "Ground Truth", header = ['x-vector'])

# import ground truth
def gt_import(CSV_NAME: str) -> torch.Tensor:
    isExist = os.path.exists(os.path.join(PATH_DATA, CSV_NAME))    
    if not isExist: raise NameError("File not exist!")
    df = pd.read_csv(os.path.join(PATH_DATA, CSV_NAME))
    gt = torch.Tensor(df.values[:, 1:]).reshape(len(df))
    return gt

# export image
def image_export(dataset: torch.Tensor, IMAGE_NAME: str) -> None:
    PATH_IMAGE = os.path.join(PATH_DATA, IMAGE_NAME)
    isExist = os.path.exists(PATH_IMAGE)
    if not isExist: os.mkdir(PATH_IMAGE)
    for i in range(len(dataset)):
        array = dataset[i].numpy().astype(np.uint8)    # np.uint8 before Image.fromarray
        im = Image.fromarray(array)
        im.save(os.path.join(PATH_IMAGE, "{}.png".format(i)), format="png")
        
# import image
def image_import(gt: torch.Tensor, IMAGE_NAME: str) -> torch.Tensor:
    PATH_IMAGE = os.path.join(PATH_DATA, IMAGE_NAME)
    isExist = os.path.exists(PATH_IMAGE)
    if not isExist: raise NameError("File not exist!")
    img = torch.empty(len(gt), 1, 28*3, 28*4)
    for i in range(len(gt)):
        im = Image.open(os.path.join(PATH_IMAGE, "{}.png".format(i)))
        im = np.array(list(im.getdata())).reshape(1, 28*3, 28*4)
        img[i] = torch.tensor(im)
    return img