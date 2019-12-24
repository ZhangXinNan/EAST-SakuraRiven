

from PIL import Image
import numpy as np
import cv2


def cv2pil(img):
    shape = img.shape
    if len(shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def pil2cv(img_pil):
    img_np = np.array(img_pil)
    shape = img_np.shape
    if len(shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np

