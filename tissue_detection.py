import numpy as np
import matplotlib.pyplot as plt

import cv2
import openslide

def tissue_detection(img):

    black_px = np.where((img[:, :, 0] <= 5) & (img[:, :, 1] <= 5) & (img[:, :, 2] <= 5))
    img[black_px] = [255, 255, 255]
    plt.imshow(img)
    plt.show()

    img = cv2.medianBlur(img, 11)
    plt.imshow(img)
    plt.show()

    bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(bgr_image)
    plt.show()

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    plt.imshow(hsv_image[:,:,1])
    plt.show()

    saturation = hsv_image[:,:,1]

    # Otsu's thresholding
    _, th2 = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.imshow(th2)
    plt.show()






