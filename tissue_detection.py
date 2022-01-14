import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import openslide

def tissue_detection(img):

    kernel_size = 3
    original_img = copy.deepcopy(img)

    img = img[:,:,0:3]

    black_px = np.where((img[:, :, 0] <= 5) & (img[:, :, 1] <= 5) & (img[:, :, 2] <= 5))
    img[black_px] = [255, 255, 255]
    #plt.imshow(img)
    #plt.show()

    img = cv2.medianBlur(img, 11)
    #plt.imshow(img)
    #plt.show()

    bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #plt.imshow(bgr_image)
    #plt.show()

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    #plt.imshow(hsv_image[:,:,1])
    #plt.show()

    saturation = hsv_image[:,:,1]

    # Otsu's thresholding
    _, th2 = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones(shape=(kernel_size, kernel_size))

    dilated = cv2.dilate(th2, kernel, iterations=1)

    #plt.imshow(th2)
    #plt.show()

    img_width, img_height = th2.shape

    mask = np.zeros(shape=(img_width, img_height, 3)).astype('uint8')
    mask[:,:, 1] = dilated

    alpha = 0.8
    overlay = cv2.addWeighted(mask, alpha, original_img[:,:,0:3], 1-alpha, 0.0)
    cv2.imwrite("original.png", original_img)
    cv2.imwrite("test.png", overlay)

    #cv2.imshow("", overlayed_img)
    #cv2.waitKey()




