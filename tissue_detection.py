import numpy as np
import cv2


def tissue_detection(img, remove_top_percentage=0.2):

    assert 0 <= remove_top_percentage < 1, (f"remove_top_percentage needs to be in [0, 1). You passed "
                                            f"{remove_top_percentage}.")

    kernel_size = 3

    # remove alpha channel
    img = img[:, :, 0:3]

    top_border = int(len(img)*remove_top_percentage)
    # hack for removing border artifacts
    img[0:top_border, :, :] = [0, 0, 0]

    # remove black background pixel
    black_px = np.where((img[:, :, 0] <= 5) & (img[:, :, 1] <= 5) & (img[:, :, 2] <= 5))
    img[black_px] = [255, 255, 255]

    # apply median filter to remove artifacts created by transitions to background pixels
    median_filtered_img = cv2.medianBlur(img, 11)

    # convert to HSV color space
    hsv_image = cv2.cvtColor(median_filtered_img, cv2.COLOR_RGB2HSV)

    # get saturation channel
    saturation = hsv_image[:, :, 1]

    # Otsu's thresholding
    _, threshold_image = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # apply dilation to image to close spots inside mask regions
    kernel = np.ones(shape=(kernel_size, kernel_size))
    tissue_mask = cv2.dilate(threshold_image, kernel, iterations=1)
    # tissue_mask = cv2.erode(tissue_mask, kernel)

    return tissue_mask
