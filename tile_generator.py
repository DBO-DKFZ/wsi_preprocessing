import cv2
import openslide

import numpy as np
import matplotlib.pyplot as plt

import tissue_detection


class WSIHandler:

    def __init__(self):
        self.slide = None
        self.total_width = 0
        self.total_height = 0
        self.levels = 0

    def load_slide(self, slide_path):
        self.slide = openslide.OpenSlide(slide_path)

        self.total_width = self.slide.dimensions[0]
        self.total_height = self.slide.dimensions[1]
        self.levels = self.slide.level_count-1

    def get_none_zoomed_img(self, level=None):
        if level is None:
            level = self.levels
        none_zoomed_dims = self.slide.level_dimensions[level]
        none_zoomed_img = self.slide.read_region((0,0), level, none_zoomed_dims)
        image = np.array(none_zoomed_img)

        return image

    def apply_tissue_detection(self):
        mask = tissue_detection.tissue_detection(self.get_none_zoomed_img())

        return mask


if __name__ == "__main__":
    test_slide = openslide.OpenSlide('D:\\Development\\histopathological_image_preprocessing\\resources\\patient_002_node_0.tif')

    slide_handler = WSIHandler()
    slide_handler.load_slide('D:\\Development\\histopathological_image_preprocessing\\resources\\patient_002_node_0.tif')
    slide_handler.get_none_zoomed_img(level=4)
    slide_handler.apply_tissue_detection()


    """
    props = test_slide.properties
    i = 0
    height = 1024
    width = 1024

    level = 7

    print("Slide dimensions: ", test_slide.dimensions[0], test_slide.dimensions[1])

    level_dims = test_slide.level_dimensions[level]

    print("Level dimensions", level_dims)

    tiles_per_row, res_row = divmod(test_slide.level_dimensions[level][0], width)
    tiles_per_col, res_col = divmod(test_slide.level_dimensions[level][1], height)

    print("Tiles per row:", tiles_per_row, "Tiles per column: ", tiles_per_col)

    if tiles_per_row == 0 or tiles_per_col == 0:
        plt.imshow(test_slide.read_region((0,0,), level, test_slide.level_dimensions[level]))

        image = test_slide.read_region((0,0), level, (res_row, res_col))
        image = np.array(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow("Slide", image)
        cv2.waitKey(0)
    else:
        for col in range(tiles_per_col):
            for row in range(tiles_per_row):

                image = test_slide.read_region((row*width,col*height),level ,(width,height))
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.imshow("Slide", image)
                cv2.waitKey(0)


    print("Fin")

    """
