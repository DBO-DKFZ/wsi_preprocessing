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
        self.current_level = 0

    def load_slide(self, slide_path):
        self.slide = openslide.OpenSlide(slide_path)

        self.total_width = self.slide.dimensions[0]
        self.total_height = self.slide.dimensions[1]
        self.levels = self.slide.level_count-1

    def get_img(self, level=None, show=False):
        if level is None:
            level = self.levels

        dims = self.slide.level_dimensions[level]
        image = np.array(self.slide.read_region((0, 0), level, dims))

        if show:
            plt.imshow(image)
            plt.show()

        return image, level

    def apply_tissue_detection(self, level=None, show=False):

        if level is not None:
            image, level = self.get_img(level, show)
        else:
            image, level = self.get_img(show=show)

        self.current_level = level

        tissue_mask = tissue_detection.tissue_detection(image)

        if show:
            plt.imshow(tissue_mask)
            plt.show()

        return tissue_mask, level

    @staticmethod
    def extract_patch(image, x_coord, y_coord, width, height):

        patch = image[y_coord:y_coord+height, x_coord:x_coord+width, :]

        return patch

    def get_relevant_tiles(self, tissue_mask, tile_size, min_coverage, level, show=False):

        # TODO: Handling border cases using the residue
        rows, row_residue = divmod(tissue_mask.shape[0], tile_size)
        cols, col_residue = divmod(tissue_mask.shape[1], tile_size)

        colored = cv2.cvtColor(tissue_mask, cv2.COLOR_GRAY2RGB)

        print("Filtering ", rows*cols, "tiles...")

        relevant_tiles_dict = {}
        tile_nb = 0
        for row in range(rows):
            for col in range(cols):

                tile = tissue_mask[row*tile_size:row*tile_size+tile_size, col*tile_size:col*tile_size+tile_size]
                tissue_coverage = np.count_nonzero(tile)/tile.size

                if tissue_coverage >= min_coverage:

                    relevant_tiles_dict.update({tile_nb: {"x": col*tile_size, "y": row*tile_size,
                                                          "size": tile_size, "level": level}})

                    tissue_mask = cv2.rectangle(colored, (col*tile_size, row*tile_size),
                                                (col*tile_size+tile_size, row*tile_size + tile_size), (255, 0, 0), 1)
                    tile_nb += 1
        if show:
            plt.imshow(colored)
            plt.show()

        return relevant_tiles_dict

    def extract_patch_coordinates(self, tile_dict, min_coverage):

        patch_dict = {}

        # TODO: Check if int casting is valid
        scaling_factor = int(self.slide.level_downsamples[self.current_level])
        print("Scaling factor is", scaling_factor)
        for tile_key in tile_dict:
            tile = self.slide.read_region((tile_dict[tile_key]["x"]*scaling_factor,
                                           tile_dict[tile_key]["y"]*scaling_factor),
                                          level=1,
                                          size=(tile_dict[tile_key]["size"] * scaling_factor,
                                                tile_dict[tile_key]["size"] * scaling_factor))
            plt.imshow(tile)
            plt.show()
        return patch_dict

if __name__ == "__main__":
    test_slide = openslide.OpenSlide('D:\\Development\\histopathological_image_preprocessing\\resources\\patient_002_node_0.tif')

    coverage = 0.5
    level = 7

    slide_handler = WSIHandler()
    slide_handler.load_slide('D:\\Development\\histopathological_image_preprocessing\\resources\\patient_002_node_0.tif')
    #slide_handler.get_img(level=8, show=True)
    mask, level = slide_handler.apply_tissue_detection(level=level, show=True)
    tile_dict = slide_handler.get_relevant_tiles(mask, tile_size=4, min_coverage=coverage, level=level, show=True)
    patch_dict = slide_handler.extract_patch_coordinates(tile_dict, coverage)