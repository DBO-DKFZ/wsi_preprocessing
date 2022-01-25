import copy
import tissue_detection
import cv2
import os
import json

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from xml.dom import minidom

# Fix to get the dlls to load properly under python >= 3.8 and windows
script_dir = os.path.dirname(os.path.realpath(__file__))
try:
    openslide_dll_path = os.path.join(script_dir, "..", "openslide-win64-20171122", "bin")
    os.add_dll_directory(openslide_dll_path)
    print(openslide_dll_path)
except Exception as e:
    print(e)

import openslide


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
        self.levels = self.slide.level_count - 1

    def load_annotation(self, annotation_path):

        annotation_dict = {}

        if annotation_path.endswith('.geojson'):
            with open(annotation_path) as annotation_file:
                annotations = json.load(annotation_file)
            # Only working for features of the type polygon
            for polygon_nb in range(len(annotations["features"])):
                annotation_dict.update({polygon_nb:annotations["features"][polygon_nb]["geometry"]["coordinates"][0]})

        elif annotation_path.endswith('.xml'):
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for elem in root:
                polygon_nb = 0
                for subelem in elem:
                    items = subelem.attrib
                    if "Type" in items.keys():
                        if items["Type"] == "Polygon":
                            annotation_dict.update({polygon_nb:None})
                            polygon_list = []
                            for coordinates in subelem:
                                for coord in coordinates:
                                    polygon_list.append([float(coord.attrib['X']), float(coord.attrib['Y'])])
                            annotation_dict[polygon_nb] = polygon_list
                            polygon_nb += 1
        else:
            print("Unknown file format")
            return None


        return annotation_dict

    def annotation2mask(self, annotation_dict, mask_level):

        mask_dimension = self.slide.level_dimensions[mask_level]
        scaling_factor = int(self.slide.level_downsamples[mask_level])

        annotation_mask = np.zeros(shape=(mask_dimension[1], mask_dimension[0]))

        scaled_list = []
        for polygon in annotation_dict:
            point_list = np.array([[int(np.round(sublist[0]/scaling_factor)), int(np.round(sublist[1]/scaling_factor))] for sublist in annotation_dict[polygon]],dtype=np.int32)
            scaled_list.append(point_list)

            cv2.fillPoly(img=annotation_mask, pts=[point_list], color=(1,1,1))
            plt.imshow(annotation_mask)
            plt.show()

        return annotation_mask.astype(np.bool)

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

        patch = image[y_coord:y_coord + height, x_coord:x_coord + width, :]

        return patch

    def get_relevant_tiles(self, tissue_mask, tile_size, min_coverage, level, show=False):

        # TODO: Handling border cases using the residue
        rows, row_residue = divmod(tissue_mask.shape[0], tile_size)
        cols, col_residue = divmod(tissue_mask.shape[1], tile_size)

        colored = cv2.cvtColor(tissue_mask, cv2.COLOR_GRAY2RGB)

        print("Filtering ", rows * cols, "tiles...")

        relevant_tiles_dict = {}
        tile_nb = 0
        for row in range(rows):
            for col in range(cols):

                tile = tissue_mask[row * tile_size:row * tile_size + tile_size,
                       col * tile_size:col * tile_size + tile_size]
                tissue_coverage = np.count_nonzero(tile) / tile.size

                if tissue_coverage >= min_coverage:
                    relevant_tiles_dict.update({tile_nb: {"x": col * tile_size, "y": row * tile_size,
                                                          "size": tile_size, "level": level}})

                    tissue_mask = cv2.rectangle(colored, (col * tile_size, row * tile_size),
                                                (col * tile_size + tile_size, row * tile_size + tile_size), (255, 0, 0),
                                                1)
                    tile_nb += 1

        print("Found", tile_nb, "relevant tiles")
        if show:
            plt.imshow(colored)
            plt.show()

        return relevant_tiles_dict

    def filter_tile(self, tile, kernel_size=3):

        # convert to HSV color space
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)

        # Otsu's thresholding
        _, threshold_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones(shape=(kernel_size, kernel_size))
        tissue_mask = cv2.dilate(threshold_image, kernel, iterations=1)

        plt.imshow(tissue_mask)
        plt.show()

        return tissue_mask

    def extract_patches(self, tile_dict, filter_mask=None, overlap=0, min_coverage=0, patch_size=256, file_dir=None):

        px_overlap = int(patch_size*overlap)
        patch_dict = {}

        # TODO: Check if int casting is valid
        scaling_factor = int(self.slide.level_downsamples[self.current_level])
        print("Scaling factor is", scaling_factor)

        tile_nb = 0

        for tile_key in tile_dict:

            tile_x = tile_dict[tile_key]["x"] * scaling_factor
            tile_y = tile_dict[tile_key]["y"] * scaling_factor
            tile_size = tile_dict[tile_key]["size"] * scaling_factor

            patch_dict.update({tile_nb: {"x_pos": tile_x, "y_pos": tile_y, "size": tile_size, "patches": {}}})

            # ToDo: rows and cols arent calculated correctly, instead a quick fix by using breaks was applied
            rows = int(np.ceil((tile_size + overlap) / (patch_size - px_overlap)))
            cols = int(np.ceil((tile_size + overlap) / (patch_size - px_overlap)))

            tile = np.array(self.slide.read_region((tile_x, tile_y), level=0, size=(tile_size, tile_size)))
            tile = tile[:, :, 0:3]

            patch_nb = 0
            stop_y = False
            for row in range(rows):
                stop_x = False

                for col in range(cols):

                    patch_x = int(col * (patch_size - px_overlap))
                    patch_y = int(row * (patch_size - px_overlap))

                    if patch_y+patch_size >= tile_size:
                        stop_y = True
                        patch_y = tile_size-patch_size

                    if patch_x+patch_size >= tile_size:
                        stop_x = True
                        patch_x = tile_size-patch_size

                    patch = tile[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size, :]

                    # ToDo: Filter mechanic to apply any filter mask on patches
                    patch_dict[tile_nb]["patches"].update({patch_nb: {"x_pos": patch_x, "y_pos": patch_y, "patch_size": patch_size}})

                    if file_dir is not None:
                        file_name = str(tile_nb)+"_"+str(patch_nb)+".png"
                        patch = Image.fromarray(patch)
                        patch.save(os.path.join(file_dir, file_name),  format="png")

                    patch_nb += 1
                    if stop_x:
                        break
                if stop_y:
                    break
            tile_nb += 1

        return patch_dict

    def stain_test(self):
        img, lvl = self.get_img(6, show=True)
        img = img[:, :, 0:3]

        I, H, E = tissue_detection.extract_stains(img)

        plt.imshow(H)
        plt.title("H stain")
        plt.show()
        plt.imshow(E)
        plt.title("E Stain")
        plt.show()
        plt.imshow(I)
        plt.title("Normalized")
        plt.show()


if __name__ == "__main__":
    coverage = 0.5
    level = 7
    tile_size = 16
    overlap = 0.25
    annotation_path = os.path.join("resources", "annotations_xml", "patient_004_node_4.xml")
    file_dir = "E:\\camelyon17_generalization"

    slide_handler = WSIHandler()
    slide_handler.load_slide(os.path.join(script_dir, "resources", "patient_004_node_4.tif"))
    annotation_dict = slide_handler.load_annotation(annotation_path)
    annotation_mask = slide_handler.annotation2mask(annotation_dict, level)
    #slide_handler.stain_test()
    mask, level = slide_handler.apply_tissue_detection(level=level, show=False)
    tile_dict = slide_handler.get_relevant_tiles(mask, tile_size=tile_size, min_coverage=coverage, level=level,
                                                 show=False)
    patch_dict = slide_handler.extract_patches(tile_dict, mask, file_dir=overlap, min_coverage=coverage, patch_size=256)
