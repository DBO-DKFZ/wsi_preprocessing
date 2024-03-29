# System
import json
import multiprocessing
import os
import warnings

# Advanced
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Numpy
import numpy as np

# Image Processing
from PIL import Image

# # Fix to get the dlls to load properly under python >= 3.8 and windows
script_dir = os.path.dirname(os.path.realpath(__file__))
try:
    openslide_dll_path = os.path.join(script_dir, "..", "openslide-win64-20171122", "bin")
    os.add_dll_directory(openslide_dll_path)
    # print(openslide_dll_path)

except Exception as e:
    pass

# noinspection PyPep8
import openslide

# Custom
# noinspection PyPep8
import tissue_detection

_MULTIPROCESS = True

global lock


class WSIHandler:
    def __init__(self, config_path="resources/config.json"):
        self.slide = None
        self.output_path = None
        self.total_width = 0
        self.total_height = 0
        self.levels = 0
        self.current_level = 0
        self.annotation_list = None
        self.annotation_dict = None
        self.config = self.load_config(config_path)
        assert "save_annotated_only" in self.config.keys()
        self.annotated_only = self.config["save_annotated_only"]
        self.scanner = None

        self.res_x = None
        self.res_y = None

        self.validate_label_dict()

    def validate_label_dict(self):
        self.check_at_most_one_unannotated_label()
        self.check_unannotated_label_first()

    def check_at_most_one_unannotated_label(self):
        label_dict = self.config["label_dict"]
        unannotated_labels = []
        for label, label_config in label_dict.items():
            if not label_config["annotated"]:
                unannotated_labels.append(label)
        assert len(unannotated_labels) < 2, (f"More than one label (=tissue type) is marked as unannotated in the "
                                             f"config.label_dict. Please make sure that at most one type (usually "
                                             f"non-tumor) is marked as unannotated. The labels in question are "
                                             f"{unannotated_labels}.")

    def check_unannotated_label_first(self):
        label_dict = self.config["label_dict"]
        for label in list(label_dict)[1:]:
            assert label_dict[label]["annotated"], (f"WSIHandler requires the unannotated label to be located in the "
                                                    f"first position in config.label_dict. Please move the unannotated "
                                                    f"tissue type '{label}' to the first position.")

    def print_and_log_slide_error(self, slide_name, error_msg, method_name):
        print(f"Error in slide {slide_name}. The error is: {type(error_msg).__name__}: {error_msg} in method: "
              f"{method_name}.")
        with lock:
            with open(os.path.join(self.config["output_path"], "error_log.txt"), "a") as f:
                f.write(f"Error in slide {slide_name}. The error is: {type(error_msg).__name__}: {error_msg} in "
                        f"method: {method_name}.")

    @staticmethod
    def load_config(config_path):
        assert os.path.exists(config_path), "Cannot find " + config_path
        with open(config_path) as json_file:
            config = json.load(json_file)

        assert 1 >= config["tissue_coverage"] >= 0, "Tissue coverage must be between 1 and 0"
        assert config["blocked_threads"] >= 0
        assert config["patches_per_tile"] >= 1, "Patches per tile must be >= 1"
        assert 0 <= config["overlap"] < 1, "Overlap must be between 1 and 0"
        assert config["annotation_overlap"] >= 0 and config["overlap"] < 1, "Annotation overlap must be between 1 and 0"

        return config

    def load_slide(self, slide_path):

        self.slide = openslide.OpenSlide(slide_path)
        self.total_width = self.slide.dimensions[0]
        self.total_height = self.slide.dimensions[1]
        self.levels = self.slide.level_count - 1

        processing_level = self.config["processing_level"]

        if self.levels < self.config["processing_level"]:
            print("###############################################")
            print(
                "WARNING: Processing level above highest available slide level. Maximum slide level is "
                + str(self.levels)
                + ", processing level is "
                + str(self.config["processing_level"])
                + ". Setting processing level to "
                + str(self.levels)
            )
            print("###############################################")
            processing_level = self.levels

        return processing_level

    def load_annotation(self, annotation_path):
        annotation_dict = {}
        file_format = Path(annotation_path).suffix

        # QuPath exports
        if file_format == ".geojson" or file_format == ".txt":
            with open(annotation_path) as annotation_file:
                annotations = json.load(annotation_file)

            for polygon_nb in range(len(annotations["features"])):
                if annotations["features"][polygon_nb]["geometry"]["type"] == "Polygon":
                    if (annotations["features"][polygon_nb]["properties"]["classification"]["name"] in
                            self.config["label_dict"].keys()):
                        annotation_dict.update({polygon_nb: {
                            "coordinates": annotations["features"][polygon_nb]["geometry"]["coordinates"][0],
                            "tissue_type": annotations["features"][polygon_nb]["properties"]["classification"][
                                "name"]}})
                    else:
                        warnings.warn(f'Unknown annotation type in file {annotation_file.name}: The annotation label '
                                      f'"{annotations["features"][polygon_nb]["properties"]["classification"]["name"]}"'
                                      f' is not part of the provided label dictionary '
                                      f'(keys: {list(self.config["label_dict"].keys())}. Skipping.')
                else:
                    warnings.warn(f'Not implemented warning in file {annotation_file.name}: The handling of the QuPath '
                                  f'annotation type {annotations["features"][polygon_nb]["geometry"]["type"]} '
                                  f'(id:{annotations["features"][polygon_nb]["id"]}) has not been implemented, yet. '
                                  f'Skipping.')
        # xml for CAMELYON17
        elif file_format == ".xml":
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for elem in root:
                polygon_nb = 0
                for subelem in elem:
                    items = subelem.attrib
                    if "Type" in items.keys():
                        if items["Type"] == "Polygon":
                            polygon_list = []
                            for coordinates in subelem:
                                for coord in coordinates:
                                    polygon_list.append([float(coord.attrib["X"]), float(coord.attrib["Y"])])
                            # all annotationy in CAMELYON17 are tumor, so this is a pseudo label
                            annotation_dict.update({polygon_nb: {"coordinates": polygon_list, "tissue_type": "Tumor"}})
                            polygon_nb += 1
        else:
            return None

        return annotation_dict

    def get_img(self, level=None, show=False):
        if level is None:
            level = self.levels

        dims = self.slide.level_dimensions[level]
        image = np.array(self.slide.read_region((0, 0), level, dims))

        if show:
            # Katja: fix for Wayland issue on my Ubuntu:
            # run 'export QT_QPA_PLATFORM=xcb' before opening pycharm (in the same terminal)
            plt.imshow(image)
            plt.title("Slide image")
            plt.show()

        return image, level

    def apply_tissue_detection(self, level=None, show=False):

        if level is not None:
            image, level = self.get_img(level, show)
        else:
            image, level = self.get_img(show=show)

        tissue_mask = tissue_detection.tissue_detection(image, remove_top_percentage=0)

        if show:
            plt.imshow(tissue_mask)
            plt.title("Tissue Mask")
            plt.show()

        return tissue_mask, level

    def determine_tile_size(self, level):

        if self.config["calibration"]["use_non_pixel_lengths"]:
            tile_size_0 = (self.config["calibration"]["patch_size_microns"] / self.res_x) * self.config[
                "patches_per_tile"
            ]
        else:
            tile_size_0 = self.config["patches_per_tile"] * self.config["patch_size"]

        downscale_factor = int(self.slide.level_downsamples[level])
        tile_size = int(tile_size_0 / downscale_factor)

        assert self.config["patches_per_tile"] >= 1, "Patches per tile must be greater than 1."

        return tile_size

    def get_relevant_tiles(self, tissue_mask, tile_size, min_coverage, level, show=False):

        rows, row_residue = divmod(tissue_mask.shape[0], tile_size)
        cols, col_residue = divmod(tissue_mask.shape[1], tile_size)

        if row_residue:
            rows += 1
        if col_residue:
            cols += 1

        if self.config["use_tissue_detection"]:
            colored = cv2.cvtColor(tissue_mask, cv2.COLOR_GRAY2RGB)

        if self.annotation_dict is not None:
            annotation_mask = np.zeros(shape=(tissue_mask.shape[0], tissue_mask.shape[1]))
            scaling_factor = self.slide.level_downsamples[level]
            scaled_list = [
                [[point[0] / scaling_factor, point[1] / scaling_factor]
                 for point in self.annotation_dict[polygon]["coordinates"]]
                for polygon in self.annotation_dict
            ]

            for polygon in scaled_list:
                cv2.fillPoly(annotation_mask, [np.array(polygon).astype(np.int32)], 1)

        relevant_tiles_dict = {}
        tile_nb = 0

        # +1 to solve border issues
        for row in range(rows):
            for col in range(cols):

                tile = tissue_mask[
                       row * tile_size: row * tile_size + tile_size, col * tile_size: col * tile_size + tile_size
                       ]
                tissue_coverage = np.count_nonzero(tile) / tile.size
                annotated = False

                if self.annotation_dict is not None:
                    if (np.count_nonzero(annotation_mask[row * tile_size: row * tile_size + tile_size,
                                         col * tile_size: col * tile_size + tile_size, ]) > 0):
                        annotated = True

                if (tissue_coverage >= min_coverage or
                        (self.config["keep_annotated_tiles_despite_too_little_tissue_coverage"] and annotated)):
                    relevant_tiles_dict.update(
                        {
                            tile_nb: {
                                "x": col * tile_size,
                                "y": row * tile_size,
                                "size": tile_size,
                                "level": level,
                                "annotated": annotated,
                            }
                        }
                    )
                    if self.config["use_tissue_detection"]:
                        if annotated:
                            colored = cv2.rectangle(
                                colored,
                                (col * tile_size, row * tile_size),
                                (col * tile_size + tile_size, row * tile_size + tile_size),
                                (0, 255, 0),
                                3,
                            )
                        else:
                            colored = cv2.rectangle(
                                colored,
                                (col * tile_size, row * tile_size),
                                (col * tile_size + tile_size, row * tile_size + tile_size),
                                (255, 0, 0),
                                1,
                            )

                    tile_nb += 1

        if show and self.config["use_tissue_detection"]:
            plt.imshow(colored)
            plt.title("Tiled image")
            plt.show()

        return relevant_tiles_dict

    @staticmethod
    def tissue_percentage_over_threshold(label, label_dict, percentage):
        if label_dict[label]["type"] == "==":
            if label_dict[label]["threshold"] == percentage:
                return label, percentage
        elif label_dict[label]["type"] == ">=":
            if percentage >= label_dict[label]["threshold"]:
                return label, percentage
        elif label_dict[label]["type"] == ">":
            if percentage > label_dict[label]["threshold"]:
                return label, percentage
        elif label_dict[label]["type"] == "<=":
            if percentage <= percentage[label]["threshold"]:
                return label, percentage
        elif label_dict[label]["type"] == "<":
            if percentage < label_dict[label]["threshold"]:
                return label, percentage

        return None, None

    @staticmethod
    def check_tissue_percentage_over_threshold(label, label_dict, percentage):
        if label_dict[label]["type"] == "==":
            if label_dict[label]["threshold"] == percentage:
                return True
        elif label_dict[label]["type"] == ">=":
            if percentage >= label_dict[label]["threshold"]:
                return True
        elif label_dict[label]["type"] == ">":
            if percentage > label_dict[label]["threshold"]:
                return True
        elif label_dict[label]["type"] == "<=":
            if percentage <= percentage[label]["threshold"]:
                return True
        elif label_dict[label]["type"] == "<":
            if percentage < label_dict[label]["threshold"]:
                return True
        return False

    @staticmethod
    def get_unique_nonzero_entries(ndarray):
        return np.unique(ndarray[np.nonzero(ndarray)]).astype(int)

    def get_possible_labels(self, annotation_mask):
        if self.get_unique_nonzero_entries(annotation_mask).size >= 1:
            return self.get_unique_nonzero_entries(annotation_mask).tolist()
        else:
            return [0]  # completely unlabeled patch -> only non-tumor

    @staticmethod
    def is_non_tumor(label_ids):  # non-tumor tissue is unannotated tissue that's left after tissue detection
        return label_ids[0] == 0

    def calculate_label_percentages(self, label_ids, annotation_mask):
        if self.is_non_tumor(label_ids):
            label_percentages = [(np.count_nonzero(np.max(annotation_mask, axis=(-1)) == 0) /
                                  annotation_mask[:, :, 0].size)]
        else:
            label_percentages = []
            for label_id in label_ids:
                label_percentages.append((np.count_nonzero(annotation_mask[:, :, label_id] == label_id) /
                                          annotation_mask[:, :, label_id].size))
        return label_percentages

    def get_labels_with_enough_tissue_annotated(self, label_dict, annotation_mask):
        label_ids = self.get_possible_labels(annotation_mask)
        label_percentages = self.calculate_label_percentages(label_ids, annotation_mask)

        labels_with_enough_tissue_including_non_tumor = []
        for (label_id, label_percentage) in zip(label_ids, label_percentages):
            label = list(label_dict)[label_id]
            if self.check_tissue_percentage_over_threshold(label, label_dict, label_percentage):
                labels_with_enough_tissue_including_non_tumor.append(label)

        return labels_with_enough_tissue_including_non_tumor

    def update_overlapping_annotations_file(self, slide_name, verbose):
        with open(os.path.join(self.config["output_path"],
                               "overlapping_annotations_present_in_slides.json"), "r") as file:
            overlapping_annotations_present = json.load(file)
        if not overlapping_annotations_present[slide_name]:
            overlapping_annotations_present[slide_name] = True
            with (open(os.path.join(self.config["output_path"],
                                    "overlapping_annotations_present_in_slides.json"), "w")
                  as file):
                json.dump(overlapping_annotations_present, file, indent=4)
            if verbose:
                print(f"There are overlapping annotations in slide {slide_name}.")

    @staticmethod
    def normalize_to_tile_size_px(point, tile_size_px):
        if point < 0:
            return 0
        elif point >= tile_size_px:
            return tile_size_px - 1.0
        else:
            return point

    def translate_world_coordinates_to_tile_coordinates(self, point, tile_x, tile_y, tile_size_px):
        # the shrinkage of the coordinates to tile size is necessary as cv2.fillPoly only works if the annotation is
        # completely within the tile, so I set any points larger than the tile coordinates to the closest (valid)
        # tile coordinates
        return [self.normalize_to_tile_size_px(point[0] - tile_x, tile_size_px),
                self.normalize_to_tile_size_px(point[1] - tile_y, tile_size_px)]

    def extract_calibrated_patches(
            self,
            tile_dict,
            level,
            annotations,
            label_dict,
            overlap=0,
            annotation_overlap=0,
            slide_name=None,
            output_format="png",
    ):

        scaling_factor = int(self.slide.level_downsamples[level])

        patch_dict = {}
        patch_nb = 0
        for tile_key in tile_dict:
            tile_x = tile_dict[tile_key]["x"] * scaling_factor
            tile_y = tile_dict[tile_key]["y"] * scaling_factor

            tile_size_px = tile_dict[tile_key]["size"] * scaling_factor

            patch_size_px_x = int(np.round(self.config["calibration"]["patch_size_microns"] / self.res_x))
            patch_size_px_y = int(np.round(self.config["calibration"]["patch_size_microns"] / self.res_y))

            tile = np.array(self.slide.read_region((tile_x, tile_y), level=0, size=(tile_size_px, tile_size_px)))
            tile = tile[:, :, 0:3]

            if tile_dict[tile_key]["annotated"]:
                px_overlap_x = int(patch_size_px_x * annotation_overlap)
                px_overlap_y = int(patch_size_px_y * annotation_overlap)

            else:
                px_overlap_x = int(patch_size_px_x * overlap)
                px_overlap_y = int(patch_size_px_y * overlap)

            rows = int(np.ceil(tile_size_px / (patch_size_px_y - px_overlap_y)))
            cols = int(np.ceil(tile_size_px / (patch_size_px_x - px_overlap_x)))

            # create annotation mask
            if annotations is not None:
                # Translate from world coordinates to tile coordinates
                tile_annotation_list = [
                    [self.translate_world_coordinates_to_tile_coordinates(point, tile_x, tile_y, tile_size_px)
                     for point in annotations[polygon]["coordinates"]] for polygon in annotations]

                tile_annotation_list = list(zip(tile_annotation_list, [annotations[polygon]["tissue_type"]
                                                                       for polygon in annotations]))

                # Create mask from polygons
                tile_annotation_mask = np.zeros(shape=(tile_size_px, tile_size_px, len(self.config["label_dict"])))

                annotated_tissue_types = {}
                tissue_type_number = 1
                for tissue_type, tissue_details in label_dict.items():
                    if tissue_details["annotated"]:
                        annotated_tissue_types.update({tissue_type: tissue_type_number})
                        tissue_type_number += 1

                for polygon in tile_annotation_list:
                    # note: the casting to a contiguous array is due to OpenCV requiring C-order (row major) for
                    # implementation purposes, compare the answer by vvolhejn here
                    # https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
                    # basically: many (all?) copy operations in numpy do this, ascontiguousarray is one of the more
                    # verbose ones
                    tile_annotation_mask[:, :, annotated_tissue_types[polygon[1]]] = (
                        cv2.fillPoly(np.ascontiguousarray(
                            tile_annotation_mask[:, :, annotated_tissue_types[polygon[1]]]),
                            [np.array(polygon[0]).astype(np.int32)], annotated_tissue_types[polygon[1]]))

            stop_y = False

            for row in range(rows):
                stop_x = False

                for col in range(cols):

                    # Calculate patch coordinates
                    patch_x = int(col * (patch_size_px_x - px_overlap_x))
                    patch_y = int(row * (patch_size_px_y - px_overlap_y))

                    if patch_y + patch_size_px_y >= tile_size_px:
                        stop_y = True
                        patch_y = tile_size_px - patch_size_px_y

                    if patch_x + patch_size_px_x >= tile_size_px:
                        stop_x = True
                        patch_x = tile_size_px - patch_size_px_x

                    global_x = patch_x + tile_x
                    global_y = patch_y + tile_y

                    patch = tile[patch_y: patch_y + patch_size_px_y, patch_x: patch_x + patch_size_px_x, :]

                    if np.sum(patch) == 0:
                        break

                    # check if the patch is annotated
                    annotated = False

                    if annotations is not None:
                        patch_mask = tile_annotation_mask[patch_y: patch_y + patch_size_px_y,
                                     patch_x: patch_x + patch_size_px_x, :]
                        labels = self.get_labels_with_enough_tissue_annotated(label_dict, patch_mask)
                        if labels is not None:
                            if len(labels) > 1:
                                self.update_overlapping_annotations_file(
                                    slide_name, verbose=self.config["overlapping_annotations_verbose"])

                            for label in labels:
                                # this check is done to ensure that non-tumor tissue (unannotated) is handled properly
                                if self.config["label_dict"][label]["annotated"]:
                                    annotated = True

                    else:
                        labels = "unlabeled"

                    if labels is not None:
                        if self.annotated_only and annotated or not self.annotated_only:

                            file_name = slide_name + "_" + str(global_x) + "_" + str(global_y) + "." + output_format

                            if self.config["calibration"]["resize"]:
                                patch = cv2.resize(patch, (self.config["patch_size"], self.config["patch_size"]))

                            patch = Image.fromarray(patch)
                            for label in labels:
                                patch.save(os.path.join(self.output_path, label, file_name), format=output_format)

                                patch_dict.update(
                                    {
                                        patch_nb: {
                                            "slide_name": slide_name,
                                            "patch_path": os.path.join(label, file_name),
                                            "label": label,
                                            "x_pos": global_x,
                                            "y_pos": global_y,
                                            "patch_size": patch_size_px_x,
                                            "resized": self.config["calibration"]["resize"],
                                        }
                                    }
                                )
                                patch_nb += 1
                    if stop_x:
                        break
                if stop_y:
                    break

        return patch_dict

    def make_dirs(self, output_path, slide_name, label_dict, annotated):
        try:
            slide_path = os.path.join(output_path, slide_name)
            if not os.path.exists(slide_path):
                os.makedirs(slide_path)
            if not annotated:
                unlabeled_path = os.path.join(slide_path, "unlabeled")
                if not os.path.exists(unlabeled_path):
                    os.makedirs(unlabeled_path)
            else:
                for label in label_dict:
                    sub_path = os.path.join(slide_path, label)
                    if not os.path.exists(sub_path):
                        os.makedirs(sub_path)
                    for patch in os.listdir(sub_path):
                        os.remove(os.path.join(sub_path, patch))
            self.output_path = slide_path

        except Exception as e:
            self.print_and_log_slide_error(slide_name, e, "make_dirs")

    def extract_patches(
            self,
            tile_dict,
            level,
            annotations,
            label_dict,
            overlap=0,
            annotation_overlap=0,
            patch_size=256,
            slide_name=None,
            output_format="png",
    ):
        patch_dict = {}

        scaling_factor = int(self.slide.level_downsamples[level])
        patch_nb = 0

        for tile_key in tile_dict:
            # skip unannotated tiles in case only annotated patches should be saved
            if self.annotated_only and not tile_dict[tile_key]["annotated"]:
                pass
            else:
                # ToDo: rows and cols arent calculated correctly, instead a quick fix by using breaks was applied

                tile_x = tile_dict[tile_key]["x"] * scaling_factor
                tile_y = tile_dict[tile_key]["y"] * scaling_factor
                tile_size = tile_dict[tile_key]["size"] * scaling_factor
                tile = np.array(self.slide.read_region((tile_x, tile_y), level=0, size=(tile_size, tile_size)))
                tile = tile[:, :, 0:3]

                # overlap separately  for annotated and unannotated patches
                if tile_dict[tile_key]["annotated"]:
                    px_overlap = int(patch_size * annotation_overlap)
                    rows = int(np.ceil(tile_size / (patch_size - px_overlap)))
                    cols = int(np.ceil(tile_size / (patch_size - px_overlap)))

                else:
                    px_overlap = int(patch_size * overlap)
                    rows = int(np.ceil(tile_size / (patch_size - px_overlap)))
                    cols = int(np.ceil(tile_size / (patch_size - px_overlap)))

                # create annotation mask
                if annotations is not None:
                    # Translate from world coordinates to tile coordinates
                    tile_annotation_list = [
                        [self.translate_world_coordinates_to_tile_coordinates(point, tile_x, tile_y, tile_size)
                         for point in annotations[polygon]["coordinates"]] for polygon in annotations]

                    tile_annotation_list = list(zip(tile_annotation_list, [annotations[polygon]["tissue_type"]
                                                                           for polygon in annotations]))

                    # Create mask from polygons
                    tile_annotation_mask = np.zeros(shape=(tile_size, tile_size, len(self.config["label_dict"])))

                    annotated_tissue_types = {}
                    tissue_type_number = 1
                    for tissue_type, tissue_details in label_dict.items():
                        if tissue_details["annotated"]:
                            annotated_tissue_types.update({tissue_type: tissue_type_number})
                            tissue_type_number += 1

                    for polygon in tile_annotation_list:
                        # note: the casting to a contiguous array is due to OpenCV requiring C-order (row major) for
                        # implementation purposes, compare the answer by vvolhejn here
                        # https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
                        # basically: many (all?) copy operations in numpy do this, ascontiguousarray is one of the more
                        # verbose ones
                        tile_annotation_mask[:, :, annotated_tissue_types[polygon[1]]] = (
                            cv2.fillPoly(np.ascontiguousarray(
                                tile_annotation_mask[:, :, annotated_tissue_types[polygon[1]]]),
                                [np.array(polygon[0]).astype(np.int32)], annotated_tissue_types[polygon[1]]))

                stop_y = False

                for row in range(rows):
                    stop_x = False

                    for col in range(cols):

                        # Calculate patch coordinates
                        patch_x = int(col * (patch_size - px_overlap))
                        patch_y = int(row * (patch_size - px_overlap))

                        if patch_y + patch_size >= tile_size:
                            stop_y = True
                            patch_y = tile_size - patch_size

                        if patch_x + patch_size >= tile_size:
                            stop_x = True
                            patch_x = tile_size - patch_size

                        global_x = patch_x + tile_x
                        global_y = patch_y + tile_y

                        patch = tile[patch_y: patch_y + patch_size, patch_x: patch_x + patch_size, :]

                        if np.sum(patch) == 0:
                            break

                        # check if the patch is annotated
                        annotated = False
                        if annotations is not None:
                            patch_mask = tile_annotation_mask[
                                         patch_y: patch_y + patch_size, patch_x: patch_x + patch_size
                                         ]

                            labels = self.get_labels_with_enough_tissue_annotated(label_dict, patch_mask)
                            if labels is not None:
                                if len(labels) > 1:
                                    self.update_overlapping_annotations_file(
                                        slide_name, verbose=self.config["overlapping_annotations_verbose"])

                                for label in labels:
                                    # this check is done to ensure that non-tumor tissue (unannotated) is handled
                                    # properly
                                    if self.config["label_dict"][label]["annotated"]:
                                        annotated = True

                        else:
                            labels = "unlabeled"

                        if labels is not None:
                            if self.annotated_only and annotated or not self.annotated_only:
                                if slide_name is not None:

                                    file_name = (
                                            slide_name + "_" + str(global_x) + "_" + str(global_y) + "." + output_format
                                    )
                                else:
                                    file_name = (
                                            str(patch_nb) + "_" + str(global_x) + "_" + str(global_y) + "." +
                                            output_format
                                    )

                                patch = Image.fromarray(patch)

                                for label in labels:
                                    patch.save(os.path.join(self.output_path, label, file_name), format=output_format)

                                    patch_dict.update(
                                        {
                                            patch_nb: {
                                                "slide_name": slide_name,
                                                "patch_path": os.path.join(label, file_name),
                                                "label": label,
                                                "x_pos": global_x,
                                                "y_pos": global_y,
                                                "patch_size": patch_size,
                                            }
                                        }
                                    )
                                    patch_nb += 1
                        if stop_x:
                            break
                    if stop_y:
                        break

        return patch_dict

    def export_dict(self, dictionary, metadata_format, filename):

        if metadata_format == "json":
            file = os.path.join(self.output_path, filename + ".json")
            with open(file, "w") as json_file:
                json.dump(dictionary, json_file, indent=4)
        elif metadata_format == "csv":
            df = pd.DataFrame(dictionary.values())
            file = os.path.join(self.output_path, filename + ".csv")
            df.to_csv(file, index=False)
        else:
            print("Could not write metadata. Metadata format has to be json or csv")

    def save_thumbnail(self, mask, slide_name, level, output_format="png"):

        remap_color = ((0, 0, 0), (255, 255, 255))

        process_level = level
        img = self.slide.read_region([0, 0], process_level, self.slide.level_dimensions[process_level])

        # Remove Alpha
        img = np.array(img)[:, :, 0:3]

        if remap_color is not None:
            indizes = np.all(img == remap_color[0], axis=2)
            img[indizes] = remap_color[1]

            copy_img = img[mask.astype(bool), :]

            median_filtered_img = cv2.medianBlur(img, 11)
            median_filtered_img[mask.astype(bool)] = copy_img

            img = median_filtered_img

        file_name = os.path.join(self.config["output_path"], slide_name, "thumbnail." + output_format)
        plt.imsave(file_name, img, format=output_format)

    def init_generic_tiff(self):

        unit_dict = {"milimeter": 1000, "centimeter": 10000, "meter": 1000000}
        self.scanner = "generic-tiff"

        assert self.slide.properties["tiff.ResolutionUnit"] in unit_dict.keys(), (
                "Unknown unit " + self.slide.properties["tiff.ResolutionUnit"]
        )

        factor = unit_dict[self.slide.properties["tiff.ResolutionUnit"]]

        # convert to mpp
        self.res_x = factor / float(self.slide.properties["tiff.XResolution"])
        self.res_y = factor / float(self.slide.properties["tiff.YResolution"])

    def init_aperio(self):
        self.scanner = "aperio"

        self.res_x = float(self.slide.properties["openslide.mpp-x"])
        self.res_y = float(self.slide.properties["openslide.mpp-y"])

    def init_mirax(self):
        self.scanner = "mirax"
        self.res_x = float(self.slide.properties["openslide.mpp-x"])
        self.res_y = float(self.slide.properties["openslide.mpp-y"])

    def init_unknown(self):
        try:
            self.scanner = self.slide.properties["openslide.vendor"]
            self.res_x = float(self.slide.properties["openslide.mpp-x"])
            self.res_y = float(self.slide.properties["openslide.mpp-y"])
        except Exception as e:
            print(e)

    def init_patch_calibration(self):

        # check scanner type
        if self.slide.properties["openslide.vendor"] == "aperio":
            self.init_aperio()
        elif self.slide.properties["openslide.vendor"] == "generic-tiff":
            self.init_generic_tiff()
        elif self.slide.properties["openslide.vendor"] == "mirax":
            self.init_mirax()
        else:
            self.init_unknown()
        # future vendors
        # elif ...

        assert self.scanner, "Not integrated scanner type, aborting"

    def process_slide(self, slide):
        slide_name = os.path.basename(slide)
        slide_name = os.path.splitext(slide_name)[0]

        print("Processing", slide_name, "process id is", os.getpid())

        try:
            annotation_path = os.path.join(
                self.config["annotation_dir"], slide_name + "." + self.config["annotation_file_format"]
            )
            if os.path.exists(annotation_path):

                annotated = True
                self.annotation_dict = self.load_annotation(annotation_path)
            else:
                annotated = False
                self.annotation_dict = None

        except Exception as e:
            self.print_and_log_slide_error(slide_name, e, "process_slide - load_annotations")
            return 0

        self.make_dirs(
            output_path=self.config["output_path"],
            slide_name=slide_name,
            label_dict=self.config["label_dict"],
            annotated=annotated,
        )

        slide_path = os.path.join(self.config["slides_dir"], slide)
        try:
            level = self.load_slide(slide_path)
        except Exception as e:
            self.print_and_log_slide_error(slide_name, e, "load_slide")
            return 0

        if self.config["calibration"]["use_non_pixel_lengths"]:
            try:
                self.init_patch_calibration()
            except Exception as e:
                self.print_and_log_slide_error(slide_name, e, "init_patch_calibration")
                return 0

        if self.config["use_tissue_detection"]:
            mask, level = self.apply_tissue_detection(level=level, show=self.config["show_mode"])
        else:
            mask = np.ones(shape=self.slide.level_dimensions[level]).transpose()
        try:
            tile_size = self.determine_tile_size(level)
        except Exception as e:
            self.print_and_log_slide_error(slide_name, e, "determine_tile_size")
            return 0
        try:
            tile_dict = self.get_relevant_tiles(
                mask,
                tile_size=tile_size,
                min_coverage=self.config["tissue_coverage"],
                level=level,
                show=self.config["show_mode"],
            )
        except Exception as e:
            self.print_and_log_slide_error(slide_name, e, "get_relevant_tiles")
            return 0

        # Calibrated or non calibrated patch sizes
        if self.config["calibration"]["use_non_pixel_lengths"]:
            try:
                patch_dict = self.extract_calibrated_patches(
                    tile_dict,
                    level,
                    self.annotation_dict,
                    self.config["label_dict"],
                    overlap=self.config["overlap"],
                    annotation_overlap=self.config["annotation_overlap"],
                    slide_name=slide_name,
                    output_format=self.config["output_format"]
                )
            except Exception as e:
                self.print_and_log_slide_error(slide_name, e, "extract_calibrated_patches")
                return 0
        else:
            try:
                patch_dict = self.extract_patches(
                    tile_dict,
                    level,
                    self.annotation_dict,
                    self.config["label_dict"],
                    overlap=self.config["overlap"],
                    annotation_overlap=self.config["annotation_overlap"],
                    patch_size=self.config["patch_size"],
                    slide_name=slide_name,
                    output_format=self.config["output_format"],
                )
            except Exception as e:
                self.print_and_log_slide_error(slide_name, e, "extract_patches")
                return 0

        self.export_dict(patch_dict, self.config["metadata_format"], "tile_information")
        try:
            self.save_thumbnail(mask, level=level, slide_name=slide_name,
                                output_format=self.config["output_format"])
            print("Finished slide ", slide_name)

        except Exception as e:
            self.print_and_log_slide_error(slide_name, e, "save_thumbnail")
            return 0

    @staticmethod
    def read_slide_file(slide_file_path, ext_list):

        slide_list = []

        with open(slide_file_path) as file:
            lines = file.read().splitlines()

        for line in lines:
            if os.path.isdir(line):
                for ext in ext_list:
                    for file in Path(line).resolve().glob("**/*" + ext):
                        slide = str(file)
            else:
                slide = line

            slide_list.append(slide)

        return slide_list

    @staticmethod
    def init(l):
        global lock
        lock = l

    @staticmethod
    def get_slide_name_from_slide_path(slide_path):
        return os.path.splitext(os.path.basename(slide_path))[0]

    def slides2patches(self):

        l = multiprocessing.Lock()

        extensions = [".tif", ".svs", ".mrxs"]
        slide_list = []

        if self.config["slides_file"] is not None:
            print("Using slide file: " + self.config["slides_file"])
            slide_list = self.read_slide_file(self.config["slides_file"], extensions)
        else:
            for extension in extensions:
                for file in Path(self.config["slides_dir"]).resolve().glob("**/*" + extension):
                    slide_list.append(file)

        self.annotation_list = []
        if os.path.exists(self.config["annotation_dir"]):
            annotation_list = os.listdir(self.config["annotation_dir"])
            self.annotation_list = [os.path.splitext(annotation)[0] for annotation in annotation_list]

        missing_annotations = []
        annotated_slides = [
            name
            if os.path.splitext(os.path.basename(name))[0] in self.annotation_list
            else missing_annotations.append(os.path.splitext(os.path.basename(name))[0])
            for name in slide_list
        ]
        annotated_slides = list(filter(lambda slide: True if slide is not None else False, annotated_slides))

        print("###############################################")
        print("Found", len(annotated_slides), "annotated slides")
        print("###############################################")
        print("Found", len(missing_annotations), "unannotated slides")
        print("###############################################")
        if not self.config["use_tissue_detection"]:
            print("Tissue detection deactivated")
            print("###############################################")

        if self.config["skip_unlabeled_slides"]:
            slide_list = annotated_slides
            print("Processing annotated slides only")

        if not os.path.exists(self.config["output_path"]):
            os.makedirs(self.config["output_path"])

        with open(os.path.join(self.config["output_path"], "error_log.txt"), "w"):
            pass

        if not len(slide_list) == 0:
            slide_list = sorted(slide_list)

            # writing this to file to work somewhat elegantly around multiprocessing without much restructuring of the
            # existing code
            with (open(os.path.join(self.config["output_path"], "overlapping_annotations_present_in_slides.json"), "w")
                  as file):
                json.dump(dict.fromkeys(list(map(self.get_slide_name_from_slide_path, slide_list)), False),
                          file, indent=4)

            if _MULTIPROCESS:
                available_threads = multiprocessing.cpu_count() - self.config["blocked_threads"]
                pool = multiprocessing.Pool(processes=available_threads, initializer=self.init, initargs=(l,))
                pool.map(self.process_slide, slide_list)

            else:
                for slide in slide_list:
                    self.process_slide(slide)

            slide_dict = {}

            if len(annotated_slides) == 0:
                for i in range(len(slide_list)):
                    slide = slide_list[i]
                    slide_name = self.get_slide_name_from_slide_path(slide)
                    slide_dict.update({i: {"slide_name": slide_name,
                                           "slide_path": slide,
                                           }
                                       })
                self.output_path = self.config["output_path"]
                self.export_dict(slide_dict, self.config["metadata_format"], "slide_information")
            else:
                # Save label proportion per slide
                labels = list(self.config["label_dict"].keys())

                with (open(os.path.join(self.config["output_path"],
                                        "overlapping_annotations_present_in_slides.json"), "r") as file):
                    overlapping_annotations_present_in_slide = json.load(file)
                for i in range(len(annotated_slides)):
                    slide = slide_list[i]
                    slide_name = self.get_slide_name_from_slide_path(slide)
                    slide_path = os.path.join(self.config["output_path"], slide_name)

                    n_labeled_tiles = 0
                    n_labels = {}
                    for label in labels:
                        n_label = len(os.listdir(os.path.join(slide_path, label)))
                        n_labels.update({label: n_label})
                        n_labeled_tiles += n_label

                    slide_dict_entry = {}
                    slide_dict_entry.update({"slide_name": slide_name,
                                             "slide_contains_overlapping_annotations":
                                                 overlapping_annotations_present_in_slide[slide_name]})
                    fracs = {}
                    for label, n_label in n_labels.items():
                        slide_dict_entry.update({label: n_label})
                        fracs.update({label: n_label / n_labeled_tiles * 100})
                    slide_dict_entry.update({"total": n_labeled_tiles, "frac": fracs})
                    slide_dict.update({i: slide_dict_entry})

                    self.output_path = self.config["output_path"]
                    self.export_dict(slide_dict, self.config["metadata_format"], "slide_information")

            # Save used config file
            file = os.path.join(self.config["output_path"], "config.json")
            with open(file, "w") as json_file:
                json.dump(self.config, json_file, indent=4)

            print("Finished tiling process!")

        else:
            print("###############################################")
            print("WARNING: No slides processed!")
            print("###############################################")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", default=script_dir + "/resources/config.json")
    args = parser.parse_args()

    slide_handler = WSIHandler(config_path=args.config_path)
    slide_handler.slides2patches()
