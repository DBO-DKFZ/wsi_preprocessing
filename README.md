# wsi_preprocessing

## Processing and tiling of histological slides

openslide-based processing and filtering (Only tissue filtering right now, more will follow) 
The process can be configured using a config json file.

The tissue detection is processed on a higher level to speed up the process. Thereby rough tiles will be sampled and discarded if there isnt enough tissue coverage. The tiles will then be divided into patches for training etc.

Supported annotation types are .xml (Camelyon17 and some other public datasets) or .geojson (QuPath)
Right now only binary annotation types are supported (tumor - non-tumor)

Supported slide formats are .tif and .svs right now

### Usage:

This script is designed to be used together with CuPath in case there are no annotations.
Main file is "tile_generator.py" - Configure the process via the config file and execute this file to start the process

### Additional information:

NOTE:
Right now there is a bug on Unix systems regarding openslide where image data isnt properly loaded. To fix this follow:
https://github.com/openslide/openslide-python/issues/58#issuecomment-883446558

### Config Explanation:

| Dictionary Entry                                        | Explanation                                                                                                                                                                                                                     |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tissue_coverage                                         | Threshold [0,1] for how much tissue coverage is necessary, default is 0.75                                                                                                                                                      |
| keep_annotated_tiles_despite_too_little_tissue_coverage | legacy option. Old behaviour: Keep annotated tiles even if not covered by tissue. New behaviour (to allow easier tile clean-up around the edges): discard tiles with too little tissue coverage regardless of annotation status.|
| processing_level                                        | Level of downscaling by openslide - Lowering the level will increase precision but more time is needed, default is 5                                                                                                            | 
| blocked_threads                                         | Number of threads that wont be used by the program                                                                                                                                                                              |
| patches_per_tile                                        | Number of patches used for lower resolution operations like tissue detection                                                                                                                                                    | 
| overlap                                                 | Value [0,1[ to set the overlap between neighbouring unannotated patches                                                                                                                                                         |
| annotation_overlap                                      | Value [0,1[ to set the overlap between neighbouring annotated patches                                                                                                                                                           | 
| patch_size                                              | Output pixel size of the quadratic patches                                                                                                                                                                                      |
| slides_dir                                              | Directory where the different slides and subdirs are located                                                                                                                                                                    |
| slides_file                                             | txt file containing paths to all slides to process (absolute paths)                                                                                                                                                             |
| annotation_dir                                          | Directory where the annotations are located                                                                                                                                                                                     |
| annotation_file_format                                  | File format of the input annotations ("xml","geojson")                                                                                                                                                                          | 
| output_path                                             | Output directory to where the resulting images will be stored                                                                                                                                                                   |
| skip_unlabeled_slides                                   | Boolean to skip slides without an annotation file                                                                                                                                                                               | 
| save_annotated_only                                     | Boolean to only save annotated patches                                                                                                                                                                                          |
| discard_preexistent_patches                             | Remove already tiled patches                                                                                                                                                                                                    |     
| output_format                                           | Image output format default is "png"                                                                                                                                                                                            |
| show_mode                                               | Boolean to enable plotting of some intermediate results/visualizations                                                                                                                                                          | 
| label_dict                                              | Structure to set up the operator and the threshold for checking the coverage of a certain class. Up to one unannotated tissue type (e.g. non-tumor) is possible and must go first for implementation reasons.                   |
| type                                                    | Operator type [ "==", ">=", "<="]                                                                                                                                                                                               | 
| threshold                                               | Coverage threshold for the individual class                                                                                                                                                                                     |
