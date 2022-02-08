# wsi_preprocessing

## Processing and tiling of histological slides

openslide-based processing and filtering (Only tissue filtering right now, more will follow) 
The process can be configured using a config json file.

The tissue detection is processed on a higher level to speed up the process. Thereby rough tiles will be sampled and discarded if there isnt enough tissue coverage. The tiles will then be divided into patches for training etc.

Supported annotation types are .xml (Camelyon17) or .geojson (QuPath)
Right now only binary annotation types are supported (tumor - non-tumor)

| Dictionary Entry | Explanation |
| ----------- | ----------- |
| tissue_coverage | Threshold [0,1] for how much tissue coverage is necessary, default is 0.75|
| processing_level | Level of downsampling by openslide - Lowering the level will increase precision but more time is needed, default is 5| 
| tile_size |Pixel size of the rough tiles default is 16|
| overlap | Value [0,1] to set the overlap between neighbouring patches | 
| patch_size | Output pixel size of the quadratic patches |
| slides_dir | Directory where the different slides and subdirs are located  | 
| annotation_dir | Directory where the annotations are located |
| annotation_file_format | File format of the input annotations | 
| output_path | Output directory to where the resulting images will be stored |
| skip_unlabeled_slides | Boolean to skip slides without an annotation file | 
| output_format | Image output format default is "png" |
| show_mode | Boolean to enable plotting of some intermediate results/visualizations | 
| label_dict |  Structure to set up the operator and the threshold for checking the coverage of a certain class|
| type | Operator type [ "==", ">=", "<="]| 
| threshold | Coverage threshold for the individual class |
