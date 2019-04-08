# Geoglam with Machine Learning

For the technical discussion of this project, please refer [here](technical.md)

# Directory structures

* `dist_forest_time_tile_full_84_features` contains the training and test code for the main experiment. More experiments may be added later.	

* `geoglam` contains the code that computes Geoglam fractional cover (FC) products using the trained models. This geoglam code base is a fork from [here](https://github.com/nci/geoglam). The only [code changes](geolam/fc_prod/main.py#L181) are using the trained random forest model instead of the original non-negative least square (nnls) solution.

* `visualisation/vis.sh` contains code that creates a mosaic GeoTiff image of Australia for visually inspect the model outputs.

* `visualisation/outputs/ground_truth` contains the GeoTiff images of the FC products computed using the original nnls method (i.e. the grough truth images as the baseline we want to compare against).

* `visualisation/outputs/predicted` contains the GeoTiff images inferenced from the trained random forest models. We want to visually compare these images against the ground truth images.

* `inference_australia.sh` calls `inference_geoglam.sh` to [compute the FC products](geolam/fc_prod/main.py#L181) using the trained random forest models.

# Training

* The files related to training are under `dist_forest_time_tile_full_84_features`. 

* The random forest models are trained on NCI's [raijin](http://nci.org.au/systems-services/peak-system/raijin) HPC systems. The training is distributed across three nodes for each Geoglam band (i.e. `pv`, `npv`, `bare_soil`). The PBS scripts for submitting the training jobs are `train*.pbs`. 

* `rf*.model` are the pre-trained random forest models, which can be applied directly for inference. 

* The training logs can be found under `logs`.

For detailed training methodology and experimental setup, please refer to the [technical discussion](technical.md).
