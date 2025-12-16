## Examples

### [1 Active learning](1_activelearning_N2/README.md)
#### $N_2$ decomposition on FeCo

- Data: Nitrogen dissociation on FeCo(110) surface, collected via OPES enhanced sampling simulations, taken from Perego and Bonati, *npj Computational Materials* 10, 291 (2024), doi: [10.1038/s41524-024-01481-6](https://doi.org/10.1038/s41524-024-01481-6)

- Summary: 
    1. pre-select high uncertainty configurations based on ensemble-based standard deviation
    2. filter them using the GP predictive variance to identify structurally different configurations
    3. analyze distribution of selected CVs
___

### [2 Trajectory subsampling](2_subsampling_formate/README.md)
#### Formate dehydrogenation on Cu
- 
- Data: Formate dehydrogenation on Cu(110), series of MD simulations initiated from NEB images, taken from Batzner et al., *Nature Communications*, 13,2453 (2022), doi:[10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5).

- Summary: 
    1. Filter the structures based on GP predictive variance. 
    2. Visualize the selected structures with chemiscope.

### [3 Pre-selection](3_preselection/README.md)

Illustrate how to preprocess trajectory based on uncertainty (query-by-committe), using MACE or DeepMD.

