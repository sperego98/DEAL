## Examples

### [$N_2$ decomposition on FeCo](N2/README.md)

- Ref.: Perego and Bonati, *npj Computational Materials* 10, 291 (2024), doi: [10.1038/s41524-024-01481-6](https://doi.org/10.1038/s41524-024-01481-6)
- Data: Nitrogen dissociation on FeCo(110) surface, collected via OPES enhanced sampling simulations.
- Summary: 
    1. pre-select high uncertainty configurations based on ensemble-based standard deviation
    2. filter them using the GP predictive variance to identify structurally different configurations
    3. analyze distribution of selected CVs
- See also the notebooks in: `npj_supporting_data/notebooks/2_convergence/` ([link](../npj_supporting_data/notebooks/2_convergence/))
___

### [Formate dehydrogenation on Cu](formate/README.md)
- Ref.: Batzner et al., *Nature Communications*, 13,2453 (2022), doi:[10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5).
- Data: Formate dehydrogenation on Cu(110), series of MD simulations initiated from NEB images.
- Summary: 
    1. Filter the structures based on GP predictive variance. 
    2. Visualize the selected structures with chemiscope.


