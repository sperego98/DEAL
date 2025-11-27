## $N_2$ decomposition on FeCo

- Ref.: Perego and Bonati, *npj Computational Materials* 10, 291 (2024), doi: [10.1038/s41524-024-01481-6](https://doi.org/10.1038/s41524-024-01481-6)
- Data: Nitrogen dissociation on FeCo(110) surface, collected via OPES enhanced sampling simulations.
- Summary: 
    1. pre-select high uncertainty configurations based on ensemble-based standard deviation
    2. filter them using the GP predictive variance to identify structurally different configurations
    3. analyze distribution of selected CVs
- See also the notebooks in: `DEAL/npj_supporting_data/notebooks/2_convergence/`

___

#### Instructions

- **input**
    - Copy original trajectory (xyz) from OPES simulation and corresponding COLVAR file (used for monitoring the distribution of CVs)
    ```
    cd input 
    bash get_data.sh 
    ```
- **selection**
    - pre-process data and run DEAL: [example-N2.ipynb](example-N2.ipynb)