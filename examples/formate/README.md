## Formate dehydrogenation on Cu
- Ref.: Batzner et al., *Nature Communications*, 13,2453 (2022), doi:[10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5).
- Data: Formate dehydrogenation on Cu(110), series of MD simulations initiated from NEB images.
- Summary: 
    1. Filter the structures based on GP predictive variance. 
    2. Visualize the selected structures with chemiscope.


___ 

#### Structure

- **input**
    - Original trajectory (xyz) from OPES simulation and corresponding COLVAR file (used for monitoring the distribution of CVs)
- **colvar**
    - Evaluate relevant CVs (e.g. coordination numbers) using PLUMED
- **selection**
    - Run DEAL with different thresholds and visualize structures
    
    ```
    deal -c input.yaml
    ``` 
    (consider running it on a HPC cluster or on a high-memory machine)

    

