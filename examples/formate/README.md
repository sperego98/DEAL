## Formate dehydrogenation on Cu
- Ref.: Batzner et al., *Nature Communications*, 13,2453 (2022), doi:[10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5).
- Data: Formate dehydrogenation on Cu(110), series of MD simulations initiated from NEB images.
- Summary: 
    1. Filter the structures based on GP predictive variance. 
    2. Visualize the selected structures with chemiscope.


___ 

#### Instructions

- **input**
    - Get trajectory (xyz) from supporting data 
    ```
    cd input 
    bash download.sh
    ```
- **colvar**
    - (optional) evaluate relevant CVs (e.g. coordination numbers) using PLUMED

    ```
    cd colvar
    plumed driver --plumed plumed.dat --ixyz ../input/fcu.xyz --length-units A --box 10.638,10.03,30.0
    ```
- **selection**
    - Run DEAL with different thresholds and visualize structures

    ```
    cd selection
    deal -c input.yaml
    ``` 
    (consider running it on a HPC cluster or on a large-memory machine)

    The selection can be analyzed with the jupyter notebook [`analyze_and_view.ipynb`](`selection/analyze_and_view.ipynb`).
    

