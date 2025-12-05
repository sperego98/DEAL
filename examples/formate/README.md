## Formate dehydrogenation on Cu
- Ref.: Batzner et al., *Nature Communications*, 13,2453 (2022), doi:[10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5).
- Data: Formate dehydrogenation on Cu(110), series of MD simulations initiated from NEB images.
- Summary: 
    1. Filter the structures based on GP predictive variance. 
    2. Visualize the selected structures with chemiscope.


___ 

### INSTRUCTION

#### Input
Get trajectory (xyz) from supporting data 
```bash
cd input 
bash download.sh
cd ..
```

#### Colvar (optional)
Evaluate relevant CVs (e.g. coordination numbers) using PLUMED
```bash
cd colvar
plumed driver --plumed plumed.dat --ixyz ../input/fcu.xyz --length-units A --box 10.638,10.03,30.0
cd ..
```

#### Selection
Run DEAL with different thresholds and visualize structures
```bash
deal -c input.yaml
``` 
Consider running it on a HPC cluster or on a large-memory machine.
An example SLURM script to run DEAL on Daint@Alps-CSCS HPC system is provdied for [`N2 example`](../N2/README.md).

#### Copy results (optional)
If you don't want to run DEAL, you can copy the results from the [`examples-results/formate/selection/`](examples-results/formate/selection/) folder.
```bash
cp -r ../../examples-results/formate/selection selection
```

#### Analyze results
The selection can be analyzed with the jupyter notebook [`analyse_results.ipynb`](`analyse_results.ipynb`).

The results `*_chemiscope.json.gz` can also be directly visualized online at https://chemiscope.org/



