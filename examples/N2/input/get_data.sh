# copy data from supporting data repository if not already present
if  [ -f traj-std-ev10.xyz ]; then
    echo "Trajectory file already present, skipping download."
else
    cp ../../../npj_supporting_data/notebooks/2_convergence/N2_opes_outputs/traj-std-ev10.xyz.bz2 .
    # Decompress and remove original compressed file
    bzip2 -d traj-std-ev10.xyz.bz2
fi

# copy COLVAR file
cp ../../../npj_supporting_data/notebooks/2_convergence/N2_opes_outputs/COLVAR .
