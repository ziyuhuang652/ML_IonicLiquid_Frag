# Simulation drivers

`../../bash/run_reaxff_42.sh` reproduces the seven-energy, six-orientation
ReaxFF campaign. Set `LAMMPS_EXECUTABLE` to a LAMMPS build with REAXFF support.

`../../bash/run_mace_polar_42.sh` launches the MACE-POLAR-1 wall campaign with the
paper-six orientation scheme and 40 Å maximum-separation stop.

The historical MACE-medium driver is archived at
`../../backup/scripts/historical_mace_medium_driver.py`. It contains an invalid
assumption about ASE `LennardJones(pair_parameters=...)`; see
`../../backup/misc/KNOWN_ISSUES.md`. It is not an endorsed production driver.

The analysis environment does not install GPU simulation packages. Record exact
PyTorch, MACE, CUDA, ASE, LAMMPS and model versions for replacement campaigns.
