# Benchmark_VS
This repository contains the code and notebook to reproduce the analyses presented in the paper [*"Integrating Machine Learning-Based Pose Sampling with Established Scoring Functions for Virtual Screening"*](https://doi.org/10.26434/chemrxiv-2025-96kzg-v2) [DOI: 10.26434/chemrxiv-2025-96kzg-v2]

## Structure
### Code

The `utils/analysis_utils.py` contains the code to:
- Calculate the virtual screening (VS) performance
- Extract best poses of each VS setup from docking pose files
- Generate PoseBusters check reports for the docking poses
- Calculate protein-ligand interaction fingerprint (PLIF) similarity between docked and reference molecules
- Calculate Morgan 2-based Tanimoto similarity between docked and reference molecules

### Data

The input data used for the analysis experiments include:
- **dudez**: input molecules as SMILES strings provided in DUDE-Z for each target.
- **docking_poses**: docking poses with DiffDock-L and AutoDock Vina for 43 DUDE-Z targets.
- **plif**: the reference ligands collected with SIENA and the protein-ligand interaction fingerprints generated with ProLIF for docked compounds and reference ligands for each target.
- **posebusters**: the pose validity check results generated with PoseBusters for all docking poses for each target.

The above data can be downloaded from [zenodo](https://zenodo.org/records/14905986). The data can be then placed in `data` such that you have the paths: `data/docking_poses`, `data/dudez`, `data/plif`, and `data/posebusters`. This setup enables the execution of the analysis notebook in `notebooks/` folder.

## Setup Environment

Install python 3.10 in the virtual environment of choice and install the required packages noted in `requirements.txt`.

## Running the analysis experiments

Run the notebook `notebooks/benchmark_vs.ipynb` to reproduce all the results presented in the paper.

The output files and figures will be placed in `data/analysis` by default. This can be customized if needed.

## Citing us
```
@misc{vu2025integrating,
  title={Integrating Machine Learning-Based Pose Sampling with Established Scoring Functions for Virtual Screening},
  author={Vu, Thi Ngoc Lan and Fooladi, Hosein and Kirchmair, Johannes},
  year={2025}
  url={https://doi.org/10.26434/chemrxiv-2025-96kzg-v2}
}
```






