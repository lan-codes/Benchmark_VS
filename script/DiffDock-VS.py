# Execute virtual screening using DiffDock-L

from argparse import ArgumentParser
import os
import torch
import pandas as pd
import subprocess
from multi_gpus import gpuPool

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--gpu_nodes', type=str, default=None, help='The GPU node to use for inference (e.g. "0_1"). Default: using all available GPUs.')
    parser.add_argument('--num_jobs', type=int, default=1, help='Number of jobs to split ligands. Default: 1')
    parser.add_argument('--config', type=str, default="default_inference_args.yaml", help='Path to the config file')
    parser.add_argument('--protein_path', type=str, required=True, default=None, help='Path to the protein file')
    parser.add_argument('--ligand_path', type=str, required=True, default=None, help='Either a text file containing SMILES strings or the path to separate molecule files that rdkit can read')
    parser.add_argument('--out_dir', type=str, required=True, default='output', help='Output directory')

    return parser.parse_args()

def main(args):
    '''
    Execute virtual screening with DiffDock-L.
    
    Input:
        Protein: 
            - prepared pdb file without ligand
        Ligands: expected input file format:
            - text file: .smi, .ism, .csv or .txt, each line containing "SMILES_strings [molecule id]"
            - directory: directory contains separate molecule files that rdkit can read (.sdf, .mol2, .pdbqt, .pdb)
    
    Output:
        job_cmds (dir): shell scripts to launch jobs
        job_csvs (dir): csv files containing protein, ligand and complex names
        job_logs (dir): log files for each job
        poses (dir): complex_name/rank*.sdf files
    '''

    os.makedirs(args.out_dir, exist_ok=True)
    poses_out_dir = os.path.join(args.out_dir, 'poses')
    jobCSV_dir = os.path.join(args.out_dir, 'job_csvs')
    os.makedirs(jobCSV_dir, exist_ok=True)
    jobCMD_dir = os.path.join(args.out_dir, 'job_cmds')
    os.makedirs(jobCMD_dir, exist_ok=True)
    jobLOG_dir = os.path.join(args.out_dir, 'job_logs')
    os.makedirs(jobLOG_dir, exist_ok=True)
 
    # split ligands
    ligand_path = args.ligand_path
    if os.path.isdir(ligand_path):
        ligands = [os.path.join(ligand_path, f) for f in os.listdir(ligand_path)]
        complex_names = [os.path.basename(ligand) for ligand in ligands]
    else:
        ligand_df = pd.read_csv(ligand_path, header=None, sep='\t')
        ligands = ligand_df[0].tolist()
        if len(ligand_df.columns) > 1:              # expect complex names in the second column
            complex_names = ligand_df[1].tolist()   # should be unique
        else:
            complex_names = [f"mol_{i+1}" for i in range(len(ligands))]
    
    complex = [[complex_names[i], args.protein_path, ligands[i], ] for i in range(len(ligands))]

    def split(a, n):
        if n > len(a):
            print("more jobs than files, launching 1 job per file")
            return [a[i:i+1] for i in range(len(a))]
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    chunks = list(split(complex, args.num_jobs))

    # write jobCSVs and jobCMDs
    jobCMDs = []
    for i, chunk in enumerate(chunks):
        chunk_df = pd.DataFrame(chunk, columns=['complex_name', 'protein_path', 'ligand_description'])
        chunk_df['protein_sequence'] = ''
        jobCSV = os.path.join(jobCSV_dir, f'job_csv_{i+1}.csv')
        chunk_df.to_csv(jobCSV, index=False)

        jobCMD = f"python -m inference --config {args.config} --protein_ligand_csv {jobCSV} --out_dir {poses_out_dir} 2>&1 | tee {jobLOG_dir}/job_{i+1}_log.txt"
        with open(os.path.join(jobCMD_dir, f'job_{i+1}.sh'), 'w') as f:
            f.write(jobCMD)

        jobCMDs.append(jobCMD)

    # launch jobs
    if torch.cuda.is_available():
        print(f"Launched {args.num_jobs} jobs on GPU node(s).")
        gpuPool(jobCMDs, gpu_node=args.gpu_nodes)

    else:
        print(f"Launched {args.num_jobs} jobs on the CPU.")
        for jobCMD in jobCMDs:
            subprocess.run(jobCMD, shell=True)

    print("Docking finished.")
    
    
if __name__ == '__main__':
    args = parseArguments()
    main(args)