# merge docked poses into a single file
import os, sys
import glob
from rdkit import Chem


def get_mol_name(file_path: str):
    """Get molecule name from file path.
    :param file_path: Path to docked pose file
    :return: Molecule name in the format mol_id_rank
    """
    mol_id = os.path.basename(os.path.dirname(file_path))
    rank = os.path.basename(file_path).split('.')[0].split('_')[0][4:]
    return f"{mol_id}_{rank}"

def merge_poses(indir: str, output_file: str, log_file: str):
    """Merge all docked poses in a directory into a single sdf file.
    :param indir: Directory containing docked poses organized as indir/molID/rank*.sdf
    :param output_file: Output sdf file containing all docked poses, with molID_rank as molecule name
    :param log_file: path to log file
    :param check_charge_state: Check if mol_id contains charge state info, add _1 if not (default: False)
    """
    file_paths = glob.glob(os.path.join(indir, '*', 'rank*.sdf'))    # indir/mol_id/rank*.sdf

    mols = list()

    original_stdout = sys.stdout
    log_file = open(log_file, 'w')
    sys.stdout = log_file
    
    print(f"Reading {len(file_paths)} poses from {indir}...")

    for file_path in file_paths:
        mol_name = get_mol_name(file_path)

        mol = Chem.SDMolSupplier(file_path, removeHs=False)[0]

        if mol is None:
            print(f"Error reading {file_path}") 
            continue

        mol.SetProp('_Name', mol_name)

        mols.append(mol)

    with Chem.SDWriter(output_file) as writer:
        for i, mol in enumerate(mols):
            writer.write(mol)
            if i % 1000 == 0 or i == len(mols) - 1:
                writer.flush()

    print(f"Saved {len(mols)} poses to {output_file}.", file=log_file)

    sys.stdout = original_stdout
    log_file.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Merge docked poses into a single file')
    parser.add_argument('--indir', type=str, help='Directory containing docked poses')
    parser.add_argument('--output_file', type=str, help='Output file')
    parser.add_argument('--log_file', type=str, help='Log file')
    args = parser.parse_args()

    merge_poses(args.indir, args.output_file, args.log_file)