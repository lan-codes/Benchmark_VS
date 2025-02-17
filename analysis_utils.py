from rdkit import Chem
import re
import os
import json
import numpy as np
import pandas as pd
import rdkit
from rdkit.Chem import PandasTools, AllChem, DataStructs
import rdkit.ML.Scoring.Scoring
from rdkit import RDLogger
from multiprocessing import Pool
from functools import partial
import requests
import prolif as plf    # for plf.ResidueId.from_string()
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def get_label_dict(label_dir: str, target: str) -> dict:
    """Get the label dictionary for the target in label_dir."""
    active_np = np.genfromtxt(os.path.join(label_dir, f"{target.upper()}", "ligands.teb.smi"), dtype=str, comments='None')
    decoy_np = np.genfromtxt(os.path.join(label_dir, f"{target.upper()}", "decoys.teb.smi"), dtype=str, comments='None')
    label_dict = dict()
    for compound_id in active_np[:,1]:
        label_dict[compound_id] = 1
    for compound_id in decoy_np[:,1]:
        label_dict[compound_id] = 0
    return label_dict

def read_sdf(input_file: str, docking_software: str, scoring_function: str, lig_states: bool):
    docking_software = docking_software.lower()
    scoring_function = scoring_function.lower()
    
    # suppress RDKit warning
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    frame = PandasTools.LoadSDF(input_file, molColName='molecule')

    # get compound_id
    if lig_states:
        frame['compound_id'] = frame['ID'].apply(lambda x: re.split('_| ', x)[0] + '_' + re.split('_| ', x)[1])
    else:
        frame['compound_id'] = frame['ID'].apply(lambda x: re.split('_| ', x)[0])

    # get score
    if docking_software == "vina":
        if scoring_function == "vina":
            frame['score'] = frame['meeko'].apply(lambda x: json.loads(x)['free_energy'])
        elif scoring_function == "gnina":
            frame['score'] = frame['CNNaffinity']

    elif docking_software in ["diffdock", "diffdock-l", "diffdock_l"]:
        if scoring_function == "vina":
            frame['score'] = frame['minimizedAffinity']
        if scoring_function == "vina_score":
            frame['score'] = frame['meeko'].apply(lambda x: json.loads(x)['free_energy'])
        elif scoring_function == "gnina":
            frame['score'] = frame['CNNaffinity']

    frame['score'] = frame['score'].astype(float)

    id_score_df = frame[['compound_id', 'score']]

    # for ranking poses
    if scoring_function in ['vina', 'dock', 'glide']:
        bigger_is_better = False
    else:
        bigger_is_better = True

    id_score_df = id_score_df.sort_values('score', ascending=not(bigger_is_better)).groupby('compound_id').head(1).reset_index()

    return id_score_df


def read_mol2(input_file, docking_software, scoring_function, lig_states):
    docking_software = docking_software.lower()
    scoring_function = scoring_function.lower()

    id_score = {}

    # for ranking poses
    if scoring_function in ['vina', 'dock']:
        bigger_is_better = False
    else:
        bigger_is_better = True

    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("##########"):
                prop = line.replace("##########", "").strip().split(":")
                if prop[0] == "Name":
                    compound_id = prop[1].strip()
                if prop[0] == "Total Energy":
                    score = float(prop[1].strip())

            if line.startswith("@<TRIPOS>MOLECULE"):    # collect score                
                if compound_id in id_score.keys():
                    if bigger_is_better: id_score[compound_id] = max(score, id_score[compound_id])
                    else: id_score[compound_id] = min(score, id_score[compound_id])
                else: id_score[compound_id] = score

    id_score_df = pd.DataFrame(id_score.items(), columns=['compound_id', 'score'])

    return id_score_df


def read_infile(input_file: str, docking_software: str, scoring_function: str, lig_states: bool):
    """
    This function is used to read the input file.

    Parameters:
    input_file (str): The path to the input file.
    scoring_function (str): The scoring function used to rank the molecules.

    Returns:
    a dataframe containing the compound_id and score, sorted from the best to the worst score (scoring_function dependent).
    """
    if input_file.endswith('.sdf'):
        id_score_df = read_sdf(input_file, docking_software, scoring_function, lig_states)
    elif input_file.endswith('.mol2'):
        id_score_df = read_mol2(input_file, docking_software, scoring_function, lig_states)

    print(f"Read {os.path.basename(input_file)}: {len(id_score_df)} molecules.")
    
    return id_score_df


def vs_performance(input_file: str, docking_software: str, scoring_function: str, label_dict: dict(), lig_states: bool=True):
    """
    This function is used to calculate the virtual screening performance of the result given in the input file.

    Parameters:
    input_file (str): The path to the input file.
    scoring_function (str): The scoring function used to rank the molecules.
    label_dict (dict): The dictionary containing the label (True (active) or False (inactive/decoy)) of the molecules.
    lig_states (bool): Whether to differentiate ligand states. Default is True, DUDEZ specific.

    Returns:
    Score(s) of interest: auc_score, ef_scores, bedroc_score
    """
    # check if input_file exists
    if not os.path.exists(input_file):
        print(f"{input_file} not found.")
        return None, [None, None], None

    docking_software = docking_software.lower()
    scoring_function = scoring_function.lower()

    # get the SORTED dataframe from the input file
    id_score_df = read_infile(input_file, docking_software, scoring_function, lig_states)
    id_score_df['label_id'] = id_score_df['compound_id'].apply(lambda x: re.split('_| |-', x)[0])
    id_score_df['label'] = id_score_df['label_id'].map(label_dict)

    scores = id_score_df[['score', 'label']].values.tolist()
    label_col = 1

    # calculate the score
    auc_score = rdkit.ML.Scoring.Scoring.CalcAUC(scores, label_col)
    ef_frac = [0.01, 0.05]  # EF1%, EF5%
    ef_scores = rdkit.ML.Scoring.Scoring.CalcEnrichment(scores, label_col, ef_frac)    
    bedroc_alpha = 80.5
    bedroc_score = rdkit.ML.Scoring.Scoring.CalcBEDROC(scores, label_col, bedroc_alpha)

    return auc_score, ef_scores, bedroc_score


##### Others #####
def take_best_poses(input_file: str, docking_software: str, scoring_function: str, output_dir: str, prefix: bool=None, lig_states: bool=True):
    """Reads the input file and writes the best poses to the output file."""
    docking_software = docking_software.lower()
    scoring_function = scoring_function.lower()
    
    # suppress RDKit warning
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    frame = PandasTools.LoadSDF(input_file, molColName='pose')
    print(f"Read {prefix} {os.path.basename(input_file)}: {len(frame)} poses")

    # for ranking poses
    if scoring_function in ['vina', 'dock', 'glide']:
        bigger_is_better = False
    else:
        bigger_is_better = True

    # get compound_id
    if lig_states:
        frame['compound_id'] = frame['ID'].apply(lambda x: re.split('_| ', x)[0] + '_' + re.split('_| ', x)[1])
    else:
        frame['compound_id'] = frame['ID'].apply(lambda x: re.split('_| ', x)[0])

    # get score
    if docking_software == "vina":
        if scoring_function == "vina":
            frame['score'] = frame['meeko'].apply(lambda x: json.loads(x)['free_energy'])
        elif scoring_function == "gnina":
            frame['score'] = frame['CNNaffinity']

    elif docking_software in ["diffdock", "diffdock-l", "diffdock_l"]:
        if scoring_function == "vina":
            frame['score'] = frame['minimizedAffinity']
        elif scoring_function == "gnina":
            frame['score'] = frame['CNNaffinity']

    frame['score'] = frame['score'].astype(float)

    id_score_df = frame[['compound_id', 'score', 'pose']]

    id_score_df = id_score_df.sort_values('score', ascending=not(bigger_is_better)).groupby('compound_id').head(1).reset_index()

    # write the best poses to the output file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if prefix:
            output_file = os.path.join(output_dir, f"{prefix}_{docking_software}_{scoring_function}_best_poses.sdf")
        else:
            output_file = os.path.join(output_dir, f"{docking_software}_{scoring_function}_best_poses.sdf")
        w = Chem.SDWriter(output_file)
        for compound_id in id_score_df['compound_id']:
            mol = id_score_df[id_score_df['compound_id'] == compound_id]['pose'].values[0]
            mol.SetProp('_Name', compound_id)
            mol.SetProp('score', str(id_score_df[id_score_df['compound_id'] == compound_id]['score'].values[0]))
            w.write(mol)
        w.close()

    return id_score_df


def get_id_score_label(input_file: str, docking_software: str, scoring_function: str, label_dict: dict(), outfile: str, lig_states: bool=True):
    """
    This function is used to calculate the virtual screening performance of the result given in the input file.

    Parameters:
    input_file (str): The path to the input file.
    scoring_function (str): The scoring function used to rank the molecules.
    label_dict (dict): The dictionary containing the label (True (active) or False (inactive/decoy)) of the molecules.

    Returns:
    Score(s) of interest: auc_score, ef_scores, bedroc_score
    """
    # check if input_file exists
    if not os.path.exists(input_file):
        print(f"{input_file} not found.")
        return None

    docking_software = docking_software.lower()
    scoring_function = scoring_function.lower()

    # get the SORTED dataframe from the input file
    id_score_df = read_infile(input_file, docking_software, scoring_function, lig_states)

    id_score_df['label_id'] = id_score_df['compound_id'].apply(lambda x: re.split('_| |-', x)[0])
    id_score_df['label'] = id_score_df['label_id'].map(label_dict)

    # save the id_score_label to a csv file without index
    id_score_df.drop(columns=['index'], inplace=True)
    id_score_df.to_csv(outfile, index=False)

    return


##### POSE ANALYSIS #####
def read_ligand_expo(file_name: str) -> dict:
    """
    Read Ligand Expo data, try to find a file called
    Components-smiles-stereo-oe.smi in the current directory.
    If you can't find the file, grab it from the RCSB
    :return: Ligand Expo as a dictionary with ligand id as the key
    """
    #file_name = "Components-smiles-stereo-oe.smi"
    try:
        df = pd.read_csv(file_name, sep="\t",
                         header=None,
                         names=["SMILES", "ID", "Name"])
    except FileNotFoundError:
        url = f"http://ligand-expo.rcsb.org/dictionaries/{file_name}"
        print(url)
        r = requests.get(url, allow_redirects=True)
        open('Components-smiles-stereo-oe.smi', 'wb').write(r.content)
        df = pd.read_csv(file_name, sep="\t",
                         header=None,
                         names=["SMILES", "ID", "Name"])
    df.set_index("ID", inplace=True)
    return df.to_dict()


def get_docked_id_mol(mol_dir, target):
    for docking in ["vina", "diffdock_l"]:
        for scoring in ["vina", "gnina"]:
            input_file = os.path.join(mol_dir, f"{target.upper()}_{docking}_{scoring}_best_poses.sdf")
            frame = PandasTools.LoadSDF(input_file, molColName='molecule')  # no protonation, removeHs=True (default)
            yield frame

def keep_actives(frame: pd.DataFrame, label_dict: dict, lig_id: str=None) -> pd.DataFrame:
    """Keep the active compounds in the frame.
    :param frame: dataframe
    :param label_dict: dictionary with compound id as key and label as value
    :param lig_id: column containing ligand id to generate the label id. If None, use the index.
    :return: dataframe with active compounds"""
    frame_copy = frame.copy()
    if lig_id:
        frame_copy["label_id", ""] = frame_copy[lig_id].map(lambda x: x.split("_")[0])
    else:
        frame_copy["label_id", ""] = frame_copy.index.map(lambda x: x.split("_")[0])
    frame_copy["label", ""] = frame_copy["label_id", ""].map(label_dict)
    frame_copy = frame_copy[frame_copy["label", ""] == 1]
    # drop the multi-level columns label_id and label
    frame_copy.drop([("label_id", ""), ("label", "")], axis=1, inplace=True)
    #frame_copy.drop(columns=["label_id", "label"], level=0, inplace=True)
    return frame_copy

def get_reference_mol(working_dir: str, expo_dict: dict=None) -> dict:
    """Get the reference molecules that have the PLIF calculated.
    :param working dir: directory containing ref_plifs.csv and ref_plif_input/pdb_id_ligand_prep_minimized.sdf
    :return: dictionary with ID as key and molecule as value
    """
    ref_plif = pd.read_csv(os.path.join(working_dir, "ref_plifs.csv"), header=[0, 1], index_col=0)

    ref_id_mol = dict()
    for ref_label in ref_plif.index:
        pdb_id = "_".join(ref_label.split("_")[-2:])
        mol_path = os.path.join(working_dir, "ref_plif_input", f"{pdb_id}_ligand_prep_minimized.sdf")
        mol = Chem.SDMolSupplier(mol_path)[0]
        ref_id_mol[f"{ref_label}"] = Chem.RemoveHs(mol)
        
    return ref_id_mol

def tanimoto_similarity(bv, bvs: list):
    """Calculate the Tanimoto similarity between two fingerprints. For multiprocessing.
    :param bv: bit vector
    :param bvs: list of bit vectors"""
    s = DataStructs.BulkTanimotoSimilarity(bv, bvs)
    return s

def list_list_tanimoto_similarity(list1, list2, n_jobs=30):
    """Calculate the pairwise Tanimoto similarity between two lists of fingerprints.
    :param list1: list of fingerprints
    :param list2: list of fingerprints
    :return: array of similarities in the shape of (len(list1), len(list2))"""
    with Pool(n_jobs) as pool:
        sim = pool.map(partial(tanimoto_similarity, bvs=list2), list1)
    return sim

def pairwise_tsim_mol(mol_dict_1: dict, mol_dict_2: dict, fps: list, n_jobs=30):
    """Calculate the pairwise Tanimoto similarity between the docked and reference molecules.
    :param mol_dict_1: dictionary with ID as key and molecule as value
    :param mol_dict_2: dictionary with ID as key and molecule as value
    :param fps: list of fingerprint type. Options: ['ecfp', 'fcfp']
    :return: pandas dataframe of similarities in the shape of (len(mol_dict_1), len(mol_dict_2))"""
    mols_1 = list(mol_dict_1.values())
    mols_2 = list(mol_dict_2.values())
    df_list = list()
    for fp in fps:
        fps_1 = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=(fp=="fcfp")) for mol in mols_1]
        fps_2 = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=(fp=="fcfp")) for mol in mols_2]
        with Pool(n_jobs) as pool:
            sim = pool.map(partial(tanimoto_similarity, bvs=fps_2), fps_1)
        df_list.append(pd.DataFrame(sim, index=mol_dict_1.keys(), columns=mol_dict_2.keys()))
    return df_list

def get_morgan_sim(target: str, paths: dict, fps: list, active_only: bool = False):
    """Get the Morgan-2 based Tanimoto similarity between docked and reference molecules.
    :param target: target name
    :param paths: dictionary with the related paths
    :param fps: list of fingerprint type(s) to calculate. Options: ['ecfp', 'fcfp']
    """
    
    plif_dir = paths["plif_dir"]
    mol_dir = paths["mol_dir"]
    label_dir = paths["label_dir"]

    working_dir = os.path.join(plif_dir, target.upper())
    label_dict = get_label_dict(label_dir, target)

    # get docked molecules
    docked_id_mol = {}
    for frame in get_docked_id_mol(mol_dir, target):
        if active_only:
            frame = keep_actives(frame, label_dict, "ID")
        docked_id_mol.update(dict(zip(frame["ID"], frame["molecule"])))
    print(f"Number of docked molecules: {len(docked_id_mol)}")

    # get reference molecules
    ref_id_mol = get_reference_mol(working_dir)
    print(f"Number of reference molecules: {len(ref_id_mol)}")

    # calculate pairwise Morgan-2 based Tanimoto similarity between docked and reference molecules
    morgan_sim = pairwise_tsim_mol(docked_id_mol, ref_id_mol, fps=fps)

    # format the dataframe(s)
    for i, fp in enumerate(fps):
        morgan_sim_stack = morgan_sim[i].stack().reset_index()
        morgan_sim_stack.columns = ["mol_id", "ref_lig", f"{fp}_sim"]
        morgan_sim[i] = morgan_sim_stack
    # merge the dataframes on mol_id and ref_lig
    if len(fps) > 1:
        morgan_sim_stack = pd.merge(morgan_sim[0], morgan_sim[1], on=["mol_id", "ref_lig"], how="outer")
    else:
        morgan_sim_stack = morgan_sim[0]

    return morgan_sim_stack

def get_unique_id_single(loc_ele, l):
    """Get the unique id from the list.
    :param loc_ele: tuple of (element, location)
    :param l: list of strings
    :return: unique id"""
    element = loc_ele[1]
    occurence = l[:loc_ele[0]+1].count(element)
    return f"{element}_{occurence}"

def get_unique_pose_id(l: list) -> list:
    """Get the unique id from the list.
    :param l: list of strings
    :process: adding the occurence of the string in the list to the end of the string
    :return: list of unique ids"""
    #unique_id = [f"{j}_{l[:i+1].count(i)}" for i, j in enumerate(l)]
    # make it parallel
    # generate a list of tuples with the index and the string in the list

    ele_loc = list(zip(range(len(l)), l))
    with Pool(50) as pool:
        unique_id = pool.map(partial(get_unique_id_single, l=l), ele_loc)
    return unique_id

def plif_sim_pose(ref: pd.DataFrame, query: pd.DataFrame, ignored_interactions: list=None, max_sim: bool = False, n_jobs: int=80):
    """For each pose in query: calculate the similarity to each reference ligand.
    :param ref: reference dataframe with all the reference plifs
    :param query: query dataframe with all the docked plifs
    :param max_sim: if True, return the maximum similarity and the corresponding reference ligand
                    else return all the similarities to each reference ligand
    """
    # read the reference and docked plifs
    df_ref = ref
    df_docked = query

    df_ref["Reference"] = True
    df_docked["Reference"] = False

    df_ref_poses = (
        pd.concat([df_ref, df_docked])
        .fillna(False)
        .sort_index(
            axis=1,
            level=1,
            key=lambda index: [plf.ResidueId.from_string(x) for x in index],
        )
    )

    if ignored_interactions:
        to_drop = df_ref_poses.columns.get_level_values("interaction").isin(ignored_interactions)
        df_ref_poses = df_ref_poses.drop(columns=df_ref_poses.columns[to_drop])

    ### SIMILARITY TO ANY CRYSTAL LIGAND
    ref_df = df_ref_poses[df_ref_poses["Reference"] == True].drop(columns="Reference", level="protein").copy()
    query_df = df_ref_poses[df_ref_poses["Reference"] == False].drop(columns="Reference", level="protein").copy()

    ref_bitvectors = plf.to_bitvectors(ref_df)
    query_bitvectors = plf.to_bitvectors(query_df)
    # for each query, calculate the maximum similarity to any of the reference ligands
    tanimoto_sims = list_list_tanimoto_similarity(query_bitvectors, ref_bitvectors, n_jobs=n_jobs)
    assert len(tanimoto_sims) == len(query_bitvectors), "similarity calculation is not correct."
    assert len(tanimoto_sims[0]) == len(ref_bitvectors), "similarity calculation is not correct."
    plif_sim = pd.DataFrame(tanimoto_sims, index=query_df.index, columns=ref_df.index)

    if max_sim:
        plif_sim["ref_nearest_ligand"] = plif_sim.idxmax(axis=1)
        plif_sim["max_plif_sim"] = plif_sim.drop(columns="ref_nearest_ligand").max(axis=1)
        return plif_sim.loc[:, ["max_plif_sim", "ref_nearest_ligand"]], ref_df, query_df

    return plif_sim, ref_df, query_df


def get_plif_sim(target: str, docking: str, paths: dict, ignored_interactions: list=None):
    """Get the PLIF similarity between docked and reference molecules.
    :param target: target name
    :param docking: docking method
    :param paths: dictionary with the related paths"""

    plif_dir = paths["plif_dir"]

    working_dir = os.path.join(plif_dir, target.upper())
    plif = pd.read_csv(os.path.join(working_dir, f"docked_plifs_{docking}.csv"), header=[0, 1], index_col=0)
    # get the unique id for the VINA plif data (for merging with pb_valid data)
    if docking == "vina":
        plif.index = get_unique_pose_id(list(plif.index))
        seq_align_dict_path = os.path.join(working_dir, "seq_align_dict_all.pkl")
        vina_seq_align_dict = pd.read_pickle(seq_align_dict_path)["vina"]
        plif.columns = pd.MultiIndex.from_tuples([(vina_seq_align_dict[col[0]], col[1]) for col in plif.columns])
        plif.columns.names = ["protein", "interaction"]

    # read the reference plif data
    ref_plif = pd.read_csv(os.path.join(working_dir, "ref_plifs.csv"), header=[0, 1], index_col=0)
    ref_plif.columns.names = ["protein", "interaction"]

    plif_sim, ref_df, query_df = plif_sim_pose(ref_plif, plif, ignored_interactions=ignored_interactions, max_sim=False, n_jobs=80)

    # formatting
    plif_sim_stack = plif_sim.stack().reset_index()
    plif_sim_stack.columns = ["pose_id", "ref_lig", f"plif_sim_{docking}"]
    plif_sim_stack["target"] = target.upper()

    return plif_sim_stack, ref_df, query_df


def get_pb_valid(target: str, docking: str, paths: dict):
    """Get the posebuster valid data.
    :param target: target name
    :param docking: docking method
    :param paths: dictionary with the related paths"""
    pb_dir = paths["pb_dir"]
    
    if docking == "vina":
        pb = pd.read_csv(os.path.join(pb_dir, f"{target.upper()}_vina_vina.csv"), header=0, index_col=0)
        pb.index = get_unique_pose_id(pb["molecule"].tolist())
    elif docking.lower() in ["ddl", "diffdock_l"]:
        pb = pd.read_csv(os.path.join(pb_dir, f"{target.upper()}_diffdock_L_vina.csv"), header=0, index_col=0)
        pb.index = pb.molecule.values.tolist()

    # pb_valid = True if all values except "molecule" are True in the row, False otherwise
    pb["pb_valid"] = pb.iloc[:, 1:].all(axis=1)
    return pb


def plot_pb_valid_plif_sim(sim: pd.DataFrame, target: str, outfile: str=None):
    color_code = {"Vina": "tab:blue", "DiffDock-L": "tab:orange"}
    plot_order = ["Vina", "DiffDock-L"]
    fig, ax = plt.subplots()
    stat_target = list()
    plif_100 = pd.DataFrame()   
    plif_ratios = [0.5, 0.85, 1.0]
    plif_ratio_labels = list()
    for r in plif_ratios:
        if r == 1:
            plif_ratio_labels.append(f"PB_valid & PLIF_sim = {r}")
        else:
            plif_ratio_labels.append(f"PB_valid & PLIF_sim >= {r}")
    ratio_hatches = ["/", "\\", "x", "o"]
    for i, docking in enumerate(plot_order):
        stat_docking = list()
        plif_sim_actives = sim[sim["docking"] == docking]
        col_unique = ["mol_id", "target"] if "target" in plif_sim_actives.columns else ["mol_id"]
        total_mol = plif_sim_actives[col_unique].drop_duplicates().shape[0]
        plif_pb_valid = plif_sim_actives.dropna(subset=["pb_valid"])
        plif_pb_valid = plif_pb_valid[plif_pb_valid["pb_valid"] == True]    
        plif_pb_valid_nmol = plif_pb_valid[col_unique].drop_duplicates().shape[0]
        plif_pb_valid_ratio = plif_pb_valid_nmol/total_mol
        stat_docking.extend([target, docking, total_mol, plif_pb_valid_nmol, plif_pb_valid_ratio])
        ax.bar(x=i, height=plif_pb_valid_ratio, label="PB_valid", fill=False, edgecolor=color_code[docking])
        for j, r in enumerate(plif_ratios):
            df_r = plif_pb_valid[plif_pb_valid["plif_sim"] >= float(r)]
            df_r_nmol = df_r[col_unique].drop_duplicates().shape[0]
            df_r_ratio = df_r_nmol / total_mol
            stat_docking.extend([df_r_nmol, df_r_ratio])
            if j == len(plif_ratios)-1:
                ax.bar(x=i, height=df_r_ratio, label=plif_ratio_labels[j], color=color_code[docking])
            else:
                ax.bar(x=i, height=df_r_ratio, label=plif_ratio_labels[j], hatch=ratio_hatches[j], fill=False, edgecolor=color_code[docking])
            if r == 1:
                plif_100 = pd.concat([plif_100, df_r])
        stat_target.append(stat_docking)
    
    # customize colors for legend
    handles = list()
    labels = ["PB_valid", 'PB_valid & PLIF_sim >= 0.5', 'PB_valid & PLIF_sim >= 0.85', 'PB_valid & PLIF_sim = 1']
    handles.append(Patch(facecolor='white', edgecolor='black'))
    handles.append(Patch(facecolor='white', edgecolor='black', hatch=f"///"))
    handles.append(Patch(facecolor='white', edgecolor='black', hatch="xxx"))
    handles.append(Patch(color='black'))

    # set axis labels
    ax.set_ylabel("Ratio", fontsize=15, labelpad=10)
    ax.set_xlabel("Pose sampling method", fontsize=15, labelpad=10)
    plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))
    plt.xticks(ticks=[0, 1], labels=plot_order, fontsize=15)
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    return stat_target, plif_100


def count_interactions(df: pd.DataFrame, ignored_interactions: list=None):
    """Count the number of interactions (True cells) for each row in the dataframe.
    :param df: dataframe with interaction data
    :param ignored_interactions: list of interactions to ignore. Default: ["VdWContact", "FaceToFace", "EdgeToFace"]
    :return: a list of the number of interactions for each row"""
    if ignored_interactions:
        to_drop = df.columns.get_level_values(1).isin(ignored_interactions)
        df = df.drop(columns=df.columns[to_drop])
    n_interactions = df.sum(axis=1).tolist()
    return n_interactions