import os
import re
import hashlib
import random
import sys
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.PDB import PDBParser, NeighborSearch, PDBList
from Bio.PDB.Polypeptide import is_aa
from pathlib import Path
from ColabFold.colabfold.download import download_alphafold_params, default_data_dir
from ColabFold.colabfold.utils import setup_logging
from ColabFold.colabfold.batch import get_queries, run, set_model_type
import numpy as np
from argparse import ArgumentParser
import shutil
import pandas as pd
from cal_scores import calculate_scores
import pdb

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)


def add_hash(x, y):
    return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]


def main(query_sequence, protein_id, type):
    """ Input protein sequence(s) """

    num_relax = 0
    use_amber = num_relax > 0

    template_mode = "pdb100"

    # # remove whitespaces
    # query_sequence = "".join(query_sequence.split())
    # basejobname = "".join(jobname.split())
    # basejobname = re.sub(r'\W+', '', basejobname)
    if ':' in query_sequence:
        jobname = query_sequence.split(':')[1].strip()
    else:
        jobname = 'protein'
    if type is not None:
        jobname = type + '_' + jobname
    if protein_id is not None:
        jobname = protein_id + '_' + jobname

    # # check if directory with jobname exists
    # def check(folder):
    #     if os.path.exists(folder):
    #         return False
    #     else:
    #         return True

    # if not check(jobname):
    #     n = 0
    #     while not check(f"{jobname}_{n}"): n += 1
    #     jobname = f"{jobname}_{n}"

    os.makedirs(jobname, exist_ok=True)

    # save queries
    queries_path = os.path.join(jobname, f"{jobname}.csv")
    with open(queries_path, "w") as text_file:
        text_file.write(f"id,sequence\n{jobname},{query_sequence}")

    if template_mode == "pdb100":
        use_templates = True
        custom_template_path = None
    elif template_mode == "custom":
        custom_template_path = os.path.join(jobname, f"template")
        os.makedirs(custom_template_path, exist_ok=True)
        uploaded = files.upload()
        use_templates = True
        for fn in uploaded.keys():
            os.rename(fn, os.path.join(custom_template_path, fn))
    else:
        custom_template_path = None
        use_templates = False

    print("jobname", jobname)
    print("sequence", query_sequence)
    print("length", len(query_sequence.replace(":", "")))

    """ MSA options """
    msa_mode = "mmseqs2_uniref_env"
    pair_mode = "unpaired_paired"

    if "mmseqs2" in msa_mode:
        a3m_file = os.path.join(jobname, f"{jobname}.a3m")

    elif msa_mode == "custom":
        a3m_file = os.path.join(jobname, f"{jobname}.custom.a3m")
        if not os.path.isfile(a3m_file):
            custom_msa_dict = files.upload()
            custom_msa = list(custom_msa_dict.keys())[0]
            header = 0
            import fileinput
            for line in fileinput.FileInput(custom_msa, inplace=1):
                if line.startswith(">"):
                    header = header + 1
                if not line.rstrip():
                    continue
                if line.startswith(">") == False and header == 1:
                    query_sequence = line.rstrip()
                print(line, end='')

            os.rename(custom_msa, a3m_file)
            queries_path = a3m_file
            print(f"moving {custom_msa} to {a3m_file}")

    else:
        a3m_file = os.path.join(jobname, f"{jobname}.single_sequence.a3m")
        with open(a3m_file, "w") as text_file:
            text_file.write(">1\n%s" % query_sequence)

    """ Advanced settings"""
    # ["auto", "alphafold2_ptm", "alphafold2_multimer_v1", "alphafold2_multimer_v2",
    # "alphafold2_multimer_v3", "deepfold_v1"]
    model_type = "alphafold2_multimer_v3" if ':' in query_sequence else "auto"
    print(f"Using model {model_type}")
    num_recycles = "6"
    recycle_early_stop_tolerance = "auto"
    relax_max_iterations = 200
    pairing_strategy = "greedy"
    max_msa = "auto"  # @param ["auto", "512:1024", "256:512", "64:128", "32:64", "16:32"]
    num_seeds = 1  # @param [1,2,4,8,16] {type:"raw"}
    use_dropout = False  # @param {type:"boolean"}

    num_recycles = None if num_recycles == "auto" else int(num_recycles)
    recycle_early_stop_tolerance = None if recycle_early_stop_tolerance == "auto" else float(
        recycle_early_stop_tolerance)
    if max_msa == "auto": max_msa = None

    """ Run Prediction """
    result_dir = jobname
    log_filename = os.path.join(jobname, "log.txt")
    setup_logging(Path(log_filename))

    queries, is_complex = get_queries(queries_path)
    model_type = set_model_type(is_complex, model_type)

    if "multimer" in model_type and max_msa is not None:
        use_cluster_profile = False
    else:
        use_cluster_profile = True

    download_alphafold_params(model_type, Path("."))

    results = run(
        queries=queries,
        result_dir=result_dir,
        use_templates=use_templates,
        custom_template_path=custom_template_path,
        num_relax=num_relax,
        msa_mode=msa_mode,
        model_type=model_type,
        num_models=5,
        num_recycles=num_recycles,
        relax_max_iterations=relax_max_iterations,
        recycle_early_stop_tolerance=recycle_early_stop_tolerance,
        num_seeds=num_seeds,
        use_dropout=use_dropout,
        model_order=[1, 2, 3, 4, 5],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        rank_by="auto",
        pair_mode=pair_mode,
        pairing_strategy=pairing_strategy,
        stop_at_score=float(100),
        # prediction_callback=prediction_callback,
        # dpi=dpi,
        zip_results=False,
        save_all=False,
        max_msa=max_msa,
        use_cluster_profile=use_cluster_profile,
        # input_features_callback=input_features_callback,
        save_recycles=False,
        user_agent="colabfold/google-colab-main",
    )
    
    if protein_id is not None:
        os.makedirs(f'/home/tc415/moPPIt/moppit/pdb_results/{protein_id}', exist_ok=True)
        shutil.move(f'/home/tc415/moPPIt/moppit/{jobname}', f'/home/tc415/moPPIt/moppit/pdb_results/{protein_id}/{jobname}')
    else:
        shutil.move(f'/home/tc415/moPPIt/moppit/{jobname}', f'/home/tc415/moPPIt/moppit/pdb_results/{jobname}')

    calculate_scores(f'/home/tc415/moPPIt/moppit/pdb_results/{protein_id}/{jobname}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.f)
    for i in range(len(df)):
        seq = df.iloc[i]['Sequence']
        protein_id = df.iloc[i]['ID']

        main(seq, protein_id, None)