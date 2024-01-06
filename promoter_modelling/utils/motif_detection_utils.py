import os
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm

from joblib import Parallel, delayed

np.random.seed(97)

# function to get the motif to index mapping
def get_motif_to_ind_mapping(motifs_path):
    motifs = open(motifs_path, "r").readlines()
    
    motif_to_ind = {}
    ind_to_motif = {}
    ind = 0
    for line in motifs:
        if line.startswith("MOTIF"):
            line = line.strip().split(" ")
            motif_to_ind[line[1]] = ind
            ind_to_motif[ind] = line[1]
            ind += 1
    assert len(motif_to_ind) == ind
    assert len(ind_to_motif) == ind
    return motif_to_ind, ind_to_motif

# function to split the motifs into individual files
def split_motifs_into_individual_files(motifs_path, output_dir, motif_to_ind):
    motifs = open(motifs_path, "r").readlines()
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for i, line in enumerate(motifs):
        if line.startswith("MOTIF"):
            line = line.strip().split(" ")
            motif_name = "motif_" + str(motif_to_ind[line[1]])
            motif_path = os.path.join(output_dir, motif_name + ".meme")
            
            motif_file = open(motif_path, "w+")

            # write header
            motif_file.write("MEME version 4")
            motif_file.write("\n\n")
            motif_file.write("ALPHABET= ACGT")
            motif_file.write("\n\n")
            motif_file.write("strands: + -")
            motif_file.write("\n\n")
            motif_file.write("Background letter frequencies (from uniform background):")
            motif_file.write("\n")
            motif_file.write("A 0.25000 C 0.25000 G 0.25000 T 0.25000")
            motif_file.write("\n\n")
            motif_file.write("MOTIF {} {}".format(motif_name, line[1]))
            motif_file.write("\n")

            # write motif
            for j in range(i+1, len(motifs)):
                if motifs[j].startswith("MOTIF"):
                    i = j - 1
                    break
                motif_file.write(motifs[j])
            motif_file.close()                    

# function to download the Vierstra motifs
def download_vierstra_motifs(vierstra_motifs_data_dir, link="https://resources.altius.org/~jvierstra/projects/motif-clustering-v2.0beta/consensus_pwms.meme"):
    if not os.path.exists(vierstra_motifs_data_dir):
        os.mkdir(vierstra_motifs_data_dir)
    
    # download the Vierstra motifs
    vierstra_motifs_download_path = os.path.join(vierstra_motifs_data_dir, "pfm_meme.txt")
    vierstra_motifs_download_link = link
    if not os.path.exists(vierstra_motifs_download_path):
        os.system("wget -O {} {}".format(vierstra_motifs_download_path, vierstra_motifs_download_link))
        print("wget -O {} {}".format(vierstra_motifs_download_path, vierstra_motifs_download_link))

    # get the motif to index and index to motif mapping
    motif_to_ind_save_path = os.path.join(vierstra_motifs_data_dir, "motif_to_ind.npy")
    ind_to_motif_save_path = os.path.join(vierstra_motifs_data_dir, "ind_to_motif.npy")
    if not os.path.exists(motif_to_ind_save_path):
        motif_to_ind, ind_to_motif = get_motif_to_ind_mapping(vierstra_motifs_download_path)
        np.save(motif_to_ind_save_path, motif_to_ind)
        np.save(ind_to_motif_save_path, ind_to_motif)
    
    # split the motifs into individual files
    split_motifs_into_individual_files(vierstra_motifs_download_path, os.path.join(vierstra_motifs_data_dir, "individual_motifs"), motif_to_ind)

# function to generate commands for motif detection
def generate_cmds_for_motif_detection(sequences, \
                                      description, \
                                      motifs_path, \
                                      root_save_dir, \
                                      both_strands=True, \
                                      fimo_cmd_path="fimo"):
    if not os.path.exists(root_save_dir):
        os.mkdir(root_save_dir)
    output_dir = os.path.join(root_save_dir, description)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if os.path.exists(os.path.join(output_dir, "fimo.tsv")):
        return []
    
    temp_file_name = os.path.join(root_save_dir, description, "seqs.fasta")

    # write sequences to file
    if not os.path.exists(temp_file_name):
        with open(temp_file_name, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(">{}\n".format(i))
                f.write("{}\n".format(seq))
    
    cmds = []
    if both_strands:
        cmds.append("{} --max-stored-scores 1000000 --verbosity 1 --oc {} {} {}".format(fimo_cmd_path, output_dir, motifs_path, temp_file_name))
    else:
        cmds.append("{} --max-stored-scores 1000000 --verbosity 1 --oc {} --norc {} {}".format(fimo_cmd_path, output_dir, motifs_path, temp_file_name))
        
    return cmds

# function to parse the fimo output
def parse_fimo_output(path, sequences, ind_to_motif, q_val_thres):
    occurrence_freqs = {}
    occurrence_freqs["sequence"] = sequences
    try:
        motifs = pd.read_csv(path, sep="\t", comment="#")
        for i in range(motifs.shape[0]):
            row = motifs.iloc[i]
            if row["q-value"] > q_val_thres:
                continue
                
            motif_ind = int(row["motif_id"].split("_")[1])
            if ind_to_motif[motif_ind] not in occurrence_freqs:
                occurrence_freqs[ind_to_motif[motif_ind]] = [0]*len(sequences)
            seq_ind = int(row["sequence_name"])
            occurrence_freqs[ind_to_motif[motif_ind]][seq_ind] += 1
    except:
        pass
    
    occurrence_freqs = pd.DataFrame(occurrence_freqs)
    return occurrence_freqs

# function to detect the Vierstra motifs in the sequences
def detect_vierstra_motifs_in_sequences(input_sequences, description, root_save_dir, vierstra_motifs_data_dir, \
                                        q_val_thres=0.05, num_cores=-1, both_strands=True, fimo_cmd_path="fimo"):
    motifs_file = os.path.join(vierstra_motifs_data_dir, "pfm_meme.txt")
    if not os.path.exists(motifs_file):
        download_vierstra_motifs(vierstra_motifs_data_dir)
        assert os.path.exists(motifs_file)
    
    # get the motif to index and index to motif mapping
    motif_to_ind_save_path = os.path.join(vierstra_motifs_data_dir, "motif_to_ind.npy")
    ind_to_motif_save_path = os.path.join(vierstra_motifs_data_dir, "ind_to_motif.npy")
    motif_to_ind = np.load(motif_to_ind_save_path, allow_pickle=True).item()
    ind_to_motif = np.load(ind_to_motif_save_path, allow_pickle=True).item()

    # all individual motifs
    all_motifs = sorted(os.listdir(os.path.join(vierstra_motifs_data_dir, "individual_motifs")))

    # generate commands for motif detection
    cmds = []
    for motif in all_motifs:
        motif_path = os.path.join(vierstra_motifs_data_dir, "individual_motifs", motif)
        cmds += generate_cmds_for_motif_detection(input_sequences, description + "_" + motif.split(".")[0], motif_path, root_save_dir, \
                                                  both_strands=both_strands, fimo_cmd_path=fimo_cmd_path)
        
    # run the commands in parallel
    if len(cmds) > 0:
        Parallel(n_jobs=num_cores)(delayed(os.system)(cmd) for cmd in tqdm(cmds))

    # parse the fimo output
    motif_occurrences_df = {}
    motif_occurrences_df["sequence"] = input_sequences
    for i in range(len(ind_to_motif)):
        motif_occurrences_df[ind_to_motif[i]] = [0]*len(input_sequences)
    
    for motif in all_motifs:
        fimo_output_path = os.path.join(root_save_dir, description + "_" + motif.split(".")[0], "fimo.tsv")
        parsed_outs = parse_fimo_output(fimo_output_path, input_sequences, ind_to_motif, q_val_thres)
        
        for col in parsed_outs.columns:
            if col == "sequence":
                continue
            motif_occurrences_df[col] += parsed_outs[col]    
    motif_occurrences_df = pd.DataFrame(motif_occurrences_df)
    
    return motif_occurrences_df
