import numpy as np
import pandas as pd
import os
import pdb

from tqdm import tqdm

from fastsk import FastSK
from fastsk.utils import FastaUtility

import editdistance
from fastdist import fastdist

from promoter_modelling.utils.fasta_utils import create_fasta_from_sequences
from promoter_modelling.utils.motif_detection_utils import detect_vierstra_motifs_in_sequences

np.random.seed(97)

def compute_average_pairwise_gapped_kmer_occurrences_similarity(sequences, kmer_length=20, gap_length=14, num_cores=-1):
    """
    Computes the average pairwise similarity between sequences using gapped k-mers.
    """
    # create a fasta file from the sequences
    fasta_file_path = "temp.fasta"
    fasta_file = create_fasta_from_sequences(sequences, fasta_file_path)

    # read the fasta file
    reader = FastaUtility()
    seqs, _ = reader.read_data(fasta_file_path)

    # compute fast SK kernel
    fastsk = FastSK(g=kmer_length, m=gap_length, t=num_cores, approx=False)
    fastsk.compute_kernel(seqs, seqs[:5]) # use a dummy set of sequences for the test set (second argument)
    
    # compute the pairwise similarity matrix
    similarity_matrix = np.array(fastsk.get_train_kernel())
    
    # compute the average similarity
    average_similarity = np.mean(similarity_matrix[np.triu_indices(len(sequences), k=1)])
    
    # delete the fasta file
    os.remove(fasta_file_path)
    
    return average_similarity, similarity_matrix

def compute_average_pairwise_vierstra_motif_occurrences_similarity(sequences, vierstra_motifs_data_dir, \
                                                                   fimo_outputs_save_dir="./temp_fimo_outputs", description="temp", \
                                                                   q_val_thres=0.05, num_cores=-1, both_strands=True, fimo_cmd_path="fimo"):
    """
    Computes the average pairwise similarity between sequences using the Vierstra motifs.
    Also returns the average number of motif occurrences detected in each sequence and the average number of unique motifs detected in each sequence.
    """
    # detect the Vierstra motifs in the sequences
    motif_occurrences_df = detect_vierstra_motifs_in_sequences(sequences, description, fimo_outputs_save_dir, vierstra_motifs_data_dir, \
                                                               q_val_thres=q_val_thres, num_cores=num_cores, \
                                                               both_strands=both_strands, fimo_cmd_path=fimo_cmd_path)

    # delete the fimo outputs if only needed temporarily
    if fimo_outputs_save_dir == "./temp_fimo_outputs":
        os.system("rm -r " + fimo_outputs_save_dir)
        
    # compute the pairwise similarity matrix
    motif_occurrences_columns = motif_occurrences_df.columns[1:]
    motif_occurrences_matrix = np.array(motif_occurrences_df[motif_occurrences_columns].values)

    similarity_matrix = np.nan_to_num(fastdist.matrix_pairwise_distance(motif_occurrences_matrix, fastdist.cosine, "cosine", return_matrix=True), nan=0, posinf=0, neginf=0)
    
    # compute the average similarity
    average_similarity = np.mean(similarity_matrix[np.triu_indices(len(sequences), k=1)])

    # compute the average number of total motifs detected in each sequence
    average_num_motifs = np.mean(np.sum(motif_occurrences_matrix, axis=1))

    # compute the average number of unique motifs detected in each sequence
    average_num_unique_motifs = np.mean(np.sum(motif_occurrences_matrix > 0, axis=1))
    
    return average_similarity, average_num_motifs, average_num_unique_motifs, similarity_matrix

def compute_average_pairwise_edit_distance(sequences):
    """
    Computes the average pairwise edit distance between sequences.
    """
    # compute the pairwise edit distance matrix
    edit_distance_matrix = np.zeros((len(sequences), len(sequences)))
    for i in range(len(sequences)):
        for j in range(i, len(sequences)):
            edit_distance_matrix[i, j] = editdistance.eval(sequences[i], sequences[j])
            edit_distance_matrix[j, i] = edit_distance_matrix[i, j]
    
    # compute the average edit distance
    average_edit_distance = np.mean(edit_distance_matrix[np.triu_indices(len(sequences), k=1)])
    
    return average_edit_distance, edit_distance_matrix

def compute_average_nucleotide_entropy(sequences):
    """
    Computes the average nucleotide entropy of sequences.
    """
    # compute the nucleotide entropy of each sequence
    nucleotide_entropies = []
    for sequence in sequences:
        nucleotide_counts = np.zeros(4)
        for nucleotide in sequence:
            if nucleotide == "A":
                nucleotide_counts[0] += 1
            elif nucleotide == "C":
                nucleotide_counts[1] += 1
            elif nucleotide == "G":
                nucleotide_counts[2] += 1
            elif nucleotide == "T":
                nucleotide_counts[3] += 1
        nucleotide_entropies.append(-np.sum((nucleotide_counts / len(sequence)) * np.log2(nucleotide_counts / len(sequence))))
    
    # compute the average nucleotide entropy
    average_nucleotide_entropy = np.mean(nucleotide_entropies)
    
    return average_nucleotide_entropy, nucleotide_entropies