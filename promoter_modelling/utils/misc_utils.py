import numpy as np
import pandas as pd
import os
import pdb

np.random.seed(97)

# class to hold motif information
class Motif:
    def __init__(self, motif_name, pwm):
        assert pwm.shape[1] == 4
        
        self.motif_name = motif_name
        self.pwm = pwm
        self.length = pwm.shape[0]
        
    def sample_sequence(self):
        bases = ["A", "C", "G", "T"]
        seq = ""
        for i in range(self.pwm.shape[0]):
            sampled_base = np.random.choice(bases, p=self.pwm[i]/(self.pwm[i].sum()))
            seq = seq + sampled_base
        return seq

def parse_meme_file(meme_file):
    # Parse the meme file containing multiple motifs to get all motif information
    # Input: meme_file - path to the meme file
    # Output: motifs - list of motifs (Motif class)

    motifs = []
    with open(meme_file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            
            if line.startswith("MOTIF"):
                motif_name = line.split()[1]
                motif_pwm = []

                # get the motif pwm
                for j in range(i+1, len(lines)):
                    line = lines[j]
                    if line.startswith("MOTIF"):
                        i = j-1
                        break
                    if line.startswith("letter-probability"):
                        continue
                    if len(line.strip()) > 0:
                        motif_pwm.append([float(x) for x in line.strip().split()])
                
                motifs.append(Motif(motif_name, np.array(motif_pwm)))

    return motifs

def insert_motif_into_sequence(sequence, motif, num_insertions):
    # Insert motif into sequence at random positions
    # Input: sequence - sequence to insert motif into
    #        motif - motif to insert
    #        num_insertions - number of times to insert motif
    # Output: new_sequence - sequence with motif inserted

    new_sequence = sequence
    for i in range(num_insertions):
        start = np.random.randint(0, len(sequence)-motif.length)
        new_sequence = new_sequence[:start] + motif.sample_sequence() + new_sequence[start+motif.length:]
    return new_sequence