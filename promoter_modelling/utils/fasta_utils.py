import numpy as np
import pandas as pd
import os

import kipoiseq
from kipoiseq import Interval
import pyfaidx

np.random.seed(97)

class FastaStringExtractor:    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}
#         print(self._chromosome_sizes)

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = kipoiseq.Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    strand=interval.strand
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop,
                                         rc=(trimmed_interval.strand == "-")).seq).upper()
        
        return sequence
    
    def close(self):
        return self.fasta.close()

def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

def get_sequence(chromosome, start, end, strand, fasta_extractor):
    interval = kipoiseq.Interval(chromosome, start-1, end-1, strand=strand)
    sequence = fasta_extractor.extract(interval)
    assert len(sequence) == (end - start)
    
    return sequence

def get_promoter_sequences(gene_id, all_windows, ensembl_info, fasta_extractor):
    gene_info = ensembl_info[ensembl_info["Gene stable ID"] == gene_id].iloc[0]
    tss = gene_info["Transcription start site (TSS)"]
    strand = gene_info["Strand"]
    chromosome = "chr{}".format(gene_info["Chromosome/scaffold name"])
    if strand == 1:
        strand = "."
    else:
        strand = "-"
    
    all_seqs = []
    
    for window in all_windows:   
        if strand == ".":
            begin = tss + window[0]
            end = tss + window[1]
            interval = kipoiseq.Interval(chromosome, begin-1, end-1, strand=strand)
        else:
            begin = tss - window[0]
            end = tss - window[1]
            interval = kipoiseq.Interval(chromosome, end-1, begin-1, strand=strand)
        
        sequence = fasta_extractor.extract(interval)
        try:
            assert len(sequence) == (window[1] - window[0])
        except:
            pdb.set_trace()
        
        all_seqs.append(sequence)
        
    return all_seqs

def generate_dinucleotide_shuffled_sequences(sequence, num_shuffles=1, fasta_shuffle_letters_path="fasta_shuffle_letters"):
    # first create FASTA file containing sequence
    with open("temp.fasta", "w") as f:
        f.write(">temp\n")
        f.write(sequence)
    
    # run fasta_shuffle_letters to generate negatives
    output_path = "temp_shuffled.fasta"
    cmd = "{} -kmer 2 -seed 97 -copies {} -line 1000000 {} {}".format(fasta_shuffle_letters_path, num_shuffles, "temp.fasta", output_path)
    os.system(cmd)

    # parse output file
    assert os.path.exists(output_path)
    all_seqs = []
    output = open(output_path, "r").readlines()
    for line in output:
        line = line.strip()
        if len(line) == 0 or line.startswith(">"):
            continue
        all_seqs.append(line)
    assert len(all_seqs) == num_shuffles

    # remove temp files
    os.remove("temp.fasta")
    os.remove(output_path)

    return all_seqs

def create_fasta_from_sequences(sequences, output_path):
    with open(output_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(">0\n")
            f.write(seq)
            f.write("\n")