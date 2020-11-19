from project import *
from evaluation import *

import time
import re
import pickle

# Parsing

genome_file = open('genome.fa', 'r')
genome = genome_file.readlines()[1][:-1]
genome_file.close()

reads_file = open('reads.fa', 'r')
reads = [read[:-1] for read in reads_file.readlines()[1::2]]
reads_file.close()

genes_file = open('genes.tab', 'r')

gene_data, isoform_data, exon_data = [], [], []

for line in genes_file.readlines():

    data = re.split(';| |\t|\n', line)[:-1]
    
    if data[0] == 'gene':
        gene_data.append(data)
    elif data[0] == 'isoform':
        isoform_data.append(data)
    elif data[0] == 'exon':
        exon_data.append(data)

known_genes, known_isoforms, known_exons = set(), dict(), dict()

for data in exon_data:
    isoform_id, start, end = data[1], int(data[2]), int(data[3])
    exon = Exon(isoform_id, start, end)
    known_exons[exon.id] = exon

for data in isoform_data:
    isoform_id, exons = data[1], [known_exons[exon_id] for exon_id in data[2:]]
    isoform = Isoform(isoform_id, exons)
    known_isoforms[isoform.id] = isoform

for data in gene_data:
    gene_id, isoforms = data[1], [known_isoforms[isoform_id] for isoform_id in data[2:]]
    gene = Gene(gene_id, isoforms)
    known_genes.add(gene)

genes_file.close()

# Tests

GENOME_SA_FILENAME = 'genome_sa.txt'

def diff_k(s_1, s_2, i_1, i_2, n, k):
    """
    Returns the pairwise difference of the first n characters of s_1 and s_2 starting
    at indices i_1 and i_2, respectively, up to a difference of k, and otherwise returns
    a number higher than k.
    """

    diff = 0

    for i in range(n):
        if s_1[i_1+i] != s_2[i_2+i]:
            diff += 1
        if diff > k:
            break
    
    return diff

def test_package(write=True):

    start_init = time.time()

    result = get_suffix_array_package(genome + '$')

    end_init = time.time()

    print('init:', end_init - start_init)

    if write:
        np.savetxt(GENOME_SA_FILENAME, result, fmt='%d')

def test_init(write=True):

    start_init = time.time()

    aligner = Aligner(genome, known_genes)

    end_init = time.time()

    print('init:', end_init - start_init)

    if write:
        np.savetxt('genome_sa_init.txt', aligner.whole_genome_FM['sa'], fmt='%d')

def test_align_known_read(read=True):

    if read:
        sa = np.loadtxt(GENOME_SA_FILENAME, dtype=int)
        aligner = Aligner(genome, known_genes, genome_sa=sa)
    else:
        start_init = time.time()
        aligner = Aligner(genome, known_genes)
        end_init = time.time()
        print('init', end_init - start_init)

    start_align = time.time()

    start_index = 8250613 # Exon ENSE00001802701 (line 3 of genes.tab)
    read = genome[start_index:start_index+50]
    print(aligner.align(read))

    end_align = time.time()

    print('align:', end_align - start_align)

def test_align_unknown_read(read=True):

    if read:
        sa = np.loadtxt(GENOME_SA_FILENAME, dtype=int)
        aligner = Aligner(genome, known_genes, genome_sa=sa)
    else:
        start_init = time.time()
        aligner = Aligner(genome, known_genes)
        end_init = time.time()
        print('init', end_init - start_init)

    start_align = time.time()

    start_index = 6455453 # unknown exon (line 740 of genes.tab)
    read = genome[start_index:start_index+50]
    print(read)
    print(aligner.align(read))

    end_align = time.time()

    print('align:', end_align - start_align)

def test_align_split_read(read=True):

    if read:
        sa = np.loadtxt(GENOME_SA_FILENAME, dtype=int)
        aligner = Aligner(genome, known_genes, genome_sa=sa)
    else:
        start_init = time.time()
        aligner = Aligner(genome, known_genes)
        end_init = time.time()
        print('init', end_init - start_init)

    start_align = time.time()

    start_index = 6455453 # unknown exon (line 740 of genes.tab)
    offset = 2345
    read = genome[start_index:start_index+25] + genome[start_index+offset:start_index+offset+25]
    print(read)
    print(aligner.align(read))

    end_align = time.time()

    print('align:', end_align - start_align)

def test_align_all_reads(read=True):

    if read:
        sa = np.loadtxt(GENOME_SA_FILENAME, dtype=int)
        aligner = Aligner(genome, known_genes, genome_sa=sa)
    else:
        start_init = time.time()
        aligner = Aligner(genome, known_genes)
        end_init = time.time()
        print('init', end_init - start_init)

    max_align_time, avg_align_time = -float('inf'), 0
    priority_1_matches, priority_2_matches = 0, 0

    for read in reads:

        start_align = time.time()

        result = aligner.align(read)
        print(result)
        
        if result == None:
            priority_1_matches += 1
        elif len(result) > 0:
            priority_2_matches += 1

        end_align = time.time()

        difference = 0
        for res in result:
            difference += diff_k(read, genome, res[0], res[1], res[2], 6)
        print(difference)

        align_time = end_align - start_align

        # print(align_time)

        max_align_time = max(max_align_time, align_time)
        avg_align_time += align_time

    avg_align_time /= len(reads)

    print('align (max):', max_align_time)
    print('align (avg):', avg_align_time)
    print('priority 1 matches: {0}/{1}'.format(priority_1_matches, len(reads)))
    print('priority 2 matches: {0}/{1}'.format(priority_2_matches, len(reads)))

# test_init(write=True)
# test_align_known_read(read=False)
# test_align_unknown_read(read=True)
test_align_all_reads(read=True)
# test_package()
# test_align_split_read()

# start_index = 6455453 # unknown exon (line 740 of genes.tab)
# print(genome[start_index:start_index+50])
# start_index = 4505547
# print(genome[start_index:start_index+50])

# test_string_1 = 'ACGTTAGCCAGT'*50+'$'
# print(get_suffix_array(test_string_1))
# print(get_suffix_array_package(test_string_1))