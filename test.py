from project import *
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
# Pickles used to store suffix arrays only

GENOME_SA_FILENAME = 'genome_sa.txt'

def test_init():

    start_init = time.time()

    aligner = Aligner(genome, known_genes)

    end_init = time.time()

    print('init:', end_init - start_init)

    np.savetxt(GENOME_SA_FILENAME, aligner.whole_genome_FM['sa'], fmt='%d')

def test_align_known_read():

    sa = np.loadtxt(GENOME_SA_FILENAME, dtype=int)
    aligner = Aligner(genome, known_genes, genome_sa=sa)

    print('data loaded')

    start_align = time.time()

    start_index = 8250613 # Exon ENSE00001802701 (line 3 of genes.tab)
    read = genome[start_index:start_index+50]
    print(aligner.align(read))

    end_align = time.time()

    print('align:', end_align - start_align)




def test_align_all_reads():

    sa = np.loadtxt(GENOME_SA_FILENAME, dtype=int)
    aligner = Aligner(genome, known_genes, genome_sa=sa)

    print('data loaded')

    max_align_time, avg_align_time = -float('inf'), 0

    for read in reads:

        start_align = time.time()

        print(read, aligner.align(read))

        end_align = time.time()

        align_time = end_align - start_align

        print(align_time)

        max_align_time = max(max_align_time, align_time)
        avg_align_time += align_time

    avg_align_time /= len(reads)

    print('align (max):', max_align_time)
    print('align (avg):', avg_align_time)

# test_init()
test_align()
