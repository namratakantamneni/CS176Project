from project import *
import time
import re

genome_file = open('genome.fa', 'r')
genome = genome_file.readlines()[1][:-1]
genome_file.close()

reads_file = open('reads.fa', 'r')
reads = reads_file.readlines()[1::2][:-1]
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

start = time.time()

aligner = Aligner(genome, known_genes)

end = time.time()

print(end - start)
