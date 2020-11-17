from project import *
import time

genome_file = open('genome.fa', 'r')
genome = genome_file.readlines()[1][:-1]
genome_file.close()

reads_file = open('reads.fa', 'r')
reads = reads_file.readlines()[1::2][:-1]
reads_file.close()

# genes_file = open('genes.tab', 'r')

start = time.time()

get_suffix_array(genome + '$')

end = time.time()

print(end - start)

# aligner = Aligner()