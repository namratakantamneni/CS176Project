""" 
    RNA Alignment Assignment
    
    Implement each of the functions below using the algorithms covered in class.
    You can construct additional functions and data structures but you should not
    change the functions' APIs.

    You will be graded on the helper function implementations as well as the RNA alignment, although
    you do not have to use your helper function.
    
    *** Make sure to comment out any print statement so as not to interfere with the grading script
"""

import sys # DO NOT EDIT THIS
from shared import *
import numpy as np

ALPHABET = [TERMINATOR] + BASES

def get_suffix_array(s):
    """
    Naive implementation of suffix array generation (0-indexed). You do not have to implement the
    KS Algorithm. Make this code fast enough so you have enough time in Aligner.__init__ (see bottom).

    Input:
        s: a string of the alphabet ['A', 'C', 'G', 'T'] already terminated by a unique delimiter '$'
    
    Output: list of indices representing the suffix array

    >>> get_suffix_array('GATAGACA$')
    [8, 7, 5, 3, 1, 6, 4, 0, 2]
    """

    RADIX = 50

    # sa: the suffix array.
    sa = np.arange(len(s), dtype='uint32')

    # ranges: index ranges in 'sa' which are not fully sorted.
    # format: [start index (inclusive), end index (exclusive), # of valid ranges]
    ranges = [np.empty(len(s), dtype='uint32'), np.empty(len(s), dtype='uint32'), 1] 
    ranges[0][0], ranges[1][0] = 0, len(s)

    for d in range(0, len(s), RADIX):

        next_ranges = [np.empty(len(s), dtype='uint32'), np.empty(len(s), dtype='uint32'), 0] # next set of ranges

        for r in range(ranges[2]):

            # buckets: a dict storing the results for one layer of counting sort.
            # format: {C: (indices where C is the next char, # of valid indices) ... }
            buckets = dict()

            for i in range(ranges[0][r], ranges[1][r]):

                key = s[i+d:min(i+d+RADIX,len(s))]

                if key in buckets.keys():
                    buckets[key].append(i)
                else:
                    buckets[key] = [i]

            bucket_order = sorted(buckets.keys())
            
            j = ranges[0][r]
            
            for key in bucket_order:

                indices = buckets[key]
                sa[j:j+len(indices)] = indices

                if len(indices) > 1:
                    next_ranges[0][next_ranges[2]], next_ranges[1][next_ranges[2]] = j, j+len(indices)
                    next_ranges[2] += 1
                
                j += len(indices)
                    
        ranges = next_ranges
    
    return list(sa)

def get_bwt(s, sa):
    """
    Input:
        s: a string terminated by a unique delimiter '$'
        sa: the suffix array of s

    Output:
        L: BWT of s as a string
    """
    bwt_arr = []

    for elem in sa:  
        bwt_arr.append(s[elem-1])

    bwt_str = ''.join(bwt_arr)

    return bwt_str

def get_F(L):
    """
    Input: L = get_bwt(s)
    Output: F, first column in Pi_sorted
    """
    counts = {c: 0 for c in ALPHABET}

    for c in L:
      counts[c] += 1
    
    F = ''
    for c in ALPHABET:
      F += counts[c] * c
    
    return F

def get_M(F):
    """
    Returns the helper data structure M (using the notation from class). M is a dictionary that maps character
    strings to start indices. i.e. M[c] is the first occurrence of "c" in F.

    If a character "c" does not exist in F, you may set M[c] = -1
    """

    M = {c: -1 for c in ALPHABET}

    chars_left = 5

    for i in range(len(F)):
        
        c = F[i]

        if M[c] == -1:
            M[c] = i
            chars_left -= 1
        
        if not chars_left:
          break

    return M

def get_occ(L):
    """
    Returns the helper data structure OCC (using the notation from class). OCC should be a dictionary that maps 
    string character to a list of integers. If c is a string character and i is an integer, then OCC[c][i] gives
    the number of occurrences of character "c" in the bwt string up to and including index i
    """
    OCC = dict()

    for char in L:
        if not char in OCC.keys():
            OCC[char] = [0]
    OCC[L[0]][0] = 1 # assume len(L) > 0
    
    for i in range(1, len(L)):
        for key in OCC.keys():
            OCC[key].append(OCC[key][i-1] + (1 if L[i] == key else 0))
    
    return OCC

def exact_suffix_matches(p, M, occ):
    """
    Find the positions within the suffix array sa of the longest possible suffix of p 
    that is a substring of s (the original string).
    
    Note that such positions must be consecutive, so we want the range of positions.

    Input:
        p: the pattern string
        M, occ: buckets and repeats information used by sp, ep

    Output: a tuple (range, length)
        range: a tuple (start inclusive, end exclusive) of the indices in sa that contains
            the longest suffix of p as a prefix. range=None if no indices matches any suffix of p
        length: length of the longest suffix of p found in s. length=0 if no indices matches any suffix of p

        An example return value would be ((2, 5), 7). This means that p[len(p) - 7 : len(p)] is
        found in s and matches positions 2, 3, and 4 in the suffix array.

    >>> s = 'ACGT' * 10 + '$'
    >>> sa = get_suffix_array(s)
    >>> sa
    [40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1, 38, 34, 30, 26, 22, 18, 14, 10, 6, 2, 39, 35, 31, 27, 23, 19, 15, 11, 7, 3]
    >>> L = get_bwt(s, sa)
    >>> L
    'TTTTTTTTTT$AAAAAAAAAACCCCCCCCCCGGGGGGGGGG'
    >>> F = get_F(L)
    >>> F
    '$AAAAAAAAAACCCCCCCCCCGGGGGGGGGGTTTTTTTTTT'
    >>> M = get_M(F)
    >>> sorted(M.items())
    [('$', 0), ('A', 1), ('C', 11), ('G', 21), ('T', 31)]
    >>> occ = get_occ(L)
    >>> type(occ) == dict, type(occ['$']) == list, type(occ['$'][0]) == int
    (True, True, True)
    >>> occ['$']
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    >>> exact_suffix_matches('ACTGA', M, occ)
    ((1, 11), 1)
    >>> exact_suffix_matches('$', M, occ)
    ((0, 1), 1)
    >>> exact_suffix_matches('AA', M, occ)
    ((1, 11), 1)
    """
    if not p[-1] in M.keys():
      return (None, 0)
    
    sp = M[p[-1]]
    ep = sp + occ[p[-1]][-1] - 1
    range, length = (sp, ep+1), 1

    while length < len(p):
      c = p[-length - 1]
      if not c in M.keys():
        break

      sp = M[c] + occ[c][max(0, sp-1)]
      ep = M[c] + occ[c][ep] - 1

      if sp > ep:
        break

      range = (sp, ep+1)
      length += 1

    return (range, length)

MIN_INTRON_SIZE = 20
MAX_INTRON_SIZE = 10000

class Aligner:

    def __init__(self, genome_sequence, known_genes):
        """
        Initializes the aligner. Do all time intensive set up here. i.e. build suffix array.

        genome_sequence: a string (NOT TERMINATED BY '$') representing the bases of the of the genome
        known_genes: a python set of Gene objects (see shared.py) that represent known genes. You can get the isoforms 
                     and exons from a Gene object

        Time limit: 500 seconds maximum on the provided data. Note that our server is probably faster than your machine, 
                    so don't stress if you are close. Server is 1.25 times faster than the i7 CPU on my computer
        """

        s = genome_sequence + '$'
        sa = get_suffix_array(s)
        L = get_bwt(s, sa)
        F = get_F(L)
        M = get_M(F)
        occ = get_occ(L)

        self.genome = {'s': s, 'sa': sa, 'L': L, 'F': F, 'M': M, 'occ': occ} # no introns/exons

        self.genes = dict() # dict: {gene_id -> dict: {isoform_id -> dict: {'s': string, 'sa': array, 'L': string, 'F': string, 'M': dict, 'occ': dict}}}

        for gene in known_genes:

            gene_data = dict()

            for isoform in gene.isoforms:

                isoform_data = dict()

                isoform_data['s'] = ''
                for exon in isoform.exons:
                    isoform_data['s'] += genome_sequence[exon.start:exon.end]
                isoform_data['s'] += '$'

                isoform_data['sa']  = get_suffix_array(isoform_data['s'])
                isoform_data['L']   = get_bwt(isoform_data['s'], isoform_data['sa'])
                isoform_data['F']   = get_F(isoform_data['L'])
                isoform_data['M']   = get_M(isoform_data['F'])
                isoform_data['occ'] = get_occ(isoform_data['L'])

                gene_data[isoform.id] = isoform_data

            self.genes[gene.id] = gene_data

    def align(self, read_sequence):
        """
        Returns an alignment to the genome sequence. An alignment is a list of pieces. 
        Each piece consists of a start index in the read, a start index in the genome, and a length 
        indicating how many bases are aligned in this piece. Note that mismatches are count as "aligned".

        Note that <read_start_2> >= <read_start_1> + <length_1>. If your algorithm produces an alignment that 
        violates this, we will remove pieces from your alignment arbitrarily until consecutive pieces 
        satisfy <read_start_2> >= <read_start_1> + <length_1>

        Return value must be in the form (also see the project pdf):
        [(<read_start_1>, <reference_start_1, length_1), (<read_start_2>, <reference_start_2, length_2), ...]

        If no good matches are found: return the best match you can find or return []

        Time limit: 0.5 seconds per read on average on the provided data.
        """

        def exact_suffix_matches_indexed(p, M, occ, ep, sp, d):
            """
            Exact suffix matches given starting sp, ep.
            """

            range, length = (sp, ep+1), 0

            while length < len(p):

                c = p[-length - 1]

                if not c in M.keys():
                    break

                sp = M[c] + occ[c][max(0, sp-1)]
                ep = M[c] + occ[c][ep] - 1

                if sp > ep:
                    break

                range = (sp, ep+1)
                length += 1
                
            return (range, length + d)

        # Can find all max values from the list but we should determine whether the performance really needs to be improved first.
        def one_sub_suffix_matches(p, M, occ):
            """
            Suffix matches with up to one substitution.
            """

            chars = ['A', 'C', 'G', 'T']

            if not p[-1] in M.keys():
                return (None, 0)
            
            sp = M[p[-1]]
            ep = sp + occ[p[-1]][-1] - 1
            range, length = (sp, ep+1), 1

            m_len = lambda x: x[0][1]
            best = max([(exact_suffix_matches(p[:-1] + sub, M, occ), (len(p)-1, sub)) for sub in chars if sub != p[-1]], key=m_len)

            while length < len(p):

                i = len(p) - length - 1
                c = p[i]

                best_sub = max([(exact_suffix_matches_indexed(p[:i] + sub, M, occ, ep, sp, length), (i, sub)) for sub in chars if sub != c], key=m_len)
                best = best_sub if m_len(best_sub) > m_len(best) else best

                if not c in M.keys():
                    break

                sp = M[c] + occ[c][max(0, sp-1)]
                ep = M[c] + occ[c][ep] - 1

                if sp > ep:
                    break

                range = (sp, ep+1)
                length += 1
            
            return best if m_len(best) > length + 1 else ((range, length), None) # removing the "+1" will allow extensions by one character

        def bowtie_1(p, M, occ):
            """
            Returns an alignment of a query string p in the string represented by M, occ, with up to MAX_SUBS mismatches.
            Only returns alignments that do not terminate in mismatches (i.e. will not extend an alignment by only one
            character in an iteration).

            Input:
                p: the pattern string
                M, occ: buckets and repeats information used by sp, ep

            Output: a tuple (range, length)
                range: a tuple (start inclusive, end exclusive) of the indices in sa that contains
                    the longest suffix of p as a prefix. range=None if no indices matches any suffix of p
                length: length of the longest suffix of p found in s. length=0 if no indices matches any suffix of p
            """
            
            MAX_SUBS = 6        # We should do some thinking, this can probably be smaller.
            MAX_BACKTRACKS = 50 # Testing necessary

            # Greedy implementation which runs for MAX_SUBS iterations, choosing the best substitution per iteration.
            best_align, subs = None, {}

            while len(subs) < MAX_SUBS:

                one_sub = one_sub_suffix_matches(p, M, occ)
                
                best_align = one_sub[0]
                
                if not one_sub[1]:
                  break
                
                i, c = one_sub[1]

                subs[i] = c
                p = p[:i] + c + p[i+1:]

            return best_align, subs

        def diff_k(s_1, s_2, i_1, i_2, n, k):
            """
            Returns the pairwise difference of the first n characters of s_1 and s_2 starting at indices i_1 and i_2, respectively,
            up to a difference of k, and otherwise returns a number higher than k.
            """

            diff = 0

            for i in range(n):
                if s_1[i_1+i] == s_2[i_2+i]:
                    diff += 1
                if diff > k:
                    break
            
            return diff

        def bowtie_2(p):
            """
            Returns an alignment of a query string p to the genome.

            Input:
                p: the pattern string

            Output: a tuple (gene_id, isoform_id, index)
                gene_id: id of the gene with best alignment to p
                isoform_id: id of the isoform with best alignment to p
                index: start index in the isoform
            """
            
            MAX_SUBS = 6

            SEED_LEN = 16
            SEED_GAP = 10

            seeds = [p[i:i+SEED_LEN] for i in range(0, len(p), SEED_GAP)]

            # Priority 1
            hits = dict()

            for gene in self.genes.items():

                gene_hits = dict()
                
                for iso in gene[1].items():

                    iso_hits = dict()

                    sa_ranges = [bowtie_1(seed, iso[1]['M'], iso[1]['occ']) for seed in seeds]
                    
                    # ex. sa_range: (((3, 4), 8), {10: 'C', 6: 'C'})
                    for i in range(len(sa_ranges)):

                        sa_range_i, sa_range_j = sa_ranges[i][0][0][0], sa_ranges[i][0][0][1]
                        sa_range_len = sa_ranges[i][0][1]

                        s_hits = [iso[1]['sa'][j] + sa_range_len - SEED_LEN - i * SEED_GAP for j in range(sa_range_i, sa_range_j)]
                        s_hits = filter(s_hits, lambda x: x <= len(iso['s']) - len(p)) # ensure the whole length of hit is valid

                        for hit in s_hits:
                            iso_hits[hit] = iso_hits[hit] + 1 if hit in iso_hits.keys() else 1
                        
                    gene_hits[iso[0]] = {item[1]: item for item in iso_hits.items()}
                
                hits[gene[0]] = gene_hits
            
            best_align = (None, None, 0)
            least_subs = MAX_SUBS + 1
                    
            for n in range(len(seeds) - 1, 1, -1): # check starting indices with 2 or more hits

                changed = False # can remove this condition to speed up (remove bottom layer)

                for gene in hits.items():

                    for iso in gene[1].items():

                        for i in iso[1][n]:

                            iso_s = self.genes[gene[0]][iso[0]]['s']
                            subs = diff_k(p, iso_s, 0, i, len(p), MAX_SUBS)

                            if subs < least_subs:
                                best_align = (gene[0], iso[0], i)
                                least_subs = subs
                                changed = True
                            
                            if subs == 0:
                                break
                
                if not changed and best_align[0] != None:
                    break

            if best_align[0] != None:
                return best_align

                    
            # Priority 2

            return None # TODO

        # Priority 1: Align to known isoform with 6 or less mismatches
        # 1. Splice query into 5 seeds of length 16
        # 2. Find all the "transcriptomes" by splicing together the exons given by the Gene object
        # 3. For each of the transcriptomes:
        #     3.1. Call bowtie_1 on each of the seeds for the transcriptome
        #     3.2. Goal 1: find 5 sequential seeds with offset 10
        #     3.3. Goal 2: find 2/3/4 sequential seeds, then infer where the remaining seeds are
        #     3.4. Check pairwise value of each of these seeds via direct character comparison and return best
        #
        # Priority 2: Align to unknown isoforms
        # 1. Basically the same as the other one, but if there are 2/3/4 sequential seeds there may be introns in between.
        # 2. Determine gap length and then use pairwise character comparison to determine where the "intron" should be.

        def genome_read(alignment):
          """
          Converts an alignment to (start index, end index) tuples in the genome

          Input: a tuple (gene_id, isoform_id, i)
              gene_id: id of the gene with best alignment to p
              isoform_id: id of the isoform with best alignment to p
              i: start index in the isoform
          Output:
             EDIT

          """
          
        alignment = bowtie_2(read_sequence)
        return genome_read(alignment) if alignment[0] != None else []
