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
import time
import sufarray

ALPHABET = [TERMINATOR] + BASES

def get_suffix_array_package(s):
    sa = sufarray.SufArray(s)
    return sa.get_array()

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
    sa = np.arange(len(s))

    # ranges: index ranges in 'sa' which are not fully sorted.
    # format: [start index (inclusive), end index (exclusive), # of valid ranges]
    ranges = [np.empty(len(s), dtype='int'), np.empty(len(s), dtype='int'), 1] 
    ranges[0][0], ranges[1][0] = 0, len(s)


    for d in range(0, len(s), RADIX):

        if d % 100000 == 0:
            print(d)

        sa_inc = sa + d

        next_ranges = [np.empty(len(s), dtype='int'), np.empty(len(s), dtype='int'), 0] # next set of ranges

        for r in range(ranges[2]):

            # buckets: a dict storing the results for one layer of counting sort.
            # format: {C: (indices where C is the next char, # of valid indices) ... }
            buckets = dict()

            for sa_i in range(ranges[0][r], ranges[1][r]):

                s_i = sa_inc[sa_i]
                key = s[s_i:s_i+RADIX]

                if key in buckets.keys():
                    buckets[key].append(sa[sa_i])
                else:
                    buckets[key] = [sa[sa_i]]
            
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
    
    return sa.tolist()

def get_bwt(s, sa):
    """
    Input:
        s: a string terminated by a unique delimiter '$'
        sa: the suffix array of s

    Output:
        L: BWT of s as a string
    """
    bwt_arr = []

    for i in sa:
        if abs(i) > len(s):
            print(i, len(s))
        bwt_arr.append(s[i-1])

    return ''.join(bwt_arr)

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

    def __init__(self, genome_sequence, known_genes, genome_sa=[]):
        """
        Initializes the aligner. Do all time intensive set up here. i.e. build suffix array.

        genome_sequence: a string (NOT TERMINATED BY '$') representing the bases of the of the genome
        known_genes: a python set of Gene objects (see shared.py) that represent known genes. You can get the isoforms 
                     and exons from a Gene object

        Time limit: 500 seconds maximum on the provided data. Note that our server is probably faster than your machine, 
                    so don't stress if you are close. Server is 1.25 times faster than the i7 CPU on my computer
        """

        s = genome_sequence + '$'
        sa = genome_sa if len(genome_sa) > 0 else get_suffix_array(s)
        L = get_bwt(s, sa)
        F = get_F(L)
        M = get_M(F)
        occ = get_occ(L)

        self.whole_genome_FM = {'s': s, 'sa': sa, 'L': L, 'F': F, 'M': M, 'occ': occ} # no introns/exons

        self.known_isoforms_FM = dict() # dict: {gene_id -> dict: {isoform_id -> dict: {'s': string, 'sa': array, 'L': string, 'F': string, 'M': dict, 'occ': dict}}}

        for gene in known_genes:

            gene_data = dict()

            for isoform in gene.isoforms:

                isoform_data = dict()

                isoform_data['start_and_end_indices'] = []
                isoform_data['s'] = ''

                for exon in isoform.exons:
                    isoform_data['s'] += genome_sequence[exon.start:exon.end]
					tup = (exon.start:exon.end)
                    isoform_data['start_and_end_indices'].append(tup)
                isoform_data['s'] += '$'

                isoform_data['sa']  = get_suffix_array(isoform_data['s'])
                isoform_data['L']   = get_bwt(isoform_data['s'], isoform_data['sa'])
                isoform_data['F']   = get_F(isoform_data['L'])
                isoform_data['M']   = get_M(isoform_data['F'])
                isoform_data['occ'] = get_occ(isoform_data['L'])

                gene_data[isoform.id] = isoform_data

            self.known_isoforms_FM[gene.id] = gene_data
        
        self.known_genes = known_genes

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

        def genome_read(alignment):
            """

            Input: a tuple (gene_id, isoform_id, i)
                gene_id: id of the gene with best alignment to p
                isoform_id: id of the isoform with best alignment to p
                i: start index in the isoform
            Output:
                Alignment as a python list of k tuples of ((read start index), (genome start index), (length))

            """
            gene_id = alignment[0]
            isoform_id = alignment[1]
            start_index = [2]
			
            gene_data = known_isoforms[gene_id]
            isoform = gene_data[isoform_id]
			exon_indices = isoform[start_and_end_indices] #this is the start and end indices of each exon in the genome
			read_length = 50
			
			lst = []
			
			c = 0
			while (exon_indices[c][0]-exon_indices[0][0])<start_index:
				c = c+1
			curr_exon = exon_indices[c-1]
			
			frst_tuple = (start_index, (curr_exon[0]+start_index), curr_exon[1]-(curr_exon[0]+start_index))
			lst.append(frst_tuple)
			read_length = 50-(curr_exon[1]-(curr_exon[0]+start_index))
			c = c+1
			curr_exon = exon_indices[c-1]
			
			while read_length > 0:
				if curr_exon[1]-curr_exon[0]>read_length:
					tup = (0,curr_exon[0],curr_exon[1]-curr_exon[0])
					lst.append(tup)
					read_length = read_length - (curr_exon[1]-curr_exon[0])
					c = c+1
					curr_exon = exon_indices[c-1]
				else:
					tup = 0,curr_exon[0],read_length
					lst.append(tup)
            return lst
            
        MAX_SUBS = 6

        SEED_LEN = 16
        SEED_GAP = 10

        MIN_HITS = 2 # assume at least MIN_HITS seeds will hit if there is a match

        p = read_sequence
        seeds = [p[i:i+SEED_LEN] for i in range(0, len(p), SEED_GAP)]

        best_align = (None, None, 0)
        least_subs = MAX_SUBS + 1

        # Priority 1

        hits = dict()

        for gene in self.known_isoforms_FM.items():

            gene_hits = dict()
            
            for isoform in gene[1].items():

                isoform_hit_ct = dict() # {index in isoform[1]['s']: number of hits}

                sa_ranges = [bowtie_1(seed, isoform[1]['M'], isoform[1]['occ']) for seed in seeds]

                # ex. sa_range: (((3, 4), 8), {10: 'C', 6: 'C'})
                for i in range(len(sa_ranges)):
                    
                    if sa_ranges[i][0] == None:
                        continue

                    sa_range_i, sa_range_j = sa_ranges[i][0][0][0], sa_ranges[i][0][0][1]
                    sa_range_len = sa_ranges[i][0][1]

                    isoform_sa, offset = isoform[1]['sa'], sa_range_len - len(seeds[i]) - i * SEED_GAP
                    s_hits = [isoform_sa[j] + offset for j in range(sa_range_i, sa_range_j)]

                    # ensure the whole length of hit is valid
                    s_hits = filter(lambda x: x <= len(isoform[1]['s']) - len(p), s_hits)

                    for hit in s_hits:
                        if hit in isoform_hit_ct.keys():
                            isoform_hit_ct[hit] += 1
                        else:
                            isoform_hit_ct[hit] = 1

                isoform_hits = {num_hits: [] for num_hits in range(MIN_HITS, len(seeds)+1)}
                    
                for hit in isoform_hit_ct.items():
                    if hit[1] >= MIN_HITS:
                        isoform_hits[hit[1]].append(hit[0])

                gene_hits[isoform[0]] = isoform_hits
            
            hits[gene[0]] = gene_hits
                
        for n in range(len(seeds), MIN_HITS-1, -1):

            changed = False # can remove this condition to speed up (remove bottom layer)

            for gene in hits.items():

                for isoform in gene[1].items():

                    for i in isoform[1][n]:

                        iso_s = self.known_isoforms_FM[gene[0]][isoform[0]]['s']
                        subs = diff_k(p, iso_s, 0, i, len(p), MAX_SUBS)

                        if subs < least_subs:
                            best_align = (gene[0], isoform[0], i)
                            least_subs = subs
                            changed = True
                        
                        if subs == 0:
                            return genome_read(best_align)
            
            if not changed and best_align[0] != None:
                break

        if best_align[0] != None:
            return genome_read(best_align)

        # Priority 2

        genome_s, genome_sa = self.whole_genome_FM['s'], self.whole_genome_FM['sa']
        genome_M, genome_occ = self.whole_genome_FM['M'], self.whole_genome_FM['occ']
        genome_sa_ranges = [bowtie_1(seed, genome_M, genome_occ) for seed in seeds]

        for genome_sa_range in genome_sa_ranges:
            print(genome_sa_range, genome_s[genome_sa_range[0][0][0]:genome_sa_range[0][0][0]+SEED_LEN])

        start_hit_ct, end_hit_ct = dict(), dict()
        valid_genome_hit = lambda x: 0 <= x < len(genome_occ)

        # start_hit: True updates start_hits, False updates end_hits
        # add: True adds, False subtracts
        def update_hits(seed, update_start, add):

            genome_sa_range = genome_sa_ranges[seed-1][0]

            if genome_sa_range == None:
                return

            genome_sa_range_i, genome_sa_range_j = genome_sa_range[0]
            genome_sa_range_len = genome_sa_range[1]

            offset = genome_sa_range_len - len(seeds[seed-1]) - (seed-1) * SEED_GAP + (0 if update_start else len(p) - 1)
            genome_s_hits = [genome_sa[i] + offset for i in range(genome_sa_range_i, genome_sa_range_j)]
            print(genome_s_hits)
            genome_s_hits = filter(valid_genome_hit, genome_s_hits)

            print(update_start)

            hit_ct, increment = start_hit_ct if update_start else end_hit_ct, 1 if add else -1

            for genome_s_hit in genome_s_hits:
                if genome_s_hit in hit_ct.keys():
                    hit_ct[genome_s_hit] += increment
                else:
                    hit_ct[genome_s_hit] = 1 # Assume add==True
        
        best_align = []

        # No introns
        
        for seed in range(1, len(seeds)+1):
            update_hits(seed, update_start=False, add=True)

        for key in end_hit_ct.keys():
            print(key, end_hit_ct[key])

        genome_end_hits = {num_hits: [] for num_hits in range(MIN_HITS, len(seeds)+1)}

        for end_hit in end_hit_ct.items():
            if end_hit[1] >= MIN_HITS:
                genome_end_hits[end_hit[1]].append(end_hit[0])
        
        for n in range(len(seeds), MIN_HITS-1):

            changed = False

            for genome_end_hit in genome_end_hits[n]:

                genome_hit = genome_end_hit - len(p) + 1
                subs = diff_k(p, genome_s, 0, genome_hit, len(p), MAX_SUBS)

                if subs < least_subs:
                    best_align = [(0, genome_hit, len(p))]
                    least_subs = subs
                    changed = True
                
                if subs == 0:
                    print('aligned')
                    return best_align

        # One intron

        # for gap_seed in range(1, len(seeds)+1):

        #     if gap_seed == 1:

        #         for seed in range(2, len(seeds)+1):
        #             update_hits(seed, update_start=False, add=False) # subtracting because we just added all seeds

        #     else:

        #         update_hits(gap_seed-1, update_start=True, add=True)
        #         update_hits(gap_seed, update_start=False, add=False)

        #     genome_start_hits = {num_hits: [] for num_hits in range(1, gap_seed)}
        #     genome_end_hits = {num_hits: [] for num_hits in range(1, len(seeds)-gap_seed+1)}

        #     for start_hit in start_hit_ct:
        #         genome_end_hits[start_hit[1]].append(start_hit[0])
        #     for end_hit in end_hit_ct.items():
        #         genome_end_hits[end_hit[1]].append(end_hit[0])
        
        #     max_start_hits, max_end_hits = max(genome_start_hits.keys()), max(genome_end_hits.keys())
        #     starts, ends = genome_start_hits[max_start_hits], genome_end_hits[max_end_hits]

        return best_align
