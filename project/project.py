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
# import sufarray
import functools

ALPHABET = [TERMINATOR] + BASES

# def get_suffix_array_package(s):
#     sa = sufarray.SufArray(s)
#     return sa.get_array()

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

    scores = {ALPHABET[i]: i for i in range(len(ALPHABET))}
    n = len(s)

    class Suffix:
        def __init__(self, i):
            self.index     = i
            self.rank      = scores[s[i  ]]
            self.next_rank = scores[s[i+1]] if i < len(s)-1 else 0
    
    def suffix_cmp(suf1, suf2):
        if suf1.rank == suf2.rank:
            if suf1.next_rank == suf2.next_rank:
                return 0
            return -1 if suf1.next_rank < suf2.next_rank else 1
        else:
            return -1 if suf1.rank < suf2.rank else 1
    
    suffix_key = functools.cmp_to_key(suffix_cmp)
    suffixes = sorted([Suffix(i) for i in range(n)], key=suffix_key)

    def suffix_sort(): # not in use
        max_rank = suffixes[-1].rank
        ranks = dict()
        for suffix in suffixes:
            r = suffix.next_rank
            if r in ranks:
                ranks[r].append(suffix)
            else:
                ranks[r] = [suffix]
        result = [e for r in range(max_rank+1) if r in ranks.keys() \
                for e in ranks[r]]
        ranks = dict()
        for suffix in result:
            r = suffix.rank
            if r in ranks:
                ranks[r].append(suffix)
            else:
                ranks[r] = [suffix]
        result = [e for r in range(max_rank+1) if r in ranks.keys() \
                for e in ranks[r]]
        return result

    ind = [0] * n

    k = 4
    while k < 2*n:

        rank = 0
        prev_rank = suffixes[0].rank
        suffixes[0].rank = rank
        ind[suffixes[0].index] = 0

        for i in range(1, n):
            if suffixes[i].rank == prev_rank and \
                    suffixes[i].next_rank == suffixes[i-1].next_rank:
                suffixes[i].rank = rank
            else:
                prev_rank = suffixes[i].rank
                rank += 1
                suffixes[i].rank = rank
            ind[suffixes[i].index] = i
        
        for i in range(n):
            next_index = suffixes[i].index + k//2
            suffixes[i].next_rank = suffixes[ind[next_index]].rank if \
                    next_index < n else 0

        suffixes.sort(key=suffix_key)
        k *= 2

    return [suffix.index for suffix in suffixes]

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

        self.isoform_indices = dict()

        self.known_isoforms_FM = dict() # dict: {gene_id -> dict: {isoform_id -> dict: {'s': string, 'sa': array, 'L': string, 'F': string, 'M': dict, 'occ': dict}}}

        for gene in known_genes:

            gene_data = dict()

            for isoform in gene.isoforms:

                isoform_data = dict()

                isoform_data['s'] = ''

                self.isoform_indices[isoform.id] = []

                for exon in isoform.exons:
                    isoform_data['s'] += genome_sequence[exon.start:exon.end]
                    self.isoform_indices[isoform.id].append((exon.start, exon.end))
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

        def genome_read(alignment, length):
            """

            Input: 
                alignment: a tuple (isoform_id, i)
                    gene_id: id of the gene with best alignment to p
                    isoform_id: id of the isoform with best alignment to p
                    i: start index in the isoform
                length: the length of the read
            Output:
                Alignment as a python list of k tuples of ((read start index), (genome start index), (length))

            """
            
            exon_index_pairs = self.isoform_indices[alignment[0]]
            read_offset = alignment[1]

            for exon_index_pair_i in range(len(exon_index_pairs)):

                exon_index_pair = exon_index_pairs[exon_index_pair_i]
                start_i, end_i = exon_index_pair
                exon_length = end_i - start_i

                if read_offset < exon_length - 1:

                    read_start_index = start_i + read_offset
                    length_in_exon = end_i - read_start_index

                    result = [(0, read_start_index, min(length_in_exon, length))]

                    if length_in_exon < length:
                        result.append((length_in_exon, exon_index_pairs[exon_index_pair_i+1][0], length - length_in_exon))
                    
                    return result

                else:

                    read_offset -= exon_length

        MAX_SUBS = 6

        SEED_LEN = 16
        SEED_GAP = 10

        MIN_HITS = 2 # assume at least MIN_HITS seeds will hit if there is a match

        try:

            p = read_sequence
            seeds = [p[i:i+SEED_LEN] for i in range(0, len(p), SEED_GAP)]

            best_align = (None, 0)
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
                                best_align = (isoform[0], i)
                                least_subs = subs
                                changed = True
                            
                            if subs == 0:
                                return genome_read(best_align, len(p))
                
                if not changed and best_align[0] != None:
                    break

            if best_align[0] != None:
                return genome_read(best_align, len(p))

            # Priority 2

            genome_s, genome_sa = self.whole_genome_FM['s'], self.whole_genome_FM['sa']
            genome_M, genome_occ = self.whole_genome_FM['M'], self.whole_genome_FM['occ']
            genome_sa_ranges = [bowtie_1(seed, genome_M, genome_occ) for seed in seeds]

            start_hit_ct, end_hit_ct = dict(), dict()
            valid_genome_hit = lambda x: 0 <= x < len(genome_s)

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
                genome_s_hits = filter(valid_genome_hit, genome_s_hits)


                hit_ct, increment = start_hit_ct if update_start else end_hit_ct, 1 if add else -1

                for genome_s_hit in genome_s_hits:
                    if genome_s_hit in hit_ct.keys():
                        hit_ct[genome_s_hit] += increment
                    else:
                        hit_ct[genome_s_hit] = 1 # Assume add==True
            
            best_align = []

            # Mo introns, or introns in first or last seed
            
            for seed in range(1, len(seeds)+1):
                update_hits(seed, update_start=False, add=True)

            genome_end_hits = {num_hits: [] for num_hits in range(MIN_HITS, len(seeds)+1)}

            for end_hit in end_hit_ct.items():
                if end_hit[1] >= MIN_HITS:
                    genome_end_hits[end_hit[1]].append(end_hit[0])
            
            for n in range(len(seeds), MIN_HITS-1, -1):

                changed = False

                for genome_end_hit in genome_end_hits[n]:

                    genome_hit = genome_end_hit - len(p) + 1
                    
                    subs = diff_k(p, genome_s, 0, genome_hit, len(p), MAX_SUBS)

                    if subs < least_subs:
                        best_align = [(0, genome_hit, len(p))]
                        least_subs = subs
                        changed = True
                    
                    if subs == 0:
                        return best_align

                    # implement if gap of 2

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
            #         if start_hit[1] > 0:
            #             genome_end_hits[start_hit[1]].append(start_hit[0])
            #     for end_hit in end_hit_ct.items():
            #         if end_hit[1] > 0:
            #             genome_end_hits[end_hit[1]].append(end_hit[0])
                
            #     if len(genome_start_hits.keys()) == 0 or len(genome_end_hits.keys()) == 0:
            #         continue
            
            #     max_start_hits, max_end_hits = max(genome_start_hits.keys()), max(genome_end_hits.keys())
            #     starts, ends = genome_start_hits[max_start_hits], genome_end_hits[max_end_hits]

            #     start_scores = [np.zeros(len(p), dtype='int') for start in starts]
            #     end_scores = [np.zeros(len(p), dtype='int') for end in ends]
                
            #     for start_i in range(len(starts)):

            #         mismatches = 0

            #         for j in range(len(p)):

            #             mismatches += 0 if genome_s[start_i+j] == p[j] else 1
            #             start_scores[start_i][j] = mismatches
                
            #     for end_i in range(len(starts)):

            #         mismatches = 0

            #         for j in range(len(p)):

            #             mismatches += 0 if genome_s[end_i-j] == p[-j] else 1
            #             end_scores[end_i][-j] = mismatches
                
            #     for exon_i in range(len(p)-1):

            #         best_start, best_end = (MAX_SUBS+1, 0), (MAX_SUBS+1, 0)

            #         for start_i in range(len(starts)):

            #             score = start_scores[start_i][exon_i]
            #             if score < best_start[0]:
            #                 best_start = (score, start_i)
                    
            #         for end_i in range(len(ends)):

            #             score = end_scores[end_i][exon_i+1]
            #             if score < best_end[0]:
            #                 best_end = (score, end_i)

            #         print(best_start, best_end)
                    
            #         alignment_score = best_start[0] + best_end[0]

            #         if alignment_score < least_subs and \
            #             MIN_INTRON_SIZE <= ends[end_i]-starts[start_i]+len(p)-1 <= MAX_INTRON_SIZE:

            #             least_subs = alignment_score
            #             best_align = [(0, starts[start_i], exon_i+1), (exon_i+1, ends[end_i]-len(p)+1+exon_i, len(p)-exon_i-1)]

            #             if alignment_score == 0:
            #                 return best_align

            return best_align
        
        except:

            return []
