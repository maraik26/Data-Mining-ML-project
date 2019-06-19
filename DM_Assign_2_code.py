

# coding: utf-8

import numpy as np
import random as rnd
from operator import itemgetter
import matplotlib.pyplot as plt
import itertools
import csv
import sys
import pickle
from sklearn.metrics import jaccard_similarity_score
 
# 1) Load the data with n=6000 and m=1000. 

charac_matrix = np.loadtxt(open("data-Assignment2.txt", "rb"), delimiter=",")
 
# 2) Signature matrix using p=100 random permutations through minhash

def minHash_signatures(matrix,n):
    hash_funcs = generate_hash_functions(n)
    
    hash_value = []
    for func in hash_funcs:
        val = [hash_minHash(i,func[0],func[1],(matrix.shape[0])) for i in range(matrix.shape[0])]
        hash_value.append(val)
    
    SIG = np.zeros((n,matrix.shape[1])) + float('inf')

    for c in range(matrix.shape[1]):
        for r in range(matrix.shape[0]):
            if matrix[r,c] != 0:
                for i in range(n):
                    hi = hash_value[i-1]
                    SIG[i-1,c] = min(SIG[i-1,c],hi[r])
    return SIG

def hash_minHash(x,var,cons,n):
    return (var*x + cons) % (n + 101)


def generate_hash_functions(n):
    hash_funcs = []

    for i in range(n):
        var = rnd.randint(0,1000)
        cons = rnd.randint(0,1000)
        hash_funcs.append([var,cons])
    return hash_funcs

permuts = 100
SIG = minHash_signatures(matrix=charac_matrix, n=permuts)
 
# 3)Determine b and r for for 100 permut and t=0.3 (i.e., 30%), where b is the number of bands and r is the number of rows per band, and b*r=p. Show the S curve to justify your choice

s = np.linspace(0,1,100)
def a(r,b, s):
    return(1-(1-s**r)**b)

plt.plot(s, a(4,25,s), 'r', label = 'b=25, r=4') 
plt.plot(s, a(5,20,s), 'b', label = 'b=20, r=5') 
plt.plot(s, a(1,100,s), 'g', label = 'b=100, r=1')
plt.plot(s, a(10,10,s), 'c', label = 'b=10, r=10')  
plt.plot(s, a(2,50,s), 'y', label = 'b=50, r=2')  
plt.plot([0.3,0.3],[0.0, 1.0], label = 'Threshold = 0.3')
plt.ylabel('Plot of sharing bucket')
plt.title('S-curve')
plt.xlabel('Simiilarity')
plt.legend(bbox_to_anchor=(1,1), loc=1)
plt.savefig('img100.png')

#For permuts:100, b=25, r=4
#For permuts:500, b=125, r=4

#Using best results for b and r:
permuts = 100
b = 25 
r = 4 
t = 0.3

# 4) Find candidate pairs of signatures using LHS, by choosing a hash function h with k=10,000 buckets. 

def init_bucket_array(num_of_bands,num_of_buckets):
    array_buckets = []
    for band in range(num_of_bands):
        array_buckets.append([[] for i in range(num_of_buckets)])
    return array_buckets

def jaccard_similarity(a,b):
    vec1, vec2 = np.array(a), np.array(b)
    keepers = np.where(np.logical_not((np.vstack((vec1, vec2)) == 0).all(axis=0)))
    similarity = jaccard_similarity_score(vec1[keepers], vec2[keepers])
    return similarity

num_of_buckets = 10000

def get_candidates(b,r,num_of_buckets,t):

    array_buckets = init_bucket_array(b,num_of_buckets)

    candidates = dict()  
    i = 0

    for b_item in range(b):
        buckets = array_buckets[b_item]
        band = SIG[i : i+r , :]

        for matrix_col in range(band.shape[1]):
            key = int((7*sum(band[:,matrix_col]) + 157) % (num_of_buckets)) 
            buckets[key].append(matrix_col)
        i += r

        for item in buckets:
            
            if len(item) > 1:

# Check the similiarity of all possible pairs in the buckets

                all_pairs = list(itertools.combinations(item, 2))
                for pair in all_pairs:
                    if pair not in list(candidates.keys()):

                        col1 = band[:,pair[0]]
                        col2 = band[:,pair[1]]                        
                        similarity = jaccard_similarity(col1,col2)
                        
                        if similarity >= t:
                            candidates[pair] = similarity
                        
    return candidates

candidates = get_candidates(b,r,num_of_buckets,t)
                        
print('Candidates: {}'.format(len(candidates)))

# 5)Determine FP and FN of the result in 4, i.e., the number of dissimilar signature pairs that are candidate pairs, and the number of similar signature pairs that are not candidate pairs. 

#Find False Positive

def get_false_positives(matrix, pairs_list, t):
    false_positives = {}
    for pair in pairs_list:
        A = matrix[:,pair[0]]
        B = matrix[:,pair[1]]
        similarity = jaccard_similarity(A, B)

        if similarity < t:
            false_positives[pair] = similarity
    return false_positives

candidates_pairs_list = list(candidates.keys())
step5_fp = get_false_positives(SIG, candidates_pairs_list, t)
print('5 False Positives: {}'.format(len(step5_fp)))


#Find False Negative

def get_false_negatives(matrix,pairs_list, t):
    false_negatives = dict()
    all_matrix_combinations = list(itertools.combinations(range(matrix.shape[1]), 2))
    non_candidate_pairs = list(set(all_matrix_combinations) - set(pairs_list))

    for pair in non_candidate_pairs:
        A = matrix[:,pair[0]]
        B = matrix[:,pair[1]]
        similarity = jaccard_similarity(A, B)

        if similarity >= t:
            false_negatives[pair] = similarity
    return false_negatives


step5_fn = get_false_negatives(SIG, candidates_pairs_list, t)
print('5 False Negatives: {}'.format(len(step5_fn)))                     

# 6) Find similar pairs of signatures by removing FP

remaining_candidate_pairs = list(set(candidates.keys()) - set(step5_fp.keys()))

len(candidates), len(step5_fp), len(remaining_candidate_pairs)

# 7) Find similar pairs of objects from the remaining candidate pairs in 6 and determine the FP and FN of this result.  

step7_fp = get_false_positives(charac_matrix, remaining_candidate_pairs, t)
print('7 False Positives: {}'.format(len(step7_fp)))

step7_fn = get_false_negatives(charac_matrix, remaining_candidate_pairs, t)
print('7 False Negatives: {}'.format(len(step7_fn)))


#Question 2: Draw the figure 1 for FP and figure 2 for FN in 5 for different choices of b
plt.figure()

FP100 = [165,469,4046, 19351, 33195]
FN100 = [2874,2583,2734,1259, 994]
b = [4,5,10,20,25]
    
plt.plot(b, FP100, 'r', label = 'FP_5')
plt.plot(b, FN100, 'b', label = 'FN_5')

plt.ylabel('FP and FN')
plt.title('FP_FN_Question5')
plt.xlabel('b')
plt.legend(bbox_to_anchor=(1,1), loc=1)
plt.savefig('FP_FN100_5.png')
#plt.show()

#Question 3: Draw the figure 1 for FP and figure 2 for FN in 7 for different choices of b
plt. figure()

FP100_7 = [7,11,53, 107, 293]
FN100_7 = [7505, 7490,7270,6953, 6383]
b = [4,5,10,20,25]
    
plt.plot(b, FP100_7, 'r', label = 'FP_7')
plt.plot(b, FN100_7, 'b', label = 'FN_7')

plt.ylabel('FP_FN_Question7')
plt.title('FP_FN_Question7')
plt.xlabel('b')
plt.legend(bbox_to_anchor=(1,1), loc=1)
plt.savefig('FP_FN100_7.png')
#plt.show()

