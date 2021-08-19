##########################################################
## Hypergraph algorithms 
## Quick Python implementation - small datasets only!
##
## Data representation
##
## h: list of sets (the hyperedges), with 0-based integer vertices
##    example: h = [{0,1},{1,2,3},{2,3,4},{4,6},{5,6}]
## H = list2H(h) (hypergraph)
##    example: H = [[], [], [{0, 1}, {4, 6}, {5, 6}], [{1, 2, 3}, {2, 3, 4}]]
## 
## qH, PH = randomAlgo(H, steps=10, verbose=True, ddeg=False)
## qH, PH = cnmAlgo(H, verbose=True)
## qG, PG = randomAlgoTwoSec(H, steps=10, verbose=True)
## 
##########################################################

from scipy.special import comb as choose
import itertools
from random import shuffle, randint, sample

##########################################################

## return: number of nodes (recall: 0-based), and
## number of edges of each cardinality
def H_size(H):
    M = len(H)
    m = []
    n = 0
    for i in range(M):
        m.append(len(H[i]))
        if(len(H[i])>0):
            j = max(set.union(*H[i]))
            if j>n:
                n = j
    return n+1, m

## vertex d-degrees for each d
def d_Degrees(H, n, m):
    M = len(H)
    d = [[]]*M
    for i in range(M):
        if(m[i]>0):
            x = [y for x in H[i] for y in list(x)]
            y = [x.count(i) for i in range(n)]
            d[i] = y
    return d

## vertex total degrees
def Degrees(H,n,m,d):
    M = len(H)
    D = [0]*n
    for i in range(M):
        if(m[i]>0):
            for j in range(n):
                D[j] = D[j] + d[i][j]
    return D

##########################################################

## edge contribution: given (H,A,m)
def EdgeContribution(H,A,m):
    ec = 0
    for i in range(len(H)):
        for j in range(len(H[i])):
            for k in range(len(A)):
                if(H[i][j].issubset(A[k])):
                    ec = ec + 1
                    break
    ec = ec / sum(m)
    return ec

##########################################################

## degree tax - with d-degrees as null model
def d_DegreeTax(A,m,d):
    dt = 0
    for i in range(len(m)):
        if (m[i]>0):
            S = 0
            for j in range(len(A)):
                s = 0
                for k in A[j]:
                    s = s + d[i][k]
                s = s ** i
                S = S + s
            S = S / (i**i * m[i]**(i-1) * sum(m))
            dt = dt + S 
    return dt
    
## degree tax - with degrees as null model
def DegreeTax(A,m,D):
    dt = 0
    vol = sum(D)
    M = sum(m)
    ## vol(A_i)'s
    volA = [0]*len(A)
    for i in range(len(A)):
        for j in A[i]:
            volA[i] = volA[i] + D[j]
        volA[i] = volA[i] / vol
    ## sum over d
    S = 0
    for i in range(len(m)):
        if (m[i]>0):
            x = sum([a**i for a in volA]) * m[i] / M
            S = S + x
    return S

##########################################################

## 2-section: return extended list of edges and edge weights 
def TwoSecEdges(H,m):
    e = []
    w = []
    for i in range(len(m)):
        if(m[i]>0 and i>1):
            den = choose(i,2)
            for j in range(len(H[i])):
                s = [set(k) for k in itertools.combinations(H[i][j],2)]
                x = [1/den]*len(s)
                e.extend(s)
                w.extend(x)
    return e,w 

def TwoSecEdgeContribution(A,e,w):
    ec = 0
    for i in range(len(A)):
        for j in range(len(e)):
            if(e[j].issubset(A[i])):
                ec = ec + w[j]
    ec = ec / sum(w)
    return ec

def TwoSecDegrees(n,e,w):
    d = [0]*n
    for i in range(len(e)):
        d[list(e[i])[0]] = d[list(e[i])[0]] + w[i]
        d[list(e[i])[1]] = d[list(e[i])[1]] + w[i]
    return d

def TwoSecDegreeTax(A,d):
    dt = 0
    for i in range(len(A)):
        s = 0
        for j in list(A[i]):
            s = s + d[j]
        s = s**2
        dt = dt + s
    dt = dt / (4*(sum(d)/2)**2)
    return dt

##########################################################

## take a partition and an edge (set)
## return new partition with new edge "active"
def newPart(A,s):
    P = []
    for i in range(len(A)):
        if(len(s.intersection(A[i])) == 0):
            P.append(A[i])
        else:
            s = s.union(A[i])
    P.append(s)
    return P

##########################################################

def cnmAlgo(H, verbose=False, ddeg=False):
    ## get degrees from H
    n, m = H_size(H) 
    d = d_Degrees(H,n,m)
    D = Degrees(H,n,m,d)
    ## get all edges in a list
    e = []
    for i in range(len(H)):
        e.extend(H[i])
    ## initialize modularity, partition
    A_opt = []
    for i in range(n):
        A_opt.extend([{i}])
    if ddeg:
        q_opt = EdgeContribution(H,A_opt,m) - d_DegreeTax(A_opt,m,d)  
    else:
        q_opt = EdgeContribution(H,A_opt,m) - DegreeTax(A_opt,m,D)
    ## e contains the edges NOT yet in a part
    while len(e)>0:
        q0 = -1
        e0 = -1
        if verbose:
            print('best overall:',q_opt, 'edges left: ',len(e))
        ## pick best edge to add .. this is slow as is!
        for i in range(len(e)):
            P = newPart(A_opt,e[i])
            if ddeg:
                q = EdgeContribution(H,P,m) - d_DegreeTax(P,m,d)
            else:
                q = EdgeContribution(H,P,m) - DegreeTax(P,m,D)
            if q>q0:
                e0 = i
                q0 = q
        ## add best edge found if any
        if(q0 > q_opt):
            q_opt = q0
            A_opt = newPart(A_opt,e[e0])
            ## remove all 'active' edges
            r = []    
            for i in range(len(e)):
                for j in range(len(A_opt)):
                        if(e[i].issubset(A_opt[j])):
                            r.append(e[i])
                            break
            for i in range(len(r)):
                e.remove(r[i])
        ## early stop if no immediate improvement
        else:
            break
    return q_opt, A_opt

##########################################################

## random algorithm - start from singletons, add edges w.r.t. permutation IF q improves
def randomAlgoTwoSec(H, steps=10, verbose=False):
    ## get degrees from H
    n, m = H_size(H) 
    ed, w = TwoSecEdges(H, m)
    d = TwoSecDegrees(n, ed, w)
    ## get all edges in H
    e = []
    for i in range(len(H)):
        e.extend(H[i])
    ## initialize modularity, partition
    q_opt = -1
    A_opt= []
    ## algorithm - go through random permutations
    for ctr in range(steps):
        ## Loop here
        shuffle(e)
        ## list of singletons
        A = []
        for i in range(n):
            A.extend([{i}])
        ## starting (degree) modularity
        q0 = TwoSecEdgeContribution(A, ed, w) - TwoSecDegreeTax(A, d)
        for i in range(len(e)):
            P = newPart(A,e[i])
            q = TwoSecEdgeContribution(P,ed, w) - TwoSecDegreeTax(P, d)
            if q > q0:
                A = P
                q0 = q
        if q0 > q_opt:
            q_opt = q0
            A_opt = A
        if verbose:
            print('step',ctr,':',q_opt)
    return q_opt, A_opt


## random algorithm - start from singletons, add edges w.r.t. permutation IF q improves
def randomAlgo(H, steps=10, verbose=False, ddeg=False):
    ## get degrees from H
    n, m = H_size(H) 
    d = d_Degrees(H,n,m)
    D = Degrees(H,n,m,d)
    ## get all edges in H
    e = []
    for i in range(len(H)):
        e.extend(H[i])
    ## initialize modularity, partition
    q_opt = -1
    A_opt= []
    ## algorithm - go through random permutations
    for ctr in range(steps):
        ## Loop here
        shuffle(e)
        ## list of singletons
        A = []
        for i in range(n):
            A.extend([{i}])
        ## starting (degree) modularity
        if ddeg:
            q0 = EdgeContribution(H,A,m) - d_DegreeTax(A,m,d)
        else:   
            q0 = EdgeContribution(H,A,m) - DegreeTax(A,m,D)
        for i in range(len(e)):
            P = newPart(A,e[i])
            if ddeg:
                q = EdgeContribution(H,P,m) - d_DegreeTax(P,m,d)
            else:
                q = EdgeContribution(H,P,m) - DegreeTax(P,m,D)
            if q > q0:
                A = P
                q0 = q
        if q0 > q_opt:
            q_opt = q0
            A_opt = A
        if verbose:
            print('step',ctr,':',q_opt)
    return q_opt, A_opt

##########################################################

## Map vertices 0 .. n-1 to their respective 0-based part number
def PartitionLabels(P):
    n = 0
    for i in range(len(P)):
        n = n + len(P[i])
    label = [-1]*n
    for i in range(len(P)):
        l = list(P[i])
        for j in range(len(l)):
            label[l[j]] = i
    return label

##########################################################

## generate m edges between [idx1,idx2] inclusively
## of size between [size1,size2] inclusively
## Store in a list of lists of sets
def generateEdges(m,idx1,idx2,size1,size2):
    ## init
    L = [[]]*(size2+1)
    for i in range(size2+1):
        L[i]=[]
    v = list(range(idx1,idx2+1))
    if size2>len(v):
        size2 = len(v)
    ## generate - never mind repeats for now
    for i in range(m):
        size = randint(size1,size2)
        L[size].append(set(sample(v,size)))
    return L  

## merge two lists of lists of sets
def mergeEdges(L1,L2):
    l = max(len(L1),len(L2))
    L = [[]]*l
    for i in range(len(L1)):
        L[i] = L1[i]
    for i in range(len(L2)):
        L[i] = L[i] + L2[i]
    ## uniquify
    for i in range(l):
        L[i] = [set(j) for j in set(frozenset(i) for i in L[i])]
    return L

##########################################################

## format Hypergraph given list of hyperedges (list of sets of 0-based integers)
def list2H(h):
    ml = max([len(x) for x in h])
    H = [[]]*(ml+1)
    for i in range(ml+1):
        H[i] = []
    for i in range(len(h)):
        l = len(h[i])
        H[l].append(h[i])
    return H

## two section modularity
def modularityG(H,A):
    n, m = H_size(H) 
    ed, w = TwoSecEdges(H, m)
    d = TwoSecDegrees(n, ed, w)
    return(TwoSecEdgeContribution(A, ed, w) - TwoSecDegreeTax(A, d))

## strict H-modularity
def modularityH(H,A,ddeg=False):
    n, m = H_size(H) 
    d = d_Degrees(H,n,m)
    D = Degrees(H,n,m,d)
    if ddeg:
        return(EdgeContribution(H,A,m) - d_DegreeTax(A,m,d))
    else:
        return(EdgeContribution(H,A,m) - DegreeTax(A,m,D))

##########################################################