import scipy.io as sio
import time
# import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans, SpectralClustering
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from data import data
import argparse, os,sys,inspect
from Laplacian_HGCN import Laplacian
from sklearn.preprocessing import normalize
import random


p = argparse.ArgumentParser(description='Choose Parameter for Filter interpolation')
p.add_argument('--data', type=str, default='coauthorship', help='data name (coauthorship/cocitation)')
p.add_argument('--dataset', type=str, default='cora', help='dataset name (e.g.: cora/dblp/acm for coauthorship, cora/citeseer/pubmed for cocitation)')
p.add_argument('--num_runs', type=int, default=10, help='number of times to run experiment')
p.add_argument('--gpu', type=int, default=None, help='gpu number to use')
p.add_argument('--cuda', type=bool, default=False, help='cuda for gpu')
p.add_argument('--seeds', type=int, default=0, help='seed for randomness')
p.add_argument('--others', type=str, default='Kmeans', help='Kmeans, cliqueNcut, HyperNcut, HyperA')
# p.add_argument('--alpha', type=float, default=0.5, help='balance parameter')
# p.add_argument('-f') # for jupyter default

args = p.parse_args()

def preprocess_adj(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    
    # the weight of the hyperedge
    W = np.ones(n_edge)
    
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2[np.isinf(DV2)] = 0.
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
                
        G = DV2.dot(H.dot(W.dot(invDE.dot(HT.dot(DV2)))))   
        I = sp.eye(G.shape[0]).toarray()
        L =  I - G
      
        return L

def Hyp_adj(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]

    # the weight of the hyperedge
    W = np.ones(n_edge)

    # the degree of the node
    DV = np.sum(H * W, axis=1)
    DV = np.mat(np.diag(DV))

    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    adj = H.dot(W.dot(HT))
    adj = adj - DV
    
    return adj 

def clique_adj(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    
    # the weight of the hyperedge
    W = np.ones(n_edge)
        
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    return H.dot(W.dot(HT))

def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label

def square_dist(prelabel, feature):
    if sp.issparse(feature):
        feature = feature.todense()
    feature = np.array(feature)

    onehot = to_onehot(prelabel)

    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count == 0] = 1

    mean = onehot.dot(feature) / count
    a2 = (onehot.dot(feature * feature) / count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2 * mean.dot(mean.T))

    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist, inter_dist

def dist(prelabel, feature):
    k = len(np.unique(prelabel))
    intra_dist = 0

    for i in range(k):
        Data_i = feature[np.where(prelabel == i)]
        Dis = euclidean_distances(Data_i, Data_i)
        n_i = Data_i.shape[0]
        if n_i == 0 or n_i == 1:
            intra_dist = intra_dist
        else:
            intra_dist = intra_dist + 1 / k * 1 / (n_i * (n_i - 1)) * sum(sum(Dis))

    return intra_dist

def Normalized_cut(prelabel, Laplacian, degree):
    label = to_onehot(prelabel)
    label = label.T
    k = len(np.unique(prelabel))

    for i in range(k):
        vol = degree[np.where(prelabel == i)]
        vol = vol.T[np.where(prelabel == i)]
        vol = vol.sum(1).sum()
        vol = np.sqrt(vol)
        label[np.where(prelabel == i)] = label[np.where(prelabel == i)] / vol

    return np.trace(label.T.dot(Laplacian.dot(label))).item()

def Incidence_mat(num_nodes, Hypergraph):
    print("creating incidence matrix")
    Incidence = np.zeros(shape=(num_nodes, len(Hypergraph)))
    for edgei, (k, v) in enumerate(Hypergraph.items()):
        for i in v:
            Incidence[i][edgei] = 1
    return Incidence


# def running(others='Kmeans', rep=10, seed=0, features=None, Incidence=None, labels=None, k=None):
def running():

    intra_list = []
    inter_list = []
    acc_list = []
    stdacc_list = []
    f1_list = []
    stdf1_list =[]
    nmi_list = []
    stdnmi_list = []
    ncut_list = []
    precision_list = []
    adj_score_list = []
    recall_macro_list = []

    intra_list.append(10000000)
    inter_list.append(10000000)
    
    t = time.time()
    
    
    IntraD = np.zeros(rep)
    InterD = np.zeros(rep)
    # Ncut = np.zeros(rep)
    ac = np.zeros(rep)
    nm = np.zeros(rep)
    f1 = np.zeros(rep)
    pre = np.zeros(rep)
    rec = np.zeros(rep)
    adj_s = np.zeros(rep)
    # mod = np.zeros(rep)
    
            
    for i in range(rep):
        np.random.seed(seed)
        random.seed(seed)
           
        if others=='Kmeans':
            print('+++++++++++++++++Kmeans++++++++++++++')
            
            u = features
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=seed).fit(u)
            predict_labels = kmeans.predict(u)
        
        elif others=='HyperNcut':
            print('+++++++++++++++++HyperNcut++++++++++++++')

            print('creating Laplacian for HyperNcut')
            adj_norm = preprocess_adj(Incidence)
            print('Done Creating Laplacian')
           
            u, s, v = sp.linalg.svds(adj_norm, k=k, which='LM')
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=seed).fit(u)
            predict_labels = kmeans.predict(u)

        elif others=='HyperA':
            print('+++++++++++++++++HyperA++++++++++++++')
            
            print('creating adj for HyperA')               
            adj_norm = Hyp_adj(Incidence)
            print('Done Creating adj')

            spectral = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=seed)
            predict_labels = spectral.fit_predict(adj_norm)
        
        elif others=='cliqueNcut':
            print('+++++++++++++++++Hyperclique++++++++++++++')
            
            print('creating adj for Hyper-clique')      
            adj_norm = clique_adj(Incidence)
            print('Done Creating adj')

            spectral = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=seed)
            predict_labels = spectral.fit_predict(adj_norm)

        else:
            print('args.others are in [Kmeans, cliqueNcut, HyperNcut, HyperA] else modify lines 340-342 to include other types')

        IntraD[i], InterD[i] = square_dist(predict_labels, features)
        #intraD[i] = dist(predict_labels, features)
        cm = clustering_metrics(labels, predict_labels)
        ac[i], nm[i], f1[i], pre[i], adj_s[i], rec[i] = cm.evaluationClusterModelFromLabel()
         # mod[i] = modularity(predict_labels, adj)
        
    intramean = np.mean(IntraD)
    intermean = np.mean(InterD)
    # ncut_mean = np.mean(Ncut)
    acc_means = np.mean(ac)
    acc_stds = np.std(ac)
    nmi_means = np.mean(nm)
    nmi_stds = np.std(nm)
    f1_means = np.mean(f1)
    f1_stds = np.std(f1)
    # mod_means = np.mean(mod)
    pre_mean = np.mean(pre)
    rec_mean = np.mean(rec)
    adj_smean = np.mean(adj_s)

    # modularity_list.append(mod_means)
    # ncut_list.append(ncut_mean)
    intra_list.append(intramean)
    inter_list.append(intermean)
    acc_list.append(acc_means)
    stdacc_list.append(acc_stds)
    nmi_list.append(nmi_means)
    stdnmi_list.append(nmi_stds)
    f1_list.append(f1_means)
    stdf1_list.append(f1_stds)
    precision_list.append(pre_mean)
    recall_macro_list.append(rec_mean)
    adj_score_list.append(adj_smean)
    if others=='Kmeans':
        print('=====================FinishedKMEANS================')
    
    elif others=='HyperNcut':
        print('=====================FinishedHYPERNCUT================')
    
    elif others=='HyperA':
        print('=====================FinishedHYPERA================')
    
    elif others=='cliqueNcut':
        print('=====================FinishedHYPERCLIQUE================')

    
    print('dataset: {}_{}, ac: {}, f1: {}, nm: {}, intraD: {}, InterD: {}, pre: {}, rec: {}, adj_score: {}'.format(args.dataset, args.data, acc_means, f1_means, nmi_means, intramean, intermean, pre_mean, rec_mean, adj_smean))
    t = time.time() - t
    print('Kmeans time taken: {}'.format(t))
       
    

if __name__ == '__main__':
    '''this is not the file used for the run times in the paper as this file contains too many if statements. 
       A file implementing each model separately was used to report the average run times and memory usage. 
       But the conclusions even using this file are the same.'''

    # Using datasets from HyperGCN: A New Method For Training Graph Convolutional Networks on Hypergraphs NIPS 2019
    # coauthorship: cora, dblp
    # cocitation: citeseer, cora, pubmed

    # args = parse()
    dataset = data.load(args.data, args.dataset)

    # {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}
    labels = dataset['labels']
    num_nodes = dataset['n']
    num_hyperedges = dataset['e']
    labels = np.asarray(np.argmax(labels, axis=1))
    # labels = np.squeeze(labels, axis=1)
    k  = len(np.unique(labels))
    print('k: {}, labels: {}, labels.shape: {}'.format(k, labels, labels.shape))


    # elif args.others=='Kmeans': # for storage studies only
    features = dataset['features']

    # if args.others in ['cliqueNcut', 'HyperNcut', 'HyperA']: # for storage studies only
    Hypergraph = dataset['hypergraph']
    Incidence = Incidence_mat(num_nodes, Hypergraph)

    rep = args.num_runs
    
    others = args.others
    seed = args.seeds
    
    # running(others=others, rep=rep, seed=seed, features=features, Incidence=Incidence, labels=labels, k=k)
    running()
    