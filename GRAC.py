import scipy.io as sio
import time
# import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from data import data
import argparse, os,sys,inspect
from Laplacian_HGCN import Laplacian
from sklearn.preprocessing import normalize
import random
# from sknetwork.clustering import Louvain, BiLouvain, modularity


p = argparse.ArgumentParser(description='Choose Parameter for Filter interpolation')
p.add_argument('--data', type=str, default='coauthorship', help='data name (coauthorship/cocitation)')
p.add_argument('--dataset', type=str, default='dblp', help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation)')
# p.add_argument('--split', type=int, default=0, help='train-test split used for the dataset')
p.add_argument('--tol', type=float, default=.5, help='tolerance')
p.add_argument('--power', type=int, default=100, help='order-k')
p.add_argument('--gpu', type=int, default=None, help='gpu number to use')
p.add_argument('--cuda', type=bool, default=False, help='cuda for gpu')
p.add_argument('--seeds', type=int, default=0, help='seed for randomness')
p.add_argument('--mediators', type=int, default=1, help='use Mediators for Laplacian from FastGCN')
p.add_argument('--normalize', type=str, default=None, help='Use l2 or l1 norm or None')
p.add_argument('--alpha', type=float, default=0.5, help='balance parameter')
p.add_argument('--num_runs', type=int, default=1, help='num_runs')

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
    degree_mat = np.mat(np.diag(DV))

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
        L = I - G

        return L, H.dot(W.dot(HT)) - degree_mat, degree_mat


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


def Incidence_mat(features, Hypergraphs):
    print("creating incidence matrix")
    Incidence = np.zeros(shape=(features.shape[0], len(Hypergraph)))
    for edgei, (k, v) in enumerate(Hypergraph.items()):
        for i in v:
            Incidence[i][edgei] = 1
    return Incidence

if __name__ == '__main__':
    # Using datasets from HyperGCN: A New Method For Training Graph Convolutional Networks on Hypergraphs NIPS 2019
    # coauthorship: cora, dblp
    # cocitation: citeseer, cora, pubmed
    # mediators used for all datasets
    # l2 norm used on all datasets except citeseer and DBLP
    # tolerance 0.1 used for all datasets, but 0.5 for DBLP

    # args = parse()
    print('==================== Norm: {} ================'.format(args.normalize))
    dataset = data.load(args.data, args.dataset)

    # {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}



    for m in range(args.num_runs):
        Hypergraph = dataset['hypergraph']
        features = dataset['features']
        labels = dataset['labels']
        num_nodes = dataset['n']
        num_hyperedges = dataset['e']

        # Incidence = Incidence_mat(features, Hypergraph)
        # Lap, _, degree_mat = preprocess_adj(Incidence)

        labels = np.asarray(np.argmax(labels, axis=1))
        # labels = np.squeeze(labels, axis=1)
        k  = len(np.unique(labels))
        print('k: {}, labels: {}, labels.shape: {}'.format(k, labels, labels.shape))

        alpha = args.alpha
        max_count = 2
        

        # orig_features = features.copy()
        #features = normalize(features, norm='l2', axis=0) ###test ot good
      

        print('============================================trial: {}, Tol: {}, count: {} normalize: {}, mediators: {}================================'.format(m+1, args.tol, max_count, args.normalize, args.mediators))
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
        # modularity_list = []

        intra_list.append(10000000)
        inter_list.append(10000000)
        rep = 1
        count = 0

        t = time.time()

        for p in range(1, args.power + 1):
            seed = args.seeds
            np.random.seed(seed)
            random.seed(seed)

            e = p - 1

            adj_normalized = Laplacian(num_nodes, Hypergraph, features, args.mediators)
            adj_normalized = (1-alpha) * sp.eye(adj_normalized.shape[0]) + alpha * adj_normalized

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

            features = adj_normalized.dot(features)

            if args.normalize == 'l2':
                Trick = normalize(features, norm='l2', axis=1) 
            
            elif args.normalize == 'l1':
                Trick = normalize(features, norm='l1', axis=1)
            
            else:
                Trick = features

            u, s, v = sp.linalg.svds(Trick, k=k, which='LM')

            for i in range(rep):

                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=args.seeds).fit(u)
                predict_labels = kmeans.predict(u)

                # Ncut[i] = Normalized_cut(predict_labels, Lap, degree_mat)
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

            print('dataset: {}_{}, power: {}, ac: {}, f1: {}, nm: {}, intraD: {}, InterD: {}, pre: {}, rec: {}, adj_score: {}'.format(args.dataset, args.data, p, acc_means, f1_means, nmi_means, intramean, intermean, pre_mean, rec_mean, adj_smean))

            if intra_list[e]- intra_list[p] <= args.tol:
                count += 1
                print('count: {}'.format(count))

            if count >= max_count:
                print('=====================Breaking As Condition Met================')
                print('power: {}, ac: {}, f1: {}, nm: {}, intraD: {}, InterD: {}, pre: {}, rec: {}, adj_score: {}'.format(p, acc_list[e], f1_list[e], nmi_list[e], intra_list[e], inter_list[e], precision_list[e], recall_macro_list[e], adj_score_list[e]))
                t = time.time() - t
                print('time taken: {}'.format(t))
                break
