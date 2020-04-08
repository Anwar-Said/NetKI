import numpy as np
import math
import scipy.io
import time
from grakel import GraphKernel, datasets
# import csv,igraph
import networkx as nx
from numpy.linalg import linalg as LA
from scipy.sparse import csc_matrix
import julia
class helper:
    jl = None
    epsilon = 0.5
    theta = 0.5
    def create_env(self):
        print("creating environment...")
        # julia.install()
        self.jl = julia.Julia()
        #in case of julia path issue, uncomment jl.install()
        self.jl.include("../julia/LapSolver")

    # fetch dataset from graph kernel website using grakel
    def return_dataset(self, file_name):
        dd = datasets.fetch_dataset(file_name, verbose=True)
        graph_list = []
        for gg in dd.data:
            v = set([i[0] for i in gg[0]]).union(set([i[1] for i in gg[0]]))
            g_ = nx.Graph()
            g_.add_nodes_from(v)
            g_.add_edges_from([(i[0], i[1]) for i in gg[0]])
            graph_list.append(nx.adjacency_matrix(g_).todense())
        y = dd.target
        return graph_list, y

    def EstimateReff(self,A,B,N,E):
        # print("estimating effective resistance")
        itr = math.ceil(math.log(N,10))
        itr = 50
        Q = np.random.binomial(1,0.5,size = (itr,E))
        Q = csc_matrix((Q*2)-1)
        QB = Q.dot(B).todense()
        QB = np.array(QB)
        Z = self.jl.LapSolv_ER(A,QB,itr)
        return np.array(Z)
    def computeCentAll(self,distance, A,N,edges):
        M = math.ceil(((pow(self.epsilon, -2))) * (math.log(N, 10)))
        M = 100
        index_score = np.zeros((N, N), dtype=np.float32)
        LapSol = self.jl.LapSolv_Score(A,M)
        LapSol = np.array(LapSol)
        for e in edges:
            u, v = e
            centrality = abs((((1 - self.theta) * (N / M)) * (np.sum((LapSol[:, u] + LapSol[:, v]) ** 2))) / (
                    1 - (1 - self.theta) * LA.norm(distance[:, u] - distance[:, v]) ** 2))
            index_score[u][v] = centrality
            index_score[v][u] = centrality

        return index_score
    def GenerateAllApproxEmb(self, data):
        max_val = 1
        index_matrix = []
        for index, A in enumerate(data):
            g1 = nx.from_numpy_matrix(A)
            if not nx.is_connected(g1):
                #need to compute max connected component and relabel it
                g2 = max(nx.connected_component_subgraphs(g1), key=len)
                lab = g2.nodes()
                mapping = {}
                for l, n in enumerate(lab):
                    mapping[n] = l
                relabelled_graph = nx.relabel_nodes(g2, mapping)
                A1 = nx.adjacency_matrix(relabelled_graph)
                N = A1.shape[0]
                E = relabelled_graph.number_of_edges()
                A1 = A1.astype(np.float32)
                edges = relabelled_graph.edges()
                B = nx.incidence_matrix(relabelled_graph, oriented=True).transpose()
                distance = self.EstimateReff(A1, B, N, E)
                index_score = self.computeCentAll(distance, A1, N, edges)
                index_matrix.append(index_score)
            else:
                A = csc_matrix(A)
                N = A.shape[0]
                E = g1.number_of_edges()
                A = A.astype(np.float32)
                edges = g1.edges()
                B = nx.incidence_matrix(g1, oriented=True).transpose()
                distance = self.EstimateReff(A, B, N, E)
                index_score = self.computeCentAll(distance, A, N, edges)
                index_matrix.append(index_score)
            mx = np.max(index_score)
            if(mx>max_val):
                max_val = mx
            if(index%100==0):
                print("{} embeddings generated.".format(index))
        return index_matrix,max_val

    def load_data_from_numpy(self):
        # data = np.load("../../codes/data/substance_balanced_dataset.npy", allow_pickle=True)
        # data = np.load("../../codes/data/ARE/ARE_dataset.npy", allow_pickle=True)
        data = np.load("../../codes/data/ARE_ER_combined.npy", allow_pickle=True)
        adj = np.array(data[:-1])
        y = np.array(data[-1])
        return list(adj), list(y)

    def remove_outliers(self,data):
        scaled_data = []
        max_val = 0
        all_ = []
        for s in data:
            m = s.flatten()
            all_.extend(m)
        all__ = [x for x in all_ if x > 0]
        q75 = np.percentile(all__, 75)
        q25 = np.percentile(all__, 25)
        IQR = (q75 - q25) * 2.5
        avg = np.mean(all__)
        for d in data:
            d[(d > (q75 + IQR))] = avg
            max_ = np.max(d)
            if (max_ > max_val):
                max_val = max_
            scaled_data.append(d)
        return scaled_data, max_val

    def load_data(self,dataset):
        print("loading dataset...")
        if(dataset=="mutag"):
            mat = scipy.io.loadmat('../data/G_' + dataset + '.mat')
            graphs = mat['G_mutag'][0]
            matt = scipy.io.loadmat('../data/Y_' + dataset + '.mat')
            data_labels = np.concatenate(matt['Y'])
        else:
            mat = scipy.io.loadmat('../data/'+ dataset + '.mat')
            graphs = mat['data'][0]
            data_labels = (mat['labels'])
        print("{} dataset loaded. ".format(dataset))
        data_labels_binary = []
        for label in data_labels:
            if label==1:
                new_label=1
            else:
                new_label=0
            data_labels_binary.append(new_label)
        data_labels_binary = np.array(data_labels_binary)

        return graphs, data_labels_binary

    def scale_data(self,scores):
        scaled_data = []
        for s in scores:
            flt = s.flatten()
            q75, q25 = np.percentile(flt, [75, 25])
            iqr = q75 - q25
            flt[(flt > q75 + (1.5 * iqr))] = np.mean(flt)
            scaled_data.append(flt)
        max_val = self.return_max(scaled_data)
        return scaled_data, max_val

    def load_scores(self, dataset):
        path = '../emb_dir/'+dataset
        scores = []
        total = 188
        for s in range(0,total):
            f_name = str(s)
            scores.append(np.load(path+'/'+f_name+'.npy'))

        return scores
    def return_max(self, score):
        max_val = 1
        for s in score:
            m = np.max(s)
            if m>max_val:
                max_val = m
        return max_val
    def load_binary_labels(self, dataset):
        matt = scipy.io.loadmat('../data/Y_' + dataset + '.mat')
        data_labels = np.concatenate(matt['Y'])
        data_labels_binary = []
        for label in data_labels:
            if label == 1:
                new_label = 1
            else:
                new_label = 0
            data_labels_binary.append(new_label)
        data_labels_binary = np.array(data_labels_binary)
        return data_labels_binary
    def generateHistogram(self,scores,max_val,w):
        feature_matrix = []
        for emb in scores:
            flt = emb.flatten()
            hist, bin_edges = np.histogram(flt, range=(0, max_val), bins=w)
            feature_matrix.append(hist)
        return np.array(feature_matrix)
    def loadEmbeddings(self,path):
        data = []
        with open(path) as file:
            read = csv.reader(file, delimiter=',')
            for row in read:
                data.append(row)
        file.close()
        print("length:", len(data))
        new_data = []
        for d in data:
            if (len(d) > 0):
                new_data.append(d)
        return np.array(new_data)

    def GenerateExactEmbeddings(self,G):
        nodes = G.number_of_nodes()
        # edges = G.number_of_edges()
        E = G.edges()
        nbins = 200
        # range_hist = (0, 20)
        L = nx.laplacian_matrix(G).todense()
        Lv = np.linalg.pinv(L)
        trace = np.trace(Lv)
        K_index = nodes * trace
        # print("Kirchhoff index of the original graph: ",K_index)
        index_score = np.zeros((nodes, nodes), dtype=np.float32)
        for e in E:
            u,v = e
            induced_graph = G.copy()
            induced_graph.remove_edge(e[0], e[1])
            Lap = nx.laplacian_matrix(induced_graph).todense()
            Lapv = np.linalg.pinv(Lap)
            t = np.trace(Lapv)
            theta_index = nodes * t
            cent = theta_index - K_index
            abs_cent = abs(cent)
            index_score[u][v] = abs_cent
            index_score[v][u] = abs_cent
        return index_score
    def GenerateAllExactEmbeddings(self, data):
        feature_matrix = []
        nbins = 50
        for count,A in enumerate(data):
            g = nx.from_numpy_matrix(A)
            index_score = self.GenerateExactEmbeddings(g)
            hist, bin_edges = np.histogram(index_score.flatten(), bins=nbins)
            feature_matrix.append(hist)
            if (count % 100 == 0):
                print("{} embeddings generated.".format(count))
        return np.array(feature_matrix)