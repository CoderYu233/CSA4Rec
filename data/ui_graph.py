import numpy as np
from collections import defaultdict

from tqdm import tqdm

from data.data import Data
from data.graph import Graph
import scipy.sparse as sp
import pickle

class Interaction(Data,Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self,conf,training,test)
        self.train_userSet = defaultdict(set)
        self.train_itemSet = defaultdict(set)
        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.user_num = 0
        self.item_num = 0
        self.__generate_set()
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()
        self.kv_adj = self.kv_getkv()



    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry
            self.user_num = max(self.user_num, int(user))
            self.item_num = max(self.item_num, int(item))
        self.user_num +=1
        self.item_num +=1
        for user in range(self.user_num):
            self.user[str(user)] = int(user)
            self.id2user[int(user)] = str(user)
        for item in range(self.item_num):
            self.item[str(item)] = int(item)
            self.id2item[int(item)] = str(item)
        for entry in self.training_data:
            user1, item1, rating = entry
            self.training_set_u[user1][item1] = rating
            self.training_set_i[item1][user1] = rating
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user or item not in self.item:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)
        for entry in self.training_data:
            user1, item1, rating = entry
            user1 = int(user1)
            item1 = int(item1)
            self.train_itemSet[item1].add(user1)
            self.train_userSet[user1].add(item1)

    def kv_getkv(self):
        uu= self.config.config['uu']
        ii = self.config.config['ii']
        dataname = self.config.config['dataname']
        file_uu = "./dataset/" +dataname+"/"+dataname+"_uu=50_dict"+ ".pkl"
        with open(file_uu, 'rb') as f:
            uu_dict = pickle.load(f)

        file_ii = "./dataset/" + dataname+"/"+dataname+"_ii=50_dict" + ".pkl"
        with open(file_ii, 'rb') as f:
            ii_dict = pickle.load(f)

        print("uu=={} ii=={}".format(uu,ii))
        print("uuii load user_len{}".format(len(uu_dict)))
        uu_row_idx = []
        uu_col_idx = []
        uu_score = []

        ii_row_idx = []
        ii_col_idx = []
        ii_score = []
        for i in uu_dict:
            for index,j in enumerate(uu_dict[i]):
                if index+1 > int(uu):
                    break
                score = uu_dict[i][j]
                uu_row_idx.append(i)
                uu_col_idx.append(j)
                # uu_score.append(score)
                uu_score.append(1)
                # uu_score.append(1/(index+1))

        for i in ii_dict:
            for index,j in enumerate(ii_dict[i]):
                if index+1 > int(ii):
                    break
                score = ii_dict[i][j]
                ii_row_idx.append(i)
                ii_col_idx.append(j)
                # ii_score.append(score)
                ii_score.append(1)
                # ii_score.append(1/(index+1))
        n_nodes = self.user_num + self.item_num
        user_np = np.array(uu_row_idx)
        item_np = np.array(uu_col_idx)
        tmp_uu = sp.csr_matrix((uu_score, (user_np, item_np)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)

        user_np = np.array(ii_row_idx)
        item_np = np.array(ii_col_idx)
        tmp_ii = sp.csr_matrix((ii_score, (user_np+ self.user_num, item_np + self.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        adj_mat = tmp_uu+tmp_ii
        # if self_connection:
        #     adj_mat += sp.eye(n_nodes)
        return adj_mat


    def kv_graph_mat(self,uu=50,ii=50):
        n_nodes = self.user_num + self.item_num
        uu_dict = defaultdict(dict)
        ii_dict = defaultdict(dict)
        #create uu
        for i in tqdm(range(self.user_num)):
            user_temp = [0] * self.user_num
            for j in range(self.user_num):
                if i not in self.train_userSet or j not in self.train_userSet:
                    continue
                if i != j:
                    temp_i = self.train_userSet[i]
                    temp_j = self.train_userSet[j]
                    sim = len(temp_i & temp_j) / len(temp_i | temp_j)
                    user_temp[j] = sim
            sorted_list = sorted(user_temp, reverse=True)
            top_k_values = sorted_list[:uu]
            top_k_indices = [user_temp.index(value) for value in top_k_values]
            for k in range(uu):
                uu_dict[i][top_k_indices[k]] = top_k_values[k]
        file = "./checkpoint/" + "amazon_book_uu=50_dict"+ ".pkl"
        with open(file, 'wb') as f:
            pickle.dump(uu_dict, f)

        for i in tqdm(range(self.item_num)):
            user_temp = [0] * self.item_num
            for j in range(self.item_num):
                if i not in self.train_itemSet or j not in self.train_itemSet:
                    continue
                if i != j:
                    temp_i = self.train_itemSet[i]
                    temp_j = self.train_itemSet[j]
                    sim = len(temp_i & temp_j) / len(temp_i | temp_j)
                    user_temp[j] = sim
            sorted_list = sorted(user_temp, reverse=True)
            top_k_values = sorted_list[:ii]
            top_k_indices = [user_temp.index(value) for value in top_k_values]
            for k in range(ii):
                ii_dict[i][top_k_indices[k]] = top_k_values[k]
        file = "./checkpoint/" + "amazon_book_ii=50_dict"+ ".pkl"
        with open(file, 'wb') as f:
            pickle.dump(ii_dict, f)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):

        n_nodes = self.user_num + self.item_num

        row_idx = [self.user[pair[0]] for pair in self.training_data]

        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)

        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)

        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat


    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):

        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):

        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):

        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):

        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m
