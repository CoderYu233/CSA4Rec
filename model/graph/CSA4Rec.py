import torch
import torch.nn as nn
import torch.nn.functional as F

from data_argument import dataloader
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf, ModelConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE



class CSA4Rec(GraphRecommender):
    def __init__(self, conf, training_set, test_set,is_data = False,Globalargs=None):
        super(CSA4Rec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['CSACN'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])


        self.model = CSA4Rec_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl,is_data,Globalargs)


    def train(self):
        model = self.model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        path = "./checkpoint/"
        data_na = self.config['dataname']
        torch.save(model.state_dict(), path + "Beauty_GCL" + str(self.emb_size) + ".pth")
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class CSA4Rec_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl,is_data = False,args = None):
        super(CSA4Rec_Encoder, self).__init__()
        self.args = args
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers

        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.kv = TorchGraphInterface.convert_sparse_mat_to_tensor(data.kv_adj).cuda()



        if args !=None:
            if self.args.activation == 'sigmoid':
                self.f = nn.Sigmoid()
            elif self.args.activation == 'relu':
                self.f = nn.ReLU()
            elif self.args.activation == 'tanh':
                self.f = nn.Tanh()
            else:
                self.f = lambda x:x
            self.layers = self.args.layer
            self.keep_prob = self.args.keepprob
            self.A_split = False
        self.is_data = is_data
        self.dataset = None

        if self.is_data:
            self.dataset = dataloader.Loader(self.args)
        self.Graph = None
        if self.dataset!=None:
            self.num_users = self.dataset.n_users
            self.num_items = self.dataset.m_items

        self.cl_rate = 0.2

        self.temp = 0.15
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })

        return embedding_dict.cuda()

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        if self.Graph!=None:
            norm_adj = self.Graph
        else:
            norm_adj = self.sparse_norm_adj

        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings

    def getUsersRating(self, users):
        all_users = self.embedding_dict['user_emb']
        all_items = self.embedding_dict['item_emb']
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getUsersUsers(self, users):
        all_users = self.embedding_dict['user_emb']
        users_emb = all_users[users.long()]

        return self.f(torch.matmul(users_emb, all_users.t()))

    def getItemsItems(self, items):
        all_items = self.embedding_dict['item_emb']
        items_emb = all_items[items.long()]

        return self.f(torch.matmul(items_emb, all_items.t()))

    def getItemsRating(self, items):
        all_users = self.embedding_dict['user_emb']
        all_items = self.embedding_dict['item_emb']
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating

    def reset_all_uuii(self):
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet, include_uuii=True)
    def reset_all(self):
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet, include_uuii=False)

    def reset_graph(self):
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        graph_kv = self.kv
        users_emb = self.embedding_dict['user_emb']
        items_emb = self.embedding_dict['item_emb']
        all_emb = torch.cat([users_emb, items_emb])

        shortcut = all_emb

        all_emb_kv = all_emb
        embs_kv = []

        embs = []
        if self.args.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        if self.args.l2:
            l2_norm = torch.norm(light_out, p=2, dim=1, keepdim=True)
            light_out = light_out / l2_norm


        for layer in range(self.layers):
            all_emb_kv = torch.sparse.mm(graph_kv, all_emb_kv)
            embs_kv.append(all_emb_kv)
        embs_kv = torch.stack(embs_kv, dim=1)

        light_out_kv = torch.mean(embs_kv, dim=1)

        light_out = shortcut + self.args.resnet*light_out+self.args.resnet2*light_out_kv

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_dict['user_emb'][users]
        pos_emb_ego = self.embedding_dict['item_emb'][pos_items]
        neg_emb_ego = self.embedding_dict['item_emb'][neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):

        u_idx = torch.unique(torch.tensor([float(x) for x in idx[0]]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.tensor([float(x) for x in idx[1]]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

