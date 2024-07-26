import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp

class AbtCDR(nn.Module):

    def __init__(self, config, args):
        super(AbtCDR, self).__init__()
        
        self.device = args.device
        self.latent_dim = args.embed_size  
        self.n_layers = args.n_layers  
        self.reg_weight = args.regs  
        self.domain_lambda_source = args.lambda_s  
        self.domain_lambda_target = args.lambda_t  
        self.drop_rate = args.drop_rate  
        self.connect_way = args.connect_type  
        self.t = args.t

        self.source_num_users = config['n_users']
        self.target_num_users = config['n_users']
        self.source_num_items = config['n_items_s']
        self.target_num_items = config['n_items_t']
        self.n_fold = 1

        self.n_interaction = args.n_interaction

        self.source_user_embedding = torch.nn.Parameter(torch.empty(self.source_num_users, self.latent_dim))
        self.target_user_embedding = torch.nn.Parameter(torch.empty(self.target_num_users, self.latent_dim))

        self.source_item_embedding = torch.nn.Parameter(torch.empty(self.source_num_items, self.latent_dim))
        self.target_item_embedding = torch.nn.Parameter(torch.empty(self.target_num_items, self.latent_dim))

        self.mapping = torch.nn.Parameter(torch.empty(self.latent_dim,self.latent_dim))

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.reg_loss = EmbLoss()

      
        self.norm_adj_s = config['norm_adj_s']  
        self.norm_adj_t = config['norm_adj_t'] 
        self.domain_laplace = config['domain_adj']


        self.target_restore_user_e = None
        self.target_restore_item_e = None

       
        torch.nn.init.xavier_normal_(self.source_user_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.target_user_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.source_item_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.target_item_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.mapping, gain=1)
        
        self.all_weights = torch.nn.ModuleList()
        
        self.all_weights.append(torch.nn.Linear(64,128))
        self.all_weights.append(torch.nn.Linear(128,128))
        self.all_weights.append(torch.nn.Linear(128,64))

    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding
            item_embeddings = self.source_item_embedding
            norm_adj_matrix = self.norm_adj_s
        else:
            user_embeddings = self.target_user_embedding
            item_embeddings = self.target_item_embedding
            norm_adj_matrix = self.norm_adj_t
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, norm_adj_matrix

    def _split_A_hat(self, X,n_items):
        A_fold_hat = []
        fold_len = (self.source_num_users + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.source_num_users + n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(scipy_sparse_mat_to_torch_sparse_tensor(X[start:end]))
        return A_fold_hat

   
    
    def graph_layer(self, adj_matrix, all_embeddings):
        device = torch.device("cuda")
        side_embeddings = torch.sparse.mm(adj_matrix.to(device), all_embeddings)
        return side_embeddings



    def inter_embedding(self, source_all_embeddings, target_all_embeddings):

        source_user_embeddings, source_item_embeddings = torch.split(source_all_embeddings, [self.source_num_users, self.source_num_items])
        target_user_embeddings, target_item_embeddings = torch.split(target_all_embeddings, [self.target_num_users, self.target_num_items])
        source_user_embeddings_raw, target_user_embeddings_raw = source_user_embeddings,target_user_embeddings
    
        a = torch.matmul(source_user_embeddings,self.mapping)
       
        s = torch.exp((torch.matmul(a,target_user_embeddings.T))/self.t)
        
        sr = F.normalize(s, p=1, dim=1)
        sc = F.normalize(s, p=1, dim=0)

        source_user_embeddings = (source_user_embeddings+torch.matmul(sr,target_user_embeddings))
        target_user_embeddings = (target_user_embeddings+torch.matmul(sc.T,source_user_embeddings))

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_single_(adj):
            degree = torch.sum(adj,dim=1)
            degree_matrix = torch.diag(degree)
            norm_adj = torch.inverse(degree_matrix)@adj
            return norm_adj

        
        adj_s = torch.matmul(s,s.T)
        adj_s = normalized_adj_single_(adj_s + torch.eye(adj_s.shape[0]).cuda())
        adj_t = torch.matmul(s.T,s)
        adj_t = normalized_adj_single_(adj_t + torch.eye(adj_t.shape[0]).cuda())
        for k in range(3):
            source_user_embeddings = torch.mm(adj_s,source_user_embeddings)
            source_user_embeddings = torch.nn.ReLU()(source_user_embeddings)
            source_user_embeddings = F.normalize(source_user_embeddings,p=2,dim=1)

            target_user_embeddings = torch.mm(adj_t,target_user_embeddings)
            target_user_embeddings = torch.nn.ReLU()(target_user_embeddings)
            target_user_embeddings = F.normalize(target_user_embeddings,p=2,dim=1)
            
        source_user_embeddings = (source_user_embeddings_raw + source_user_embeddings)/2
        target_user_embeddings = (target_user_embeddings_raw + target_user_embeddings)/2

        source_alltransfer_embeddings = torch.cat([source_user_embeddings,source_item_embeddings], 0)
        target_alltransfer_embeddings = torch.cat([target_user_embeddings,target_item_embeddings], 0)


        return source_alltransfer_embeddings, target_alltransfer_embeddings

    def forward(self):
        source_all_embeddings, source_norm_adj_matrix = self.get_ego_embeddings(domain='source')
        target_all_embeddings, target_norm_adj_matrix = self.get_ego_embeddings(domain='target')

        source_embeddings_list = [source_all_embeddings]
        target_embeddings_list = [target_all_embeddings]
        for k in range(self.n_layers):
            source_all_embeddings = self.graph_layer(source_norm_adj_matrix, source_all_embeddings)
            target_all_embeddings = self.graph_layer(target_norm_adj_matrix, target_all_embeddings)

            import random
            if random.random() <0.3:
                source_all_embeddings, target_all_embeddings = self.inter_embedding(source_all_embeddings, target_all_embeddings)

            source_norm_embeddings = nn.functional.normalize(source_all_embeddings, p=2, dim=1)
            target_norm_embeddings = nn.functional.normalize(target_all_embeddings, p=2, dim=1)
            source_embeddings_list.append(source_norm_embeddings)
            target_embeddings_list.append(target_norm_embeddings)
           

        if self.connect_way == 'concat':
            source_lightgcn_all_embeddings = torch.cat(source_embeddings_list, 1)
            target_lightgcn_all_embeddings = torch.cat(target_embeddings_list, 1)
        elif self.connect_way == 'mean':
            source_lightgcn_all_embeddings = torch.stack(source_embeddings_list, dim=1)
            source_lightgcn_all_embeddings = torch.mean(source_lightgcn_all_embeddings, dim=1)
            target_lightgcn_all_embeddings = torch.stack(target_embeddings_list, dim=1)
            target_lightgcn_all_embeddings = torch.mean(target_lightgcn_all_embeddings, dim=1)

        source_user_all_embeddings, source_item_all_embeddings = torch.split(source_lightgcn_all_embeddings,
                                                                   [self.source_num_users, self.source_num_items])
        target_user_all_embeddings, target_item_all_embeddings = torch.split(target_lightgcn_all_embeddings,
                                                                   [self.target_num_users, self.target_num_items])

        return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings

    def calculate_single_loss(self, user, item, label, flag):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()

        if flag == "source":
            source_u_embeddings = source_user_all_embeddings[user]
            source_i_embeddings = source_item_all_embeddings[item]
            source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
            source_bce_loss = self.loss(source_output, torch.from_numpy(label).cuda().to(torch.float))

            source_reg_loss = self.reg_loss(source_u_embeddings, source_i_embeddings)

            source_loss = source_bce_loss + self.reg_weight * source_reg_loss

            return source_loss, 0

        if flag == "target":
            target_u_embeddings = target_user_all_embeddings[user]
            target_i_embeddings = target_item_all_embeddings[item]

            target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
            target_bce_loss = self.loss(target_output, torch.from_numpy(label).cuda().to(torch.float))

            target_reg_loss = self.reg_loss(target_u_embeddings, target_i_embeddings)

            target_loss = target_bce_loss + self.reg_weight * target_reg_loss
            return 0, target_loss

    def calculate_cross_loss(self, source_user, source_item, source_label, target_user, target_item, target_label):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()

        source_u_embeddings = source_user_all_embeddings[source_user]
        source_i_embeddings = source_item_all_embeddings[source_item]
        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_embeddings = target_item_all_embeddings[target_item]

        # calculate BCE Loss in source domain
        source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
        source_bce_loss = self.loss(source_output, torch.from_numpy(source_label).cuda().to(torch.float))

        # calculate Reg Loss in source domain
        source_reg_loss = self.reg_loss(source_u_embeddings, source_i_embeddings)

        source_loss = source_bce_loss + self.reg_weight * source_reg_loss

        # calculate BCE Loss in target domain
        target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
        target_bce_loss = self.loss(target_output, torch.from_numpy(target_label).cuda().to(torch.float))

        # calculate Reg Loss in target domain
        target_reg_loss = self.reg_loss(target_u_embeddings, target_i_embeddings)

        target_loss = target_bce_loss + self.reg_weight * target_reg_loss
        losses = source_loss + target_loss

        return source_loss, target_loss, losses

    def predict(self, user, item, flag):
        if flag =="target":
            _, _, target_user_all_embeddings, target_item_all_embeddings = self.forward()
            u_embeddings = target_user_all_embeddings[user]
            i_embeddings = target_item_all_embeddings[item]
        else:
            source_user_all_embeddings, source_item_all_embeddings, _, _ = self.forward()
            u_embeddings = source_user_all_embeddings[user]
            i_embeddings = source_item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, user):
        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, self.target_restore_user_e, self.target_restore_item_e = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e

    




class EmbLoss(nn.Module):
    """
        EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
