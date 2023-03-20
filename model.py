from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool  #### 从torch_geometric中直接调用一些操作
import torch
import torch.nn.functional as F
import torch.nn as nn

#-------------------------------------------------- 定义 HCL ----------------------------------------------------------#
class HCL(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, feat_dim, str_dim, args):
        super(HCL, self).__init__()
        '''
        hidden_dim: 隐藏层，embedding的dimension
        num_gc_layers: GIN_layer的层数
        feat_dim： node特征的dimension
        str_dim: structural encoding的dimension
        '''

        self.embedding_dim = hidden_dim * num_gc_layers                                                                     #### 5层最后拼接到一起了。 [ || || || || ]

        if args.GNN_Encoder == 'GCN':
            self.encoder_feat = Encoder_GCN(feat_dim, hidden_dim, num_gc_layers, args)                                                #### GIN，num_gc_layers层（5层）
            self.encoder_str = Encoder_GCN(str_dim, hidden_dim, num_gc_layers, args)                                                  #### GIN，num_gc_layers层（5层）
        elif args.GNN_Encoder == 'GIN':
            self.encoder_feat = Encoder_GIN(feat_dim, hidden_dim, num_gc_layers, args)                                                #### GIN，num_gc_layers层（5层）
            self.encoder_str = Encoder_GIN(str_dim, hidden_dim, num_gc_layers, args)                                                  #### GIN，num_gc_layers层（5层）         
        elif args.GNN_Encoder == 'GAT':
            self.encoder_feat = Encoder_GAT(feat_dim, hidden_dim, num_gc_layers, args)                                                #### GIN，num_gc_layers层（5层）
            self.encoder_str = Encoder_GAT(str_dim, hidden_dim, num_gc_layers, args)                                                  #### GIN，num_gc_layers层（5层）                 

        #### ---- 经过5层GIN后，映射到 graph level的空间
        self.proj_head_feat_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))

        self.proj_feat_g_Transformer = Transformer(self.embedding_dim, self.embedding_dim)              #### 过Transformer
        self.proj_str_g_Transformer = Transformer(self.embedding_dim, self.embedding_dim)               #### 过Transformer
        self.Cross_Attention_g = Cross_Attention(self.embedding_dim, self.embedding_dim)

        #### ---- 映射到 node level的空间
        self.proj_head_feat_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))
        
        self.proj_feat_n_Transformer = Transformer(self.embedding_dim, self.embedding_dim)
        self.proj_str_n_Transformer = Transformer(self.embedding_dim, self.embedding_dim)
        self.Cross_Attention_n = Cross_Attention(self.embedding_dim, self.embedding_dim)

        #### ---- 映射到 group level
        self.proj_head_b = nn.Sequential(nn.Linear(self.embedding_dim * 2, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))
        
        self.proj_b_CrossAttention = Transformer(self.embedding_dim * 2, self.embedding_dim)

        #### ---- 初始化上面的网络
        self.init_emb() 

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)        #### 初始化
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_b(self, x_f, x_s, edge_index, batch, num_graphs):
        g_f, _ = self.encoder_feat(x_f, edge_index, batch)                      #### GIN, num_gc_layers层（5层）
        g_s, _ = self.encoder_str(x_s, edge_index, batch)                       #### GIN，num_gc_layers层（5层）
        b = self.proj_head_b(torch.cat((g_f, g_s), 1))                          #### 映射到 group level，下一步进行 k-means聚类
        return b

    def forward(self, x_f, x_s, edge_index, batch, num_graphs):
        '''
        x_f: 传入的data.x, torch.Tensor, torch.Size([xxxx, 3])
        x_s: 传入的data.x_s,是structural encoding, torch.Tensor, torch.Size([xxxx, 32]) 32维是args里设置的
        edge_index: 传入的data.edge_index, torch.Tensor, torch.Size([2, xxxxxx])
        batch: 当前batch的个数，torch.Tensor, torch.Size([xxxx])
        num_graphs:, int, 128
        '''
        # print('x_f: ', type(x_f), x_f.shape, x_f)
        # print('x_s: ', type(x_s), x_s.shape, x_s)
        # print('edge_index: ', type(edge_index), edge_index.shape, edge_index)
        # print('batch: ', type(batch), batch.shape, batch)
        # print('num_graphs: ', type(num_graphs), num_graphs)

        g_f, n_f = self.encoder_feat(x_f, edge_index, batch)                    #### 数据集中的feature, 过GIN, num_gc_layers层（5层）
        g_s, n_s = self.encoder_str(x_s, edge_index, batch)                     #### structure view获取的feature, 过GIN, num_gc_layer（5层）

        # print('g_f: ', type(g_f), g_f.shape)                                  #### torch.Tensor, torch.size([128, 64])
        # print('g_s: ', type(g_s), g_s.shape)                                  #### torch.Tensor, torch.size([128, 64])

        #### 映射到 group level的特征空间
        # b = self.proj_head_b(torch.cat((g_f, g_s), 1))
        # b = self.proj_b_CrossAttention(torch.cat((g_f, g_s), 1))

        #### 映射到 graph level的特征空间
        g_f_1 = self.proj_head_feat_g(g_f)                                        
        g_s_1 = self.proj_head_str_g(g_s) 
        # g_f_1 = g_f
        # g_s_1 = g_s

        # g_f = self.proj_feat_g_Transformer(g_f)
        # g_s = self.proj_str_g_Transformer(g_s)
        g_f_2, g_s_2 = self.Cross_Attention_g(g_f, g_s)

        g_ff = g_f_1 + g_f_2
        g_ss = g_s_1 + g_s_2
        # print('g_f: ', type(g_f), g_f.shape)                                    #### torch.Tensor, torch.Size([128, 80])

        #### 映射到 node levl的特征空间
        n_f_1 = self.proj_head_feat_n(n_f)
        n_s_1 = self.proj_head_str_n(n_s)
        # n_f_1 = n_f
        # n_s_1 = n_s

        # n_f = self.proj_feat_n_Transformer(n_f)
        # n_s = self.proj_str_n_Transformer(n_s)
        n_f_2, n_s_2 = self.Cross_Attention_n(n_f, n_s)

        n_ff = n_f_1 + n_f_2
        n_ss = n_s_1 + n_s_2

        # return b, g_f, g_s, n_f, n_s                                            #### group level的； graph level的feature/structure, node level的feature/structure
        return g_f_2, g_s_2, n_f_2, n_s_2           ######## Cross-View Attention
        # return g_f_1, g_s_1, n_f_1, n_s_1           ######## 只MLP。消融实验用的
        # return g_ff, g_ss, n_ff, n_ss

    #-------------- 通过group level计算score ------------#
    @staticmethod
    def scoring_b(b, cluster_result, temperature = 0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']

        batch_size, _ = b.size()
        b_abs = b.norm(dim=1)
        prototypes_abs = prototypes.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)
        sim_matrix = torch.exp(sim_matrix / (temperature * density))

        v, id = torch.min(sim_matrix, 1)

        return v

    #---------- 计算group level的loss ------------#
    @staticmethod
    def calc_loss_b(b, index, cluster_result, temperature=0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']
        pos_proto_id = im2cluster[index].cpu().tolist()

        batch_size, _ = b.size()
        b_abs = b.norm(dim=1)
        prototypes_abs = prototypes.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)
        sim_matrix = torch.exp(sim_matrix / (temperature * density))
        pos_sim = sim_matrix[range(batch_size), pos_proto_id]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss + 1e-12)
        return loss

    @staticmethod
    def calc_loss_n(x, x_aug, batch, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        node_belonging_mask = batch.repeat(batch_size,1)
        node_belonging_mask = node_belonging_mask == node_belonging_mask.t()

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature) * node_belonging_mask
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = global_mean_pool(loss, batch)

        return loss

    @staticmethod
    def calc_loss_g(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss


######## 采用GIN进行卷积。
class Encoder_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, args):
        super(Encoder_GIN, self).__init__()
        '''
        num_features: 特征的dimension
        dim: hidden_dim, 隐藏层的dimension
        num_gc_layers: 层数
        '''

        self.num_gc_layers = num_gc_layers                  #### 层数，5层
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))             #### 从第1层开始，后面的层
            else:   
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))    #### 第0层
            conv = GINConv(nn)                                                          #### torch_geometric.nn.GINConv()
            self.convs.append(conv)
        
        self.Transformer = Transformer(num_features, num_features)                      #### 定义的Transformer
        self.pool_type = args.graph_level_pool


    def forward(self, x, edge_index, batch):
        '''
        x: torch.Tensor, torch.Size([xxxx, 特征的dimension])
        edge_index: 传入的data.edge_index, torch.Tensor, torch.Size([2, xxxxxx])
        batch: torch.Tensor, torch.Size([xxxx])
        '''
        xs = []
        # x = self.Transformer(x)                                                         #### 过Transformer
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.pool_type == 'global_max_pool':
            xpool = [global_max_pool(x, batch) for x in xs]                                 #### 每个graph，graph level的表示是把所有的node feature求和。对应论文中公式（4）
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)                                                      #### graph level的表征，node level的表征


######## 采用GCN进行卷积。和GIN做对比实验用的。
class Encoder_GCN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, args):
        super(Encoder_GCN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = GCNConv(dim, dim)
                # conv = GATConv(dim, dim)
            else:
                conv = GCNConv(num_features, dim)
                # conv = GATConv(num_features, dim)
            self.convs.append(conv)

        self.Transformer = Transformer(num_features, num_features)
        self.pool_type = args.graph_level_pool


    def forward(self, x, edge_index, batch):
        xs = []
        # x = self.Transformer(x)
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.pool_type == 'global_max_pool':
            xpool = [global_max_pool(x, batch) for x in xs]                                 #### 每个graph，graph level的表示是把所有的node feature求和。对应论文中公式（4）
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)


######## 采用GAT进行卷积。和GIN做对比实验用的。
class Encoder_GAT(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, args):
        super(Encoder_GAT, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = GATConv(dim, dim)
            else:
                conv = GATConv(num_features, dim)
            self.convs.append(conv)

        self.Transformer = Transformer(num_features, num_features)
        self.pool_type = args.graph_level_pool


    def forward(self, x, edge_index, batch):
        xs = []
        # x = self.Transformer(x)
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.pool_type == 'global_max_pool':
            xpool = [global_max_pool(x, batch) for x in xs]                                 #### 每个graph，graph level的表示是把所有的node feature求和。对应论文中公式（4）
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)
        

################################################ 添加的Transformer模型 ###############################################
class Transformer(nn.Module):
    def __init__(self, attributed_dim, n_h) -> None:
        super().__init__()
        '''
        attribute_dim: dataset的attribute维度
        n_h: 输出特征的维度，传入的是embedding_dim=64
        '''
        # concat feats and adj
        self.feats_channels = attributed_dim                                                    #### feat的通道数=dataset的attribute_dim
        self.attention_channels = attributed_dim * 2                                           #### 人为定义的超参数
        # self.attention_channels = 1024
        self.fc_cat = nn.Sequential(
            nn.Linear(self.feats_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels)
        )
        ################# attention
        self.w_qs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)     #### Q
        self.w_ks = nn.Linear(self.attention_channels, self.attention_channels, bias=False)     #### K
        self.w_vs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)     #### V
        self.layer_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)

        ################ FFN
        self.add_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.add_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.fc_ffn = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.fc2 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels // 2),
            nn.ReLU(),
            nn.Linear(self.attention_channels // 2, self.attention_channels)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, n_h)
        )

        # self.n_h = n_h                                                                          #### 最终输出特征的维度，embedding_dim=64

    def forward(self, features):
        '''
        features: 节点特征，3维，torch.Tensor, ([1, 结点总数，attribute_dim])
        adj: 邻接矩阵，3维，torch.Tensor, ([1, 结点总数，结点总数])
        '''
        # sum_adj = torch.sum(adj, 2).unsqueeze(-1)                       #### 计算每个结点的degree（indegree + outdegree）, torch.Tensor, ([1, 结点总数，1])
        # sum_adj = torch.softmax(sum_adj, 1)
        # cat_feat = torch.cat((features, sum_adj), 2)                    #### torch.Tensor, ([1, 结点总数， attribute_dim + 1])
        # cat_feat = self.fc_cat(cat_feat)

        features = features.unsqueeze(0)                                #### 从2维升成3维

        cat_feat = self.fc_cat(features)
        residual_feat = self.fc1(cat_feat)                              #### torch.Tensor, ([1, 结点总数, 1024])  
        # residual_feat = cat_feat

        Q = self.w_qs(cat_feat)                                         #### torch.Tensor, ([1, 结点总数, 1024])
        K = self.w_ks(cat_feat).permute(0, 2, 1)                        #### torch.Tensor, ([1, 1024, 结点总数])
        V = self.w_vs(cat_feat)                                         #### torch.Tensor, ([1, 结点总数, 1024])
        attn = Q @ K                                                    #### torch.Tensor, ([1, 结点总数, 结点总数])
        attn = torch.softmax(attn, 2)                                   #### torch.Tensor, ([1, 结点总数, 结点总数])
        atteneion = attn / (1e-9 + attn.sum(dim=1, keepdims=True))      #### torch.Tensor, ([1, 结点总数, 结点总数]) 
        sc = atteneion @ V                                              #### torch.Tensor, ([1, 结点总数, 1024])
        s = self.add_norm1(residual_feat + sc)                          #### torch.Tensor, ([1, 结点总数, 1024])
        ffn = self.add_norm2(s + self.fc_ffn(s))                        #### torch.Tensor, ([1, 结点总数, 1024])
                                                                        #### self.fc_ffn(s): torch.Tensor, ([1, 结点总数, 1024])
        ffn = s + self.fc_ffn(s)
        output = self.fc3(ffn)                                          #### torch.Tensor, ([1, 结点总数，embedding_dim=64])

        output = output.squeeze(0)                                      #### 再从3维降成2维
        return output



#################################################################### 添加的 Cross-View Attention 模型 #############################################################################
class Cross_Attention(nn.Module):
    def __init__(self, attributed_dim, n_h) -> None:
        super().__init__()
        '''
        attribute_dim: dataset的attribute维度
        n_h: 输出特征的维度，传入的是embedding_dim=64
        '''
        # concat feats and adj
        self.feats_channels = attributed_dim                                                    #### feat的通道数=dataset的attribute_dim
        self.attention_channels = attributed_dim                                          #### 人为定义的超参数， #### 如果要升维的话，最后就要再加一个MLP降维。但是降维的效果不好，output前的屏蔽掉了。所以这里就不升了。
        # self.attention_channels = 1024

        #------------------------------------------- view A --------------------------------------#
        self.A_projection_network = nn.Sequential(
            nn.Linear(self.feats_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels)
        )
        self.A_residual_block = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
        )
        ################# attention
        self.A_w_qs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)     #### Q
        self.A_w_ks = nn.Linear(self.attention_channels, self.attention_channels, bias=False)     #### K
        self.A_w_vs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)     #### V
        self.A_layer_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.A_layer_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)

        ################ FFN
        self.A_add_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.A_add_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.A_fc_ffn = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.A_fc2 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels // 2),
            nn.ReLU(),
            nn.Linear(self.attention_channels // 2, self.attention_channels)
        )
        self.A_fc3 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, n_h)
        )

        #------------------------------------------- view B --------------------------------------#
        self.B_projection_network = nn.Sequential(
            nn.Linear(self.feats_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels)
        )
        self.B_residual_block = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
        )
        ################# attention
        self.B_w_qs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)     #### Q
        self.B_w_ks = nn.Linear(self.attention_channels, self.attention_channels, bias=False)     #### K
        self.B_w_vs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)     #### V
        self.B_layer_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.B_layer_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)

        ################ FFN
        self.B_add_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.B_add_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.B_fc_ffn = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.B_fc2 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels // 2),
            nn.ReLU(),
            nn.Linear(self.attention_channels // 2, self.attention_channels)
        )
        self.B_fc3 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, n_h)
        )



    def forward(self, feat_a, feat_b):
        '''
        feat_a: 节点特征，3维，torch.Tensor, ([1, 结点总数，attribute_dim])
        feat_b: 同feat_a
        adj: 邻接矩阵，3维，torch.Tensor, ([1, 结点总数，结点总数])
        '''
        # sum_adj = torch.sum(adj, 2).unsqueeze(-1)                       #### 计算每个结点的degree（indegree + outdegree）, torch.Tensor, ([1, 结点总数，1])
        # sum_adj = torch.softmax(sum_adj, 1)
        # cat_feat = torch.cat((features, sum_adj), 2)                    #### torch.Tensor, ([1, 结点总数， attribute_dim + 1])
        # cat_feat = self.fc_cat(cat_feat)

        # print('feat_a: ', type(feat_a), feat_a.shape)                                  #### torch.Tensor, torch.size([128, 64])
        # print('feat_b: ', type(feat_b), feat_b.shape)                                  #### torch.Tensor, torch.size([128, 64])

        #----------------------------------------#
        feat_a = feat_a.unsqueeze(0)                                        #### 从2维升成3维
        feat_b = feat_b.unsqueeze(0)

        #----------------------------------------#
        A_feat = self.A_projection_network(feat_a)
        A_residual_feat = self.A_residual_block(A_feat)                                #### torch.Tensor, ([1, 结点总数, 1024])  
        
        B_feat = self.B_projection_network(feat_b)
        B_residual_feat = self.B_residual_block(B_feat)

        # #----------------------------------------#
        A_Q = self.A_w_qs(A_feat)                                           #### torch.Tensor, ([1, 结点总数, 1024])
        A_K = self.A_w_ks(A_feat).permute(0, 2, 1)                          #### torch.Tensor, ([1, 1024, 结点总数])
        A_V = self.A_w_vs(A_feat)                                           #### torch.Tensor, ([1, 结点总数, 1024])

        B_Q = self.B_w_qs(B_feat)                                       
        B_K = self.B_w_ks(B_feat).permute(0, 2, 1)                        
        B_V = self.B_w_vs(B_feat)                                          

        # #--------------- Cross-View Attention ------------------#
        A_attn = A_Q @ B_K                                                  #### torch.Tensor, ([1, 结点总数, 结点总数])
        B_attn = B_Q @ A_K
        # A_Q, B_Q = B_Q, A_Q
        
        #------------------ Self-Attention ---------------------#
        # A_attn = A_Q @ A_K
        # B_attn = B_Q @ B_K

        #----------------------------------------#
        A_attn = torch.softmax(A_attn, 2)                                   #### torch.Tensor, ([1, 结点总数, 结点总数])
        A_atteneion = A_attn / (1e-9 + A_attn.sum(dim=1, keepdims=True))    #### torch.Tensor, ([1, 结点总数, 结点总数]) 
        A_sc = A_atteneion @ A_V
        # A_sc = A_atteneion @ B_V                                             #### torch.Tensor, ([1, 结点总数, 1024])
        # A_sc = A_sc.permute(0, 2, 1)

        B_attn = torch.softmax(B_attn, 2)                                  
        B_atteneion = B_attn / (1e-9 + B_attn.sum(dim=1, keepdims=True))      
        B_sc = B_atteneion @ B_V

        #----------------------------------------#
        A_s = self.A_add_norm1(A_residual_feat + A_sc)                      #### torch.Tensor, ([1, 结点总数, 1024])
        # A_ffn = self.A_add_norm2(A_s + self.A_fc_ffn(A_s))                  #### torch.Tensor, ([1, 结点总数, 1024])
        A_ffn = A_s + self.A_fc_ffn(A_s)
        # A_ffn = A_s
                                                                            #### self.fc_ffn(s): torch.Tensor, ([1, 结点总数, 1024])
        B_s = self.B_add_norm1(B_residual_feat + B_sc)                         
        # B_ffn = self.B_add_norm2(B_s + self.B_fc_ffn(B_s))                        
        B_ffn = B_s + self.B_fc_ffn(B_s)
        # B_ffn = B_s

        #---------------------------------------#

        # A_output = self.A_fc3(A_ffn)                                      #### torch.Tensor, ([1, 结点总数，embedding_dim=64])
        A_output = A_ffn

        # B_output = self.B_fc3(B_ffn)        
        B_output = B_ffn      

        #---------------------------------------#
        A_output = A_output.squeeze(0)                                      #### 再从3维降成2维
        B_output = B_output.squeeze(0)
        return A_output, B_output


