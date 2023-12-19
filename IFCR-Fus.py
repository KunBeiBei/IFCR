import dgl
import tqdm
import torch
import numpy as np
import torch.nn as nn
from config import parse_args

from assessment import Recall, NDCG
from dgl.nn.pytorch import GraphConv

args = parse_args()

class GCN(nn.Module):
    def __init__(self, 
        num_user, num_item, num_buy, num_cart, num_pv, emb_dim, # nodes and edges
        MTL_w, # MTL weights
        l # number of layers
    ):
        super(GCN, self).__init__()
        # nodes embeddding
        self.user_emb = nn.Embedding(num_user, emb_dim)
        self.item_emb = nn.Embedding(num_item, emb_dim)
        # edges embeddding
        self.buy_edges_emb = nn.Embedding(num_buy, emb_dim)
        self.cart_edges_emb = nn.Embedding(num_cart, emb_dim)
        self.pv_edges_emb = nn.Embedding(num_pv, emb_dim)


        # nodes weight matrix
        # edges weight matrix
        self.node_weights = nn.ParameterDict()
        self.edge_weights = nn.ParameterDict()

        self.buy_layer  = nn.ModuleDict()
        self.cart_layer = nn.ModuleDict()
        self.pv_layer   = nn.ModuleDict()

        self.l = l

        for i in range(l):
            self.node_weights[f'weight_{i}'] = nn.Parameter(torch.rand(emb_dim, emb_dim))
            self.edge_weights[f'weight_{i}'] = nn.Parameter(torch.rand(emb_dim, emb_dim))

            self.buy_layer[f'user_conv_{i}'] = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False, allow_zero_in_degree=True)
            self.buy_layer[f'item_conv_{i}'] = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False, allow_zero_in_degree=True)

            self.cart_layer[f'user_conv_{i}'] = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False, allow_zero_in_degree=True)
            self.cart_layer[f'item_conv_{i}'] = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False, allow_zero_in_degree=True)

            self.pv_layer[f'user_conv_{i}'] = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False, allow_zero_in_degree=True)
            self.pv_layer[f'item_conv_{i}'] = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False, allow_zero_in_degree=True)


        # # fusion layer
        # self.buy_conv5_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), norm='right', bias=False, weight=False)
        # self.buy_conv5_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), norm='right', bias=False, weight=False)
        # self.cart_conv5_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), norm='right', bias=False, weight=False)
        # self.cart_conv5_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), norm='right', bias=False, weight=False)
        # self.pv_conv5_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), norm='right', bias=False, weight=False)
        # self.pv_conv5_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), norm='right', bias=False, weight=False)

        # # fusion weight matrix
        # self.fus_weight = nn.Parameter(torch.rand(emb_dim, emb_dim))

        # prediction layer
        self.buy_prediction = nn.Parameter(torch.rand(emb_dim))
        self.cart_prediction = nn.Parameter(torch.rand(emb_dim))
        self.pv_prediction = nn.Parameter(torch.rand(emb_dim))

        # active fun
        self.active = nn.LeakyReLU()
        
        self.w_pv = MTL_w['pv'] # 0/6
        self.w_cart = MTL_w['cart'] # 5/6
        self.w_buy = MTL_w['buy'] # 1/6

        # self.decay = 1e-2

    def update_edge(self, g, edge, weight):
        u = g.edges()[0]
        v = g.edges()[1]

        src_edge = torch.zeros(g.num_nodes('user'), edge.shape[1], device=edge.device)
        dst_edge = torch.zeros(g.num_nodes('item'), edge.shape[1], device=edge.device)

        src_degs, dst_degs = g.out_degrees().float().clamp(min=1), g.in_degrees().float().clamp(min=1)

        src_edge = src_edge.index_add(0, u, edge)
        dst_edge = dst_edge.index_add(0, v, edge)

        temp = ((src_edge[u] + dst_edge[v]).T / (src_degs[u] + dst_degs[v])).T

        return self.active(torch.matmul(temp, weight))
            
    def forward(self, g):
        # Beibei
        # a = 0.04
        # b = 0.04
        # Taobao
        a = 0.0045
        b = 0.0045
        src_feature = self.user_emb(g['buy'].nodes('user')) * a
        dst_feature = self.item_emb(g['buy'].nodes('item')) * a
        
        edge_buy  = self.buy_edges_emb(torch.arange(g['buy'].num_edges(), device=g.device)) * b
        edge_cart = self.cart_edges_emb(torch.arange(g['cart'].num_edges(), device=g.device)) * b
        edge_pv   = self.pv_edges_emb(torch.arange(g['pv'].num_edges(), device=g.device)) * b

        src_feature_all = src_feature
        dst_feature_all = dst_feature
        # edge_buy_all = edge_buy
        # edge_cart_all = edge_cart
        # edge_pv_all = edge_pv

        for i in range(self.l):
            buy_dst = self.buy_layer[f'item_conv_{i}'](g['buy'], (src_feature, dst_feature), weight=self.node_weights[f'weight_{i}'], edge_weight=edge_buy)
            buy_src = self.buy_layer[f'user_conv_{i}'](g['rev_buy'], (dst_feature, src_feature), weight=self.node_weights[f'weight_{i}'], edge_weight=edge_buy)

            cart_dst = self.cart_layer[f'item_conv_{i}'](g['cart'], (src_feature, dst_feature), weight=self.node_weights[f'weight_{i}'], edge_weight=edge_cart)
            cart_src = self.cart_layer[f'user_conv_{i}'](g['rev_cart'], (dst_feature, src_feature), weight=self.node_weights[f'weight_{i}'], edge_weight=edge_cart)

            pv_dst = self.pv_layer[f'item_conv_{i}'](g['pv'], (src_feature, dst_feature), weight=self.node_weights[f'weight_{i}'], edge_weight=edge_pv)
            pv_src = self.pv_layer[f'user_conv_{i}'](g['rev_pv'], (dst_feature, src_feature), weight=self.node_weights[f'weight_{i}'], edge_weight=edge_pv)

            dst_feature = self.w_buy*buy_dst + self.w_cart*cart_dst + self.w_pv*pv_dst
            src_feature = self.w_buy*buy_src + self.w_cart*cart_src + self.w_pv*pv_src

            edge_buy  = self.update_edge(g['buy'], edge_buy, self.edge_weights[f'weight_{i}'])
            edge_cart = self.update_edge(g['cart'], edge_cart, self.edge_weights[f'weight_{i}'])
            edge_pv   = self.update_edge(g['pv'], edge_pv, self.edge_weights[f'weight_{i}'])

            src_feature_all = src_feature_all + src_feature
            dst_feature_all = dst_feature_all + dst_feature
            # edge_buy_all = edge_buy_all + edge_buy
            # edge_cart_all = edge_cart_all + edge_cart
            # edge_pv_all = edge_pv_all + edge_pv


        src_feature_all = src_feature_all / (self.l+1)
        dst_feature_all = dst_feature_all / (self.l+1)
        # edge_buy_all = edge_buy_all / (self.l+1)
        # edge_cart_all = edge_cart_all / (self.l+1)
        # edge_pv_all = edge_pv_all / (self.l+1)


        # src = torch.ones(src_feature_all.shape, device=g.device)
        # dst = torch.ones(dst_feature_all.shape, device=g.device)

        # buy_dst_feature_5 = self.buy_conv5_item(g['buy'], (src, dst), weight=self.fus_weight, edge_weight=edge_buy_all)
        # buy_src_feature_5 = self.buy_conv5_user(g['rev_buy'], (dst, src), weight=self.fus_weight, edge_weight=edge_buy_all)
        
        # cart_dst_feature_5 = self.cart_conv5_item(g['cart'], (src, dst), weight=self.fus_weight, edge_weight=edge_cart_all)
        # cart_src_feature_5 = self.cart_conv5_user(g['rev_cart'], (dst, src), weight=self.fus_weight, edge_weight=edge_cart_all)
        
        # pv_dst_feature_5 = self.pv_conv5_item(g['pv'], (src, dst), weight=self.fus_weight, edge_weight=edge_pv_all)
        # pv_src_feature_5 = self.pv_conv5_user(g['rev_pv'], (dst, src), weight=self.fus_weight, edge_weight=edge_pv_all)

        # dst_feature_5 = dst_feature_all * (self.w_buy*buy_dst_feature_5 + self.w_cart*cart_dst_feature_5 + self.w_pv*pv_dst_feature_5)
        # src_feature_5 = src_feature_all * (self.w_buy*buy_src_feature_5 + self.w_cart*cart_src_feature_5 + self.w_pv*pv_src_feature_5)

        return src_feature_all, dst_feature_all
        return src_feature_5, dst_feature_5

def test(g, model, R_buy, R_test, k, device):

    metrics = []

    for i in k:
        metrics.append(Recall(i))
    for i in k:
        metrics.append(NDCG(i))

    with torch.no_grad():

        model.eval()
        model = model.to(device)
        g = g.to(device)

        src_feature, dst_feature = model(g)
        edge_buy = model.buy_prediction

        ys = []
        for src in src_feature:
            # 把一个用户复制物品的数量
            src = src.repeat(dst_feature.shape[0], 1)
            y = torch.einsum('ij,j->i', src * dst_feature, edge_buy)
            ys.append(y.view(-1))
        ys = torch.stack(ys)

        ys = ys.to(torch.device('cpu'))
        R_buy = R_buy.to(torch.device('cpu'))
        R_test = R_test.to(torch.device('cpu'))

        user_item = ys - R_buy * 100000000

        arr = []
        for metric in metrics:
            metric.start()
            metric(user_item, R_test)
            metric.stop()
            arr.append(round(metric.metric, 4))
            print('test:{}:{}'.format(metric.get_title(), round(metric.metric, 4)), end='\t')
        print()
        return arr


if __name__ == '__main__':
    SEED = 2020
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Data loading...")

    dataset = args.dataset

    # embed size
    d = args.embed_size

    c0 = args.TaoBao_c0
    c1 = args.TaoBao_i
    MTL_w = eval(args.MTL_TaoBao_w)
    decay = args.TaoBao_decay
    if dataset == 'Beibei':
        c0 = args.BeiBei_c0
        c1 = args.BeiBei_i
        MTL_w = eval(args.MTL_BeiBei_w)
        decay = args.BeiBei_decay

    traNum1 = 0
    traNum2 = args.epoch
    # directory = 'result/GHCF'
    k = eval(args.Ks)
    l = args.layer_size
    lr = args.lr

    R_buy = torch.load(dataset+'/a/R_buy.pth')
    R_test = torch.load(dataset+'/a/R_test.pth')
    # R_pv = torch.load(dataset+'/a/R_pv.pth')
    # R_cart = torch.load(dataset+'/a/R_cart.pth')

    R_buy = torch.tensor(R_buy.toarray())
    R_test = torch.tensor(R_test.toarray())

    r_buy_user = torch.load(dataset+'/a/r_buy_user.pth')
    r_buy_item = torch.load(dataset+'/a/r_buy_item.pth')
    r_pv_user = torch.load(dataset+'/a/r_pv_user.pth')
    r_pv_item = torch.load(dataset+'/a/r_pv_item.pth')
    r_cart_user = torch.load(dataset+'/a/r_cart_user.pth')
    r_cart_item = torch.load(dataset+'/a/r_cart_item.pth')

    g = dgl.heterograph({
                ('user','buy','item'): (r_buy_user, r_buy_item),
                ('user','cart','item'): (r_cart_user, r_cart_item),
                ('user','pv','item'): (r_pv_user, r_pv_item),
                ('item','rev_buy','user'): (r_buy_item, r_buy_user),
                ('item','rev_cart','user'): (r_cart_item, r_cart_user),
                ('item','rev_pv','user'): (r_pv_item, r_pv_user)
            })
            
    # pv
    item_de_pv = g['pv'].in_degrees().float()
    item_de_pv[item_de_pv == 0] = item_de_pv[item_de_pv == 0] + 1
    pop_pv = item_de_pv/torch.sum(item_de_pv)
    # buy
    item_de_buy = g['buy'].in_degrees().float() + item_de_pv
    pop_buy = item_de_buy/torch.sum(item_de_buy)
    # cart
    item_de_cart = g['cart'].in_degrees().float() + item_de_pv
    pop_cart = item_de_cart/torch.sum(item_de_cart)

    Cv_f_buy = c0 * torch.pow(pop_buy, c1)/torch.sum(torch.pow(pop_buy, c1))
    Cv_f_cart = c0 * torch.pow(pop_cart, c1)/torch.sum(torch.pow(pop_cart, c1))
    Cv_f_pv = c0 * torch.pow(pop_pv, c1)/torch.sum(torch.pow(pop_pv, c1))


    print("data Loading completed")


    model = GCN(g['buy'].num_nodes('user'), g['buy'].num_nodes('item'), g['buy'].num_edges(), g['cart'].num_edges(), g['pv'].num_edges(), d, MTL_w, l)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    recall10 = []
    recall50 = []
    recall100= []
    ndcg10 = []
    ndcg50 = []
    ndcg100= []
    losss = []

    # train & test
    for epoch in tqdm.tqdm(range(traNum1, traNum2)):
        model.train()

        g = g.to(device)

        Cv_f_buy = Cv_f_buy.to(device)
        Cv_f_cart = Cv_f_cart.to(device)
        Cv_f_pv = Cv_f_pv.to(device)

        model = model.to(device)
        optimizer.zero_grad()

        src_feature, dst_feature = model(g)

        u = torch.einsum('ab,ac->bc', src_feature, src_feature)
        buy_v = torch.einsum('ab,ac->bc', (dst_feature.T * Cv_f_buy).T, dst_feature)
        cart_v = torch.einsum('ab,ac->bc', (dst_feature.T * Cv_f_cart).T, dst_feature)
        pv_v = torch.einsum('ab,ac->bc', (dst_feature.T * Cv_f_pv).T, dst_feature)
        
        buy_e = torch.einsum('b,c->bc',  model.buy_prediction, model.buy_prediction)
        cart_e = torch.einsum('b,c->bc', model.cart_prediction, model.cart_prediction)
        pv_e = torch.einsum('b,c->bc', model.pv_prediction, model.pv_prediction)

        buy1 = src_feature[g['buy'].edges()[0]]
        buy2 = dst_feature[g['buy'].edges()[1]]
        cart1 = src_feature[g['cart'].edges()[0]]
        cart2 = dst_feature[g['cart'].edges()[1]]
        pv1 = src_feature[g['pv'].edges()[0]]
        pv2 = dst_feature[g['pv'].edges()[1]]

        buy_y = torch.einsum('ij,j->i', buy1 * buy2, model.buy_prediction)
        cart_y = torch.einsum('ij,j->i', cart1 * cart2, model.cart_prediction)
        pv_y = torch.einsum('ij,j->i', pv1 * pv2, model.pv_prediction)

        buy_temp = u * buy_v * buy_e
        cart_temp = u * cart_v * cart_e
        pv_temp = u * pv_v * pv_e

        loss1 = torch.sum(buy_temp) + torch.sum((1 - Cv_f_buy[g['buy'].edges()[1]]) * torch.pow(buy_y, 2) - 2 * buy_y)
        loss2 = torch.sum(cart_temp) + torch.sum((1 - Cv_f_cart[g['cart'].edges()[1]]) * torch.pow(cart_y, 2) - 2 * cart_y)
        loss3 = torch.sum(pv_temp) + torch.sum((1 - Cv_f_pv[g['pv'].edges()[1]]) * torch.pow(pv_y, 2) - 2 * pv_y)

        regularizer = torch.norm(src_feature) ** 2/2 + torch.norm(dst_feature) ** 2/2
        l2_loss = regularizer * decay

        loss = loss1*model.w_buy + loss2*model.w_cart + loss3*model.w_pv + l2_loss

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        losss.append(float(loss))

        if (epoch+1)%10 == 0:
            print(float(loss),float(regularizer))
            arr = test(g, model, R_buy, R_test, k, torch.device('cuda'))
            recall10.append(arr[0])
            recall50.append(arr[1])
            recall100.append(arr[2])
            ndcg10.append(arr[3])
            ndcg50.append(arr[4])
            ndcg100.append(arr[5])

    torch.save(recall10,  'jeg/T/M-Fus/recall10.pth')
    torch.save(recall50,  'jeg/T/M-Fus/recall50.pth')
    torch.save(recall100, 'jeg/T/M-Fus/recall100.pth')
    torch.save(ndcg10,    'jeg/T/M-Fus/ndcg10.pth')
    torch.save(ndcg50,    'jeg/T/M-Fus/ndcg50.pth')
    torch.save(ndcg100,   'jeg/T/M-Fus/ndcg100.pth')
    torch.save(losss,     'jeg/T/M-Fus/loss.pth')
