import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv, EGATConv, SAGEConv


class ObjSurrogate(nn.Module):
    def __init__(self, n_in=49, layers=[40, 20, 10, 10, 10], add_dropout=False) -> None:
        super().__init__()
        hidden_layers = list()

        n_hidden_prior = n_in
        for n_hidden in layers:
            hidden_layers += [
                nn.Linear(n_hidden_prior, n_hidden),
                nn.Dropout(p=0.2),
                nn.ReLU(),
            ]
            n_hidden_prior = n_hidden

        # output layer
        hidden_layers += [
            nn.Linear(n_hidden_prior, 1),
        ]

        self.net = nn.Sequential(*hidden_layers)

        if add_dropout == True:
            self.add_dropout()

    def add_dropout(self, p=0.05):
        self.net = torch.nn.Sequential(
            self.net[:1] +
            torch.nn.Sequential(*[torch.nn.Dropout(p=p),]) +
            self.net[1:3] +
            torch.nn.Sequential(*[torch.nn.Dropout(p=p),]) +
            self.net[3:]
        )

    def forward(self, q_liq_fun, bsw, gor, z_c, z_gl, q_gl_max):
        q_liq_fun_flat = q_liq_fun.flatten(1)

        # x = torch.hstack([C / 1e2, GL / 1e5, q_liq_fun_flat / 1e3, bsw.unsqueeze(1), gor.unsqueeze(1) / 1e2, z_c, z_gl])
        x = torch.hstack([z_c, z_gl, q_liq_fun_flat / 5e3, bsw.unsqueeze(1), gor.unsqueeze(1) / 1e3, (q_gl_max.unsqueeze(1) - 1e5) / 2e5])
        # x = torch.hstack([z_c, z_gl, bsw.unsqueeze(1), gor.unsqueeze(1) / 1e2])

        return self.net(x) * 2e3

class Fixer(nn.Module):
    def __init__(self, n_in=39, layers=[10, 10, 10]) -> None:
        super().__init__()
        hidden_layers = list()

        n_hidden_prior = n_in
        for n_hidden in layers:
            hidden_layers += [
                nn.Linear(n_hidden_prior, n_hidden),
                # nn.Dropout(p=0.2),
                nn.ReLU(),
            ]
            n_hidden_prior = n_hidden
        
        # output layer
        hidden_layers += [
            nn.Linear(n_hidden_prior, 10),
        ]

        self.net = nn.Sequential(*hidden_layers)

    def forward(self, q_liq_fun, bsw, gor, q_gl_max):
        q_liq_fun_flat = q_liq_fun.flatten(1)

        # x = torch.hstack([C / 1e2, GL / 1e5, q_liq_fun_flat / 1e3, bsw.unsqueeze(1), gor.unsqueeze(1) / 1e2, z_c, z_gl])
        x = torch.hstack([q_liq_fun_flat / 5e3, bsw.unsqueeze(1), gor.unsqueeze(1) / 1e3, (q_gl_max.unsqueeze(1) - 1e5) / 2e5])
        # x = torch.hstack([z_c, z_gl, bsw.unsqueeze(1), gor.unsqueeze(1) / 1e2])

        logits = self.net(x)

        return logits.unflatten(1, (2, 5))

class InstanceGCN(nn.Module):
    """Expects all features to be on the `x` data.
    """
    def __init__(self, n_z_feats, n_con_feats=1, n_x_feats=1, n_h_feats=10,
                 single_conv_for_both_passes=False, n_passes=1,
                 conv1='GraphConv', conv1_kwargs=dict(), conv2='GraphConv',
                 conv2_kwargs=dict(), conv3=None, conv3_kwargs=dict(),
                 readout_op='mean'):
        super().__init__()

        self.n_passes = n_passes
        self.n_h_feats = n_h_feats
        self.single_conv_for_both_passes = single_conv_for_both_passes

        self.n_z_feats = n_z_feats
        self.n_con_feats = n_con_feats
        self.n_x_feats = n_x_feats

        self.x_emb = torch.nn.Sequential(
            torch.nn.Linear(n_x_feats, n_h_feats),
            torch.nn.ReLU(),
        ).double()
        self.z_emb = torch.nn.Sequential(
            torch.nn.Linear(n_z_feats, n_h_feats),
            torch.nn.ReLU(),
        ).double()
        self.con_emb = torch.nn.Sequential(
            torch.nn.Linear(n_con_feats, n_h_feats),
            torch.nn.ReLU(),
        ).double()

        self.convs = list()

        if conv1 == 'GraphConv':
            c1_forward = GraphConv(n_h_feats, n_h_feats, **conv1_kwargs)
            # c1_backward = GraphConv(n_h_feats, n_h_feats, **conv1_kwargs)
        elif conv1 == 'EGATConv':
            c1_forward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                                  out_node_feats=n_h_feats, out_edge_feats=1,
                                  **conv1_kwargs)
            # c1_backward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
            #                        out_node_feats=n_h_feats, out_edge_feats=1,
            #                        **conv1_kwargs)
        elif conv1 == 'SAGEConv':
            c1_forward = SAGEConv(n_h_feats, n_h_feats, **conv1_kwargs)
            # c1_backward = SAGEConv(n_h_feats, n_h_feats, **conv1_kwargs)

        # if single_conv_for_both_passes:
        #     c1_backward = c1_forward

        self.convs.append(HeteroGraphConv({
            'z2c': c1_forward,
            'x2c': c1_forward,
            # 'c2c': c1_forward,
            # 'c2z': c1_backward,
            # 'c2x': c1_backward,
            # 'v2v': c1_backward,
            # 's2s': c1_backward,
        }).double())

        if conv2 is not None:
            if conv2 == 'GraphConv':
                c2_forward = GraphConv(n_h_feats, n_h_feats, **conv2_kwargs)
                # c2_backward = GraphConv(n_h_feats, n_h_feats, **conv2_kwargs)
            elif conv2 == 'EGATConv':
                c2_forward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                                    out_node_feats=n_h_feats, out_edge_feats=1,
                                    **conv2_kwargs)
                # c2_backward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                #                     out_node_feats=n_h_feats, out_edge_feats=1,
                #                     **conv2_kwargs)
            elif conv2 == 'SAGEConv':
                c2_forward = SAGEConv(n_h_feats, n_h_feats, **conv2_kwargs)
                # c2_backward = SAGEConv(n_h_feats, n_h_feats, **conv2_kwargs)

            # if single_conv_for_both_passes:
            #     c2_backward = c2_forward

            self.convs.append(HeteroGraphConv({
                'z2c': c2_forward,
                'x2c': c2_forward,
                # 'c2c': c2_forward,
                # 'c2v': c2_backward,
                # 'c2s': c2_backward,
                # 'v2v': c2_backward,
                # 's2s': c2_backward,
            }).double())

        if conv3 is not None:
            if conv3 == 'GraphConv':
                c3_forward = GraphConv(n_h_feats, n_h_feats, **conv3_kwargs)
                # c3_backward = GraphConv(n_h_feats, n_h_feats, **conv3_kwargs)
            elif conv3 == 'EGATConv':
                c3_forward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                                      out_node_feats=n_h_feats, out_edge_feats=1,
                                      **conv3_kwargs)
                # c3_backward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                #                        out_node_feats=n_h_feats, out_edge_feats=1,
                #                        **conv3_kwargs)
            elif conv3 == 'SAGEConv':
                c3_forward = SAGEConv(n_h_feats, n_h_feats, **conv3_kwargs)
                # c3_backward = SAGEConv(n_h_feats, n_h_feats, **conv3_kwargs)

            # if single_conv_for_both_passes:
            #     c3_backward = c3_forward

            self.convs.append(HeteroGraphConv({
                'z2c': c3_forward,
                'x2c': c3_forward,
                # 'c2c': c3_forward,
                # 'c2v': c3_backward,
                # 'c2s': c3_backward,
                # 'v2v': c3_backward,
                # 's2s': c3_backward,
            }).double())

        self.convs = nn.Sequential(*self.convs)

        self.output = torch.nn.Sequential(
            torch.nn.Linear(n_h_feats, n_h_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(n_h_feats, n_h_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(n_h_feats, 1),
        ).double()

        self.readout_op = readout_op

        # downscale all weights
        def downscale_weights(module):
            if isinstance(module, torch.nn.Linear):
                module.weight.data /= 10
        self.apply(downscale_weights)

    def forward(self, g, nfeats, eweights):
        z_features = nfeats['z'].view(-1,self.n_z_feats)
        x_features = nfeats['x'].view(-1,self.n_x_feats)
        con_features = nfeats['c'].view(-1,self.n_con_feats)

        # edges = ['v2c', 'c2v', 's2c', 'c2s']
        # edge_weights = dict()
        # for e in edges:
        #     edge_weights[e] = (g.edges[e].data['A'].unsqueeze(-1),)
        # edge_weights = g.edata['A']

        # embbed features
        h_z = self.z_emb(z_features)
        h_x = self.x_emb(x_features)
        h_con = self.con_emb(con_features)

        for _ in range(self.n_passes):
            # z -> con
            with g.local_scope():
                for conv in self.convs:
                    # TODO: figure out a way to avoid applying the convs to the
                    # whole graph, i.e., to ignore 'c2v' edges, for example, in
                    # this pass.
                    h_con = conv(g, {'c': h_con, 'z': h_z, 'x': h_x},
                                #  mod_args=edge_weights)['con']
                                #  mod_args={edge_weights_key: edge_weights})['con']
                                mod_kwargs={'edge_weights': eweights,
                                            'efeats': eweights})['c']
                    h_con = F.relu(h_con)

                # reverse the graph, keep the batch info
                graphs = dgl.unbatch(g)
                reversed_graphs = [dgl.reverse(g_) for g_ in graphs]
                rev_g = dgl.batch(reversed_graphs)

                # con -> z
                for conv in self.convs:
                    # edge_weights_key = 'edge_weights' if not isinstance(conv.mods.v2c, EGATConv) else 'efeats'
                    hs = conv(rev_g, {'c': h_con, 'z': h_z, 'x': h_x},
                            #   mod_args=edge_weights)
                            #   mod_args={edge_weights_key: edge_weights})
                            mod_kwargs={'edge_weights': eweights,
                                        'efeats': eweights})
                    h_z = F.relu(hs['z'])
                    h_x = F.relu(hs['x'])

        # per-node logits
        g.nodes['z'].data['logit'] = self.output(h_z)

        if self.readout_op is not None:
            return dgl.readout_nodes(g, 'logit', op=self.readout_op, ntype='z')
        else:
            return torch.stack([g_.nodes['z'].data['logit'] for g_ in dgl.unbatch(g)]).squeeze(-1)
