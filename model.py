from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model as HFWav2Vec2Model, AutoModel


class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        self.device = device
        self.model = HFWav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.model_mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.out_dim = 2048

    def extract_feat(self, input_data, input_data2):
        input_tmp  = input_data[:, :, 0]  if input_data.ndim  == 3 else input_data
        input_tmp2 = input_data2[:, :, 0] if input_data2.ndim == 3 else input_data2

        if next(self.model.parameters()).device != input_tmp.device:
            self.model.to(input_tmp.device)
        if next(self.model_mert.parameters()).device != input_tmp2.device:
            self.model_mert.to(input_tmp2.device)

        self.model.train()
        self.model_mert.train()

        emb      = self.model(input_tmp.float()).last_hidden_state
        emb_mert = self.model_mert(input_tmp2.float(), output_hidden_states=True).last_hidden_state

        min_len  = min(emb.size(1), emb_mert.size(1))
        emb      = emb[:, :min_len, :]
        emb_mert = emb_mert[:, :min_len, :]
        return torch.cat((emb, emb_mert), dim=2)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj       = nn.Linear(in_dim, out_dim)
        self.att_weight     = self._init_new_params(out_dim, 1)
        self.proj_with_att  = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn         = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act        = nn.SELU(inplace=True)
        self.temp       = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)
        org = x.size(); x = self.bn(x.view(-1, org[-1])).view(org)
        return self.act(x)

    def _pairwise_mul_nodes(self, x):
        nb = x.size(1)
        return x.unsqueeze(2).expand(-1, -1, nb, -1) * x.transpose(1, 2).unsqueeze(1).expand(-1, nb, -1, -1)

    def _derive_att_map(self, x):
        a = torch.tanh(self.att_proj(self._pairwise_mul_nodes(x)))
        return F.softmax(torch.matmul(a, self.att_weight) / self.temp, dim=-2)

    def _init_new_params(self, *size):
        p = nn.Parameter(torch.FloatTensor(*size)); nn.init.xavier_normal_(p); return p


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj   = nn.Linear(in_dim, out_dim)
        self.att_projM  = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM  = self._init_new_params(out_dim, 1)
        self.proj_with_att    = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM    = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn         = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act        = nn.SELU(inplace=True)
        self.temp       = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        n1 = x1.size(1)
        x  = torch.cat([self.proj_type1(x1), self.proj_type2(x2)], dim=1)
        if master is None:
            master = x.mean(dim=1, keepdim=True)
        x      = self.input_drop(x)
        att    = self._derive_att_map(x, n1, x2.size(1))
        master = self._update_master(x, master)
        x      = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        org    = x.size(); x = self.bn(x.view(-1, org[-1])).view(org)
        x      = self.act(x)
        return x.narrow(1, 0, n1), x.narrow(1, n1, x2.size(1)), master

    def _update_master(self, x, master):
        a = torch.tanh(self.att_projM(x * master))
        a = F.softmax(torch.matmul(a, self.att_weightM) / self.temp, dim=-2)
        return self.proj_with_attM(torch.matmul(a.squeeze(-1).unsqueeze(1), x)) + self.proj_without_attM(master)

    def _derive_att_map(self, x, n1, n2):
        nb  = x.size(1)
        pm  = x.unsqueeze(2).expand(-1,-1,nb,-1) * x.transpose(1,2).unsqueeze(1).expand(-1,nb,-1,-1)
        att = torch.tanh(self.att_proj(pm))
        b   = torch.zeros_like(att[:,:,:,0]).unsqueeze(-1)
        b[:, :n1,  :n1,  :] = torch.matmul(att[:, :n1,  :n1,  :], self.att_weight11)
        b[:, n1:,  n1:,  :] = torch.matmul(att[:, n1:,  n1:,  :], self.att_weight22)
        b[:, :n1,  n1:,  :] = torch.matmul(att[:, :n1,  n1:,  :], self.att_weight12)
        b[:, n1:,  :n1,  :] = torch.matmul(att[:, n1:,  :n1,  :], self.att_weight12)
        return F.softmax(b / self.temp, dim=-2)

    def _init_new_params(self, *size):
        p = nn.Parameter(torch.FloatTensor(*size)); nn.init.xavier_normal_(p); return p


class GraphPool(nn.Module):
    def __init__(self, k, in_dim, p):
        super().__init__()
        self.k    = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.sig  = nn.Sigmoid()

    def forward(self, h):
        scores = self.sig(self.proj(self.drop(h)))
        n      = max(int(h.size(1) * self.k), 1)
        _, idx = torch.topk(scores, n, dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1, -1, h.size(2)))


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], (2,3), padding=(1,1))
        self.selu  = nn.SELU(inplace=True)
        self.bn2   = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], (2,3), padding=(0,1))
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], (1,3), padding=(0,1))

    def forward(self, x):
        identity = x
        out = x if self.first else self.selu(self.bn1(x))
        out = self.selu(self.bn2(self.conv1(x)))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return out + identity


class Wav2Vec2Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        filts        = [128, [1,32], [32,32], [32,64], [64,64]]
        gat_dims     = [64, 32]
        pool_ratios  = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.ssl_model = SSLModel(self.device)
        self.LL        = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn  = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm2d(64)
        self.drop      = nn.Dropout(0.5, inplace=True)
        self.drop_way  = nn.Dropout(0.2, inplace=True)
        self.selu      = nn.SELU(inplace=True)

        
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64,128,(1,1)), nn.SELU(inplace=True),
            nn.BatchNorm2d(128),    nn.Conv2d(128,64,(1,1)),
        )

        self.pos_S   = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S   = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T   = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x, x2):
        x = self.LL(self.ssl_model.extract_feat(x.squeeze(-1), x2.squeeze(-1)))
        x = self.selu(self.first_bn(F.max_pool2d(x.transpose(1,2).unsqueeze(1), (3,3))))
        x = self.selu(self.first_bn1(self.encoder(x)))
        w = self.attention(x)

        # spectral branch
        e_S_raw = torch.sum(x * F.softmax(w, dim=-1), dim=-1).transpose(1,2)
        pos_S = self.pos_S
        if e_S_raw.size(1) != pos_S.size(1):
            pos_S = F.interpolate(
                pos_S.transpose(1,2), size=e_S_raw.size(1), mode='linear', align_corners=False
            ).transpose(1,2)
        e_S = e_S_raw + pos_S

        out_S = self.pool_S(self.GAT_layer_S(e_S))

        # temporal branch
        e_T   = torch.sum(x * F.softmax(w, dim=-2), dim=-2).transpose(1,2)
        out_T = self.pool_T(self.GAT_layer_T(e_T))

        # inference 1
        out_T1, out_S1, m1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1); out_T1 = self.pool_hT1(out_T1)
        dT, dS, dm = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=m1)
        out_T1 += dT; out_S1 += dS; m1 += dm

        # inference 2
        out_T2, out_S2, m2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2); out_T2 = self.pool_hT2(out_T2)
        dT, dS, dm = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=m2)
        out_T2 += dT; out_S2 += dS; m2 += dm

        out_T1,out_T2 = self.drop_way(out_T1), self.drop_way(out_T2)
        out_S1,out_S2 = self.drop_way(out_S1), self.drop_way(out_S2)
        m1, m2        = self.drop_way(m1),     self.drop_way(m2)

        out_T  = torch.max(out_T1, out_T2)
        out_S  = torch.max(out_S1, out_S2)
        master = torch.max(m1, m2)

        T_max,_ = torch.max(torch.abs(out_T), dim=1)
        T_avg   = out_T.mean(dim=1)
        S_max,_ = torch.max(torch.abs(out_S), dim=1)
        S_avg   = out_S.mean(dim=1)

        h = self.drop(torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1))
        return self.out_layer(h)
