from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CapsLayerWithClusterRouting(nn.Module):
    def __init__(self, C_in, C_out, K, D_in, D_out, B, out_S, stride):
        super(CapsLayerWithClusterRouting, self).__init__()
        self.K = K
        self.C_in = C_in
        self.C_out = C_out
        self.D_in = D_in
        self.D_out = D_out
        self.conv_trans = nn.ModuleList()
        for i in range(self.C_out):
            self.conv_trans.append(nn.Conv2d(self.D_in, self.C_out * self.K * self.D_out, 3,
                                             stride=stride, padding=1, bias=B))
        self.acti = nn.LayerNorm([self.D_out, out_S, out_S])

    def cluster_routing(self, votes):
        batch_size, _, h, w = votes[0].shape
        for i in range(len(votes)):
            votes[i] = votes[i].view(batch_size, self.K, self.C_out, self.D_out, h, w)
        votes_for_next_layer = []
        for i in range(self.C_out):
            to_cat = [votes[j][:, :, i:(i+1), :, :, :] for j in range(self.C_in)]
            votes_for_channel_i = torch.cat(to_cat, dim=2)
            votes_for_next_layer.append(votes_for_channel_i)

        caps_of_next_layer = []
        for i in range(self.C_out):
            stds, means = torch.std_mean(votes_for_next_layer[i], dim=1, unbiased=False)
            agreement = -torch.log(stds)
            atts_for_c1 = F.softmax(agreement, dim=1)
            caps_of_channel_i = (atts_for_c1 * means).sum(dim=1)
            caps_of_next_layer.append(caps_of_channel_i)

        return caps_of_next_layer

    def forward(self, caps):
        votes = []
        for i in range(self.C_in):
            if isinstance(caps, list):  # not first layer
                votes.append(self.conv_trans[i](caps[i]))
            else: # first layer
                votes.append(self.conv_trans[i](caps))
        caps_of_next_layer = self.cluster_routing(votes)
        for i in range(self.C_out):
            caps_of_next_layer[i] = self.acti(caps_of_next_layer[i])
        return caps_of_next_layer


class CapsuleNetwork(nn.Module):
    def __init__(self, args):
        super(CapsuleNetwork, self).__init__()
        self.caps_layer1 = CapsLayerWithClusterRouting(args.C, args.C, args.K, args.input_img_dim, args.D, args.if_bias,
                                                       out_S=args.input_img_size, stride=1)
        self.caps_layer2 = CapsLayerWithClusterRouting(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 2), stride=2)
        self.caps_layer3 = CapsLayerWithClusterRouting(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 2), stride=1)
        self.caps_layer4 = CapsLayerWithClusterRouting(args.C, args.C, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 4), stride=2)
        self.caps_layer5 = CapsLayerWithClusterRouting(args.C, args.class_num, args.K, args.D, args.D, args.if_bias,
                                                       out_S=int(args.input_img_size / 4), stride=1)
        self.classifier = nn.Linear(args.D * int(args.input_img_size / 4) ** 2, 1)

    def forward(self, x):
        caps = self.caps_layer1(x)
        caps = self.caps_layer2(caps)
        caps = self.caps_layer3(caps)
        caps = self.caps_layer4(caps)
        caps = self.caps_layer5(caps)
        caps = [c.view(c.shape[0], -1).unsqueeze(1) for c in caps]
        caps = torch.cat(caps, dim=1)
        pred = self.classifier(caps).squeeze()
        return pred
