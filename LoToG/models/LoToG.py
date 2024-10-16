# -*- coding:utf-8 -*-
"""
作者：86178
日期：2024年09月13日

"""

import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F


class LoToG(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size, max_len, use_dropout=False):

        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.rel_glo_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.temp_proto = 1

        self.drop = nn.Dropout()
        self.use_dropout = use_dropout
    def __dist__(self, x, y, dim):

        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):

        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def cosine_similarity(self, x1, x2):

        dot_product = self.__batch_dist__(x1, x2)

        norm_x1 = torch.norm(x1)
        norm_x2 = torch.norm(x2)

        similarity = dot_product / (norm_x1 * norm_x2)
        return similarity

    def forward(self, support, query, rel_text, N, K, total_Q, is_eval=None):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        # relation proto
        rel_gol, rel_loc = self.sentence_encoder(rel_text, cat=False)  # (B*N, D)   (B*N, L, D)

        rel_loc = torch.mean(rel_loc, 1)  # (B*N, D)

        rel_hyp = rel_gol + rel_loc         # (B*N, D)

        rel_proto = torch.cat((rel_hyp, rel_hyp), -1)  # (B*N, 2D)

        # rel_proto = torch.cat((rel_gol, rel_loc), -1)   #ablation
        rel_proto = rel_proto.view(-1, N, self.hidden_size * 2)  # (B, N, 2D)



        # entity proto
        support_h, support_t, _ = self.sentence_encoder(support)  # (B * N * K, D)
        query_h, query_t, _ = self.sentence_encoder(query)  # (B * total_Q, D)

        support = torch.cat((support_h, support_t), -1)
        query = torch.cat((query_h, query_t), -1)

        support = support.view(-1, N, K, self.hidden_size * 2)  # (B, N, K, 2D)
        query = query.view(-1, total_Q, self.hidden_size * 2)  # (B, total_Q, 2D)

        B = support.shape[0]  # batch_size

        support = torch.mean(support, 2)  # (B, N, 2D)


        # proto
        proto = support + rel_proto  # (B, N, 2D)

        logits = self.__batch_dist__(proto, query)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        logits_proto, labels_proto, sim_scalar = None, None, None

        if not is_eval:

            # relation-entity contrastive learning
            # # relation as global anchor
            rel_global_anchor = rel_proto.view(B * N, -1)  # (B * N, 2D)
            rel_text_anchor = rel_global_anchor.unsqueeze(1)  # (B * N, 1, 2D)


            # select positive entity prototypes
            proto_loc = support.view(B * N, -1)  # (B * N, 2D)
            pos_proto_loc = proto_loc.unsqueeze(1)  # (B * N, 1, 2D)

            # select negative entity prototypes
            neg_index = torch.zeros(B, N, N - 1)  # (B, N, N - 1)
            for b in range(B):
                for i in range(N):
                    index_ori = [i for i in range(b * N, (b + 1) * N)]
                    index_ori.pop(i)
                    neg_index[b, i] = torch.tensor(index_ori)

            neg_index = neg_index.long().view(-1).cuda()  # (B * N * (N - 1))
            neg_proto_loc = torch.index_select(proto_loc, dim=0, index=neg_index).view(B * N, N - 1, -1)

            # compute prototypical logits
            proto_loc_selected = torch.cat((pos_proto_loc, neg_proto_loc), dim=1)  # (B * N, N, 3D)
            logits_proto = self.__batch_dist__(proto_loc_selected, rel_text_anchor).squeeze(1)  # (B * N, N)
            logits_proto /= self.temp_proto  # scaling temperatures for the selected prototypes

            # targets
            labels_proto = torch.cat((torch.ones(B * N, 1), torch.zeros(B * N, N - 1)), dim=-1).cuda()  # (B * N, 2N)


            # local similarity scalar
            features_sim = torch.cat((proto_loc.view(B, N, -1), rel_global_anchor.view(B, N, -1)), dim=-1)
            features_sim = self.l2norm(features_sim)
            sim_task = torch.bmm(features_sim, torch.transpose(features_sim, 2, 1))  # (B, N, N)
            sim_scalar = torch.norm(sim_task, dim=(1, 2))  # (B)
            sim_scalar = torch.softmax(sim_scalar, dim=-1)
            sim_scalar = sim_scalar.repeat(total_Q, 1).t().reshape(-1)  # (B*totalQ)


        return logits, pred, logits_proto, labels_proto, sim_scalar


