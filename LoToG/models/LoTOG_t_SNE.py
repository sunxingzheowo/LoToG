# -*- coding:utf-8 -*-
"""
作者：86178
日期：2024年09月13日

"""

import sys

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F


class LoToG_t_SNE(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size, max_len, use_dropout=False):
        # 初始化 HCRP 模型
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.rel_glo_linear = nn.Linear(hidden_size, hidden_size * 2)  # 用于关系全局表示的线性层
        self.temp_proto = 1  # moco 0.07

        self.drop = nn.Dropout()
        self.use_dropout = use_dropout
    def __dist__(self, x, y, dim):
        # 计算指定维度上的点积
        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):
        # 计算支持集和查询集之间的批次点积
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def cosine_similarity(self, x1, x2):
        # 计算内积
        dot_product = self.__batch_dist__(x1, x2)
        # 计算向量的范数
        norm_x1 = torch.norm(x1)
        norm_x2 = torch.norm(x2)
        # 计算余弦相似度
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

        rel_gol, rel_loc = self.sentence_encoder(rel_text, cat=False)

        rel_loc = torch.mean(rel_loc, 1)  # [B*N, D]

        rel_hyp = rel_gol + rel_loc

        rel_proto = torch.cat((rel_hyp, rel_hyp), -1)  # [B*N, 2D]

        # rel_rep = torch.cat((rel_gol, rel_loc), -1)   #ablation

        support_h, support_t, _ = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query_h, query_t, _ = self.sentence_encoder(query)  # (B * total_Q, D)

        support = torch.cat((support_h, support_t), -1)
        query = torch.cat((query_h, query_t), -1)

        support = support.view(-1, N, K, self.hidden_size * 2)  # (B, N, K, 2D)
        query = query.view(-1, total_Q, self.hidden_size * 2)  # (B, total_Q, 2D)

        t_SNE_support = support.view(-1, N, K, self.hidden_size * 2)  # (B, N, K, 2D)
        support = t_SNE_support
        query = query.view(-1, total_Q, self.hidden_size * 2)  # (B, total_Q, 2D)

        B = support.shape[0]  # 批次大小

        support = torch.mean(support, 2)
        # support = self.__equalizer__(support, 2)

        ##
        rel_proto = rel_proto.view(-1, N, self.hidden_size * 2)

        proto = support + rel_proto

        t_SNE_rel_hyp = rel_hyp.view(-1, N, rel_gol.shape[1]).unsqueeze(2).expand(B, N, K, self.hidden_size)
        t_SNE_rel_rep = rel_proto.view(-1, N, rel_gol.shape[1] * 2).unsqueeze(2).expand(B, N, K, self.hidden_size*2)    #(B, N, K, 2D)
        # rel_rep = self.linear(rel_rep)
        pro_support = support + rel_proto
        t_SNE_pro_support = t_SNE_support + t_SNE_rel_rep
        # t_SNE_pro_support = t_SNE_support + t_SNE_rel_rep # (B, N, K, 2D)

        reshaped_data = t_SNE_pro_support.view(B, N * K, -1)[0]  # 取第一个维度
        print("调整后的数据形状:", reshaped_data.shape)  # 打印调整后的形状

        # 将 PyTorch 张量转换为 NumPy 数组，供 sklearn 使用
        reshaped_data_numpy = reshaped_data.cpu().numpy()
        # 使用 t-SNE 降维到二维
        perplexity_value = min(30, N * K - 1)  # 确保 perplexity 小于样本数量
        tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
        reduced_embeddings = tsne.fit_transform(reshaped_data_numpy)

        # 使用 PyTorch tensor 再次处理降维后的数据（可选）
        reduced_embeddings_tensor = torch.tensor(reduced_embeddings)

        # 设置全局字体大小
        plt.rcParams['font.size'] = 11  # 设置字体大小为 11pt

        # 创建可视化
        plt.figure(figsize=(5.50, 4.40))  # 设置图像大小

        # 根据类绘制点
        for i in range(N):
            if i == 0:
                label = 'P25:mother'
            elif i == 1:
                label = 'P40:child'
            elif i == 2:
                label = 'P26:spouse'
            else:
                label = f'Class {i}'

            plt.scatter(reduced_embeddings_tensor[i * K:(i + 1) * K, 0].numpy(),
                        reduced_embeddings_tensor[i * K:(i + 1) * K, 1].numpy(),
                        label=label, s=15)  # 设置点的大小，例如 s=50

        # 固定坐标轴范围
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)

        # 隐藏坐标轴的刻度和标签
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        # 设置外框为黑色线条
        for spine in plt.gca().spines.values():
            spine.set_color('black')  # 设置边框颜色为黑色
            spine.set_linewidth(1)     # 设置边框宽度

        # 调整图像，使绘图区域尽量填充整个图片
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # 设置图例，字体大小为 11pt
        plt.legend(fontsize=11)

        # 保存图片为PDF文件，指定DPI为400
        output_pdf_path = './t_SNE_image/10_1_proto_show_BERT.pdf'
        plt.savefig(output_pdf_path, format='pdf', dpi=400)

        # 显示图片
        plt.show()

        print(f"PDF图片已保存到: {output_pdf_path}")


        logits = self.__batch_dist__(proto, query)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        logits_proto, labels_proto, sim_scalar = None, None, None  # 初始化输出

        if not is_eval:

            # relation-entity contrastive learning
            # # relation as global anchor
            rel_global_anchor = rel_proto.view(B * N, -1)  # (B * N, 2D)
            rel_text_anchor = rel_global_anchor.unsqueeze(1)  # (B * N, 1, 2D)
            # select positive prototypes
            proto_loc = support.view(B * N, -1)  # (B * N, 2D)
            pos_proto_loc = proto_loc.unsqueeze(1)  # (B * N, 1, 2D)

            # select negative prototypes
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



