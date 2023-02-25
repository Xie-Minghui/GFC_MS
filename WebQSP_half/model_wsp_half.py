import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import math
from mindspore import Tensor, SparseTensor
import mindspore_hub as mshub
from mindspore import context


class GFC(nn.Cell):
    def __init__(self, args, ent2id, rel2id, triples):
        super().__init__()
        self.args = args
        self.num_steps = 2
        num_relations = len(rel2id)

        Tsize = len(triples)
        Esize = len(ent2id)
        idx = mindspore.Tensor([i for i in range(Tsize)])
        self.Msubj = SparseTensor(
            mindspore.ops.Stack((idx, triples[:,0])), mindspore.Tensor([1] * Tsize, mindspore.float32), P.size([Tsize, Esize]))
        self.Mobj = SparseTensor(
            mindspore.ops.Stack((idx, triples[:,2])), mindspore.Tensor([1] * Tsize, mindspore.float32), P.size([Tsize, Esize]))
        self.Mrel = SparseTensor(
            mindspore.ops.Stack((idx, triples[:,1])), mindspore.Tensor([1] * Tsize, mindspore.float32), P.size([Tsize, num_relations]))
        print('triple size: {}'.format(Tsize))
        try:
            if args.bert_name == "bert-base-uncased":
                model = "mindspore/1.9/bertbase_cnnews128"
                self.bert_encoder = mshub.load(model)
            elif args.bert_name == "roberta-base":
                model = "mindspore/1.9/bertbase_cnnews128"
                self.bert_encoder = mshub.load(model)
            else:
                raise ValueError("please input the right name of pretrained model")
        except ValueError as e:
            raise e
        dim_hidden = self.bert_encoder.config.hidden_size
        self.rel_classifier = nn.Dense(in_channels=dim_hidden, out_channels=num_relations)
        self.key_layer = nn.Dense(in_channels=dim_hidden, out_channels=dim_hidden)
        self.hop_att_layer = nn.SequentialCell([
            nn.Dense(in_channels=dim_hidden, out_channels=1)
        ])

        self.high_way = nn.SequentialCell([
            nn.Dense(in_channels=dim_hidden, out_channels=dim_hidden),
            nn.Sigmoid()
        ])

    def follow(self, e, r):
        x = P.sparse.mm(self.Msubj, e.t()) * P.sparse.mm(self.Mrel, r.t())
        return P.sparse.mm(self.Mobj.t(), x).t() # [bsz, Esize]

    def construct(self, heads, questions, answers=None, entity_range=None):
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)

        device = heads.device
        last_e = heads
        word_attns = []
        rel_probs = []
        ent_probs = []
        ctx_h_list = []
        q_word_h_hop = q_word_h
        q_word_h_dist_ctx = [0]
        for t in range(self.num_steps):
            h_key = self.key_layer(q_word_h_hop)  # [bsz, max_q, dim_h]
            q_logits = mindspore.ops.MatMul(h_key, q_word_h.transpose(-1, -2)) # [bsz, max_q, dim_h] * [bsz, dim_h, max_q] = [bsz, max_q, max_q]
            q_logits = q_logits.transpose(-1, -2)

            q_dist = mindspore.nn.Softmax(axis=2)(q_logits)  # [bsz, max_q, max_q]
            q_dist = q_dist * questions['attention_mask'].float().unsqueeze(1)  # [bsz, max_q, max_q]*[bsz, max_q]
            q_dist = q_dist / (mindspore.ops.ReduceSum(q_dist, dim=2, keepdim=True) + 1e-6) # [bsz, max_q, max_q]
            hop_ctx = mindspore.ops.MatMul(q_dist, q_word_h_hop)
            if t == 0:
                z = 0
            else:
                z = self.high_way(q_word_h_dist_ctx[-1]) 
            if t == 0:
                q_word_h_hop = q_word_h + hop_ctx
            else:
                q_word_h_hop = q_word_h + hop_ctx + z*q_word_h_dist_ctx[-1]# [bsz, max_q, max_q]*[bsz, max_q, dim_h] = [bsz, max_q, dim_h]
            q_word_h_dist_ctx.append(hop_ctx + z*q_word_h_dist_ctx[-1])

            q_word_att = mindspore.ops.ReduceSum(q_dist, dim=1, keepdim=True)  # [bsz, 1, max_q]  # 2改为1
            q_word_att = mindspore.nn.Softmax(axis=2)(q_word_att)
            q_word_att = q_word_att * questions['attention_mask'].float().unsqueeze(1)  # [bsz, 1, max_q]*[bsz, max_q]
            q_word_att = q_word_att / (mindspore.ops.ReduceSum(q_word_att, dim=2, keepdim=True) + 1e-6)  # [bsz, max_q, max_q]
            word_attns.append(q_word_att)  # bsz,1,q_max
            ctx_h = (q_word_h_hop.transpose(-1,-2) @ q_word_att.transpose(-1,-2)).squeeze(2)  # [bsz, dim_h, max_q] * [bsz, max_q,1]

            ctx_h_list.append(ctx_h)

            rel_logit = self.rel_classifier(ctx_h) # [bsz, num_relations]
            rel_dist = mindspore.ops.Sigmoid()(rel_logit)
            rel_probs.append(rel_dist)

            last_e = self.follow(last_e, rel_dist)  # faster than index_add

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float() 
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z
            ent_probs.append(last_e)

        hop_res = mindspore.ops.Stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]

        ctx_h_history = mindspore.ops.Stack(ctx_h_list, dim=2)  # [bsz, dim_h, num_hop]
        hop_logit = self.hop_att_layer(ctx_h_history.transpose(-1, -2))  # bsz, num_hop, 1
        hop_attn = mindspore.nn.Softmax(axis=2)(hop_logit.transpose(-1, -2)).transpose(-1, -2)  # bsz, num_hop, 1

        last_e = mindspore.ops.ReduceSum(hop_res * hop_attn, dim=1) # [bsz, num_ent]

        if not self.training:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                'hop_attn': P.ReduceMean(2)(hop_attn)
            }
        else:
            weight = answers * 9 + 1 
            loss = mindspore.ops.ReduceSum(entity_range * weight * P.Pow()(last_e - answers, 2)) / mindspore.ops.ReduceSum(entity_range * weight)

            return {'loss': loss}
