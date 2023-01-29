import collections
import os
import pickle
from collections import defaultdict
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
from utils.misc import *
import numpy as np


class KnowledgeGraph(nn.Cell):
    def __init__(self, args, vocab):
        super(KnowledgeGraph, self).__init__()
        self.args = args
        self.entity2id, self.id2entity = vocab['entity2id'], vocab['id2entity']
        self.relation2id, self.id2relation = vocab['relation2id'], vocab['id2relation']
        Msubj = torch.from_numpy(np.load(os.path.join(args.input_dir, 'Msubj.npy'))).long()
        Mobj = torch.from_numpy(np.load(os.path.join(args.input_dir, 'Mobj.npy'))).long()
        Mrel = torch.from_numpy(np.load(os.path.join(args.input_dir, 'Mrel.npy'))).long()
        Tsize = Msubj.size()[0]
        Esize = len(self.entity2id)
        Rsize = len(self.relation2id)
        self.Msubj = torch.sparse.FloatTensor(Msubj.t(), mindspore.Tensor([1] * Tsize, mindspore.flaot32), torch.Size([Tsize, Esize]))
        self.Mobj = torch.sparse.FloatTensor(Mobj.t(), mindspore.Tensor([1] * Tsize, mindspore.flaot32), torch.Size([Tsize, Esize]))
        self.Mrel = torch.sparse.FloatTensor(Mrel.t(), mindspore.Tensor([1] * Tsize, mindspore.float32), torch.Size([Tsize, Rsize]))
        self.num_entities = len(self.entity2id)
 
