import torch
import numpy as np
from graphalign.algorithms.restricted.HTC.utils import *
from torch import optim
import torch.nn.functional as F
from dataclasses import dataclass

from graphalign._logging import logger
from graphalign.algorithms.restricted.HTC.utils import CSLS
from graphalign.graph import GraphPair
from graphalign.algorithms.restricted.HTC.MyNet import MyNet, Reconstruction_loss
from graphalign.algorithms.algorithm import Algorithm


@dataclass
class HTC(Algorithm):
    pair: GraphPair
    src_laps: torch.Tensor
    trg_laps: torch.Tensor
    # Model
    hid_dim: int
    # Unsupervised training
    num_utrn: int
    ulr: float
    # Fine-tuning
    num_ftune: int
    flr: float
    alpha: float
    # Other
    k: int
    p: float

    def __post_init__(self) -> None:
        self.__name__: str = "HTC"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.src_feat = torch.Tensor(self.pair.src_features).to(self.device)
        self.trg_feat = torch.Tensor(self.pair.tar_features).to(self.device)
        self.src_laps = self.src_laps.to(self.device)
        self.trg_laps = self.trg_laps.to(self.device)
        self.num_node_s = self.src_feat.shape[0]
        self.num_node_t = self.trg_feat.shape[0]
        self.num_feat = self.src_feat.shape[1]
        logger.info('attribute dimension: %d' % self.num_feat)
        self.num_hid1 = self.hid_dim
        self.num_hid2 = self.hid_dim
        self.gt = self.pair.ground_truth

    def _evaluate(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        myNet = MyNet(self.num_node_s, self.num_node_t, self.num_feat, self.num_hid1, self.num_hid2, self.p).to(
            self.device)

        myNet = self.unsupervised_train(myNet)

        myNet, count_max = self.trusted_refine(myNet)

        S_MyAlign = self.weighted_integration(myNet, count_max)

        return S_MyAlign

    def unsupervised_train(self, myNet):
        myNet.train()
        utrain_optimizer = optim.Adam(myNet.parameters(), lr=self.ulr)
        rec_loss = Reconstruction_loss()
        loss_recorder = np.zeros((len(self.src_laps), self.num_utrn + 1))
        for epoch in range(self.num_utrn):
            for i in range(len(self.src_laps)):
                src_output = myNet(self.src_laps[i], self.src_feat)
                trg_output = myNet(self.trg_laps[i], self.trg_feat)
                src_recA = torch.matmul(F.normalize(src_output), F.normalize(src_output).t())
                src_recA = F.normalize((torch.min(src_recA, torch.Tensor([1]).to(self.device))), dim=1)
                trg_recA = torch.matmul(F.normalize(trg_output), F.normalize(trg_output).t())
                trg_recA = F.normalize((torch.min(trg_recA, torch.Tensor([1]).to(self.device))), dim=1)
                loss_st = (rec_loss(self.src_laps[i], src_recA) + rec_loss(self.trg_laps[i], trg_recA))
                loss_recorder[i, epoch] = loss_st
                logger.info('epoch %d | loss_%d: %.4f' % (epoch, i, loss_st))
                utrain_optimizer.zero_grad()
                loss_st.backward()
                utrain_optimizer.step()
        for i in range(len(self.src_laps)):
            src_output = myNet(self.src_laps[i], self.src_feat)
            trg_output = myNet(self.trg_laps[i], self.trg_feat)
            src_recA = torch.matmul(F.normalize(src_output), F.normalize(src_output).t())
            src_recA = F.normalize((torch.min(src_recA, torch.Tensor([1]).to(self.device))), dim=1)
            trg_recA = torch.matmul(F.normalize(trg_output), F.normalize(trg_output).t())
            trg_recA = F.normalize((torch.min(trg_recA, torch.Tensor([1]).to(self.device))), dim=1)
            loss_st = (rec_loss(self.src_laps[i], src_recA) + rec_loss(self.trg_laps[i], trg_recA))
            loss_recorder[i, epoch + 1] = loss_st
            logger.info('epoch %d | loss_%d: %.4f' % (epoch + 1, i, loss_st))
        return myNet

    def trusted_refine(self, myNet):
        if self.num_ftune > 0:
            logger.info('doing refinement')
            count_max = torch.zeros(len(self.src_laps))
            tune_flag = torch.ones(len(self.src_laps))
        else:
            count_max = torch.ones(len(self.src_laps)) / len(self.src_laps)
            return myNet, count_max
        for epoch in range(self.num_ftune):
            if tune_flag.sum() == 0:
                logger.info('done refinement')
                break
            for i in range(len(self.src_laps)):
                if tune_flag[i] == False:
                    logger.info('epoch %d_%d: undo' % (epoch, i))
                    continue
                src_output = myNet(self.src_laps[i], self.src_feat)
                trg_output = myNet(self.trg_laps[i], self.trg_feat)
                csls = CSLS(src_output.detach(), trg_output.detach(), self.k)
                index_r = torch.argmax(csls, dim=0)
                index_c = torch.argmax(csls, dim=1)
                count = 0
                qs = torch.ones(len(self.src_laps[i])).to(self.device)
                qt = torch.ones(len(self.trg_laps[i])).to(self.device)
                for j in range(len(index_r)):
                    if j == index_c[index_r[j]]:
                        count += 1
                        qs[index_r[j]] *= self.alpha
                        qt[j] *= self.alpha
                qs = qs.reshape(-1, 1)
                qt = qt.reshape(-1, 1)
                if count > count_max[i]:
                    count_max[i] = count
                    self.src_laps[i] = (qs * (self.src_laps[i] * qs).t()).t()
                    self.trg_laps[i] = (qt * (self.trg_laps[i] * qt).t()).t()
                else:
                    tune_flag[i] = False
                logger.info('epoch %d_%d: mutual closest pairs: %d' % (epoch, i, count))
        return myNet, count_max

    def weighted_integration(self, myNet, count_max):
        myNet.eval()
        total_count = sum(count_max)
        count_max = count_max / total_count
        score = torch.zeros((self.num_node_s, self.num_node_t)).to(self.device)
        for i in range(len(self.src_laps)):
            src_output = myNet(self.src_laps[i], self.src_feat)
            trg_output = myNet(self.trg_laps[i], self.trg_feat)
            s = CSLS(src_output.detach(), trg_output.detach(), self.k)
            score += count_max[i] * s
        score = score.detach().cpu().numpy()
        return score