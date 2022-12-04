# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : MI.py
# DATE : 2022/8/27 10:29
import torch
import torch.nn.functional as F
import numpy as np


class MI(torch.nn.Module):
    def __init__(self, eps=1e-7, is_logit=True, temperature=0.05, joint_estimation_samples=100):
        super(MI, self).__init__()
        self.eps = eps
        self.is_logit = is_logit
        self.t = temperature
        self.joint_estimation_samples = joint_estimation_samples

    def forward(self, x_seen, x_unseen):
        """
        compute mi loss
        x_seen: the label head output (B, C1)
        x_unseen: unlabel head output (B, C2)
        """
        v, h, B, C1 = x_seen.shape
        _, _, B, C2 = x_unseen.shape
        # permute
        x_seen = x_seen.permute(1, 0, 2, 3).reshape(h, v*B, C1)
        x_unseen = x_unseen.permute(1, 0, 2, 3).reshape(h, v*B, C2)

        if self.is_logit:
            x_seen = F.softmax(x_seen / self.t, dim=-1)
            x_unseen = F.softmax(x_unseen / self.t, dim=-1).permute(0, 2, 1)

        p_joint = torch.bmm(x_unseen / np.sqrt(v*B), x_seen / np.sqrt(v*B))

        for i in range(self.joint_estimation_samples):
            idx = torch.randperm(v*B)
            p_joint += torch.bmm(x_unseen[:, :, idx] / np.sqrt(v*B), x_seen[:, idx] / np.sqrt(v*B))

        p_joint /= (self.joint_estimation_samples + 1)

        p_seen = x_seen.mean(dim=1, keepdim=True)
        p_unseen = x_unseen.mean(dim=2, keepdim=True)

        p_joint[p_joint < self.eps] = self.eps
        p_seen[p_seen < self.eps] = self.eps
        p_unseen[p_unseen < self.eps] = self.eps
        # mutual information
        # conditional entropy
        cond_entropy = -torch.mean(torch.sum(p_joint*(torch.log(p_joint) - torch.log(p_seen)), dim=[1, 2]))

        # analysis the effect of entropy
        entropy = torch.mean(torch.sum(p_unseen * torch.log(p_unseen), dim=[1, 2]))

        mi = cond_entropy + entropy
        return mi