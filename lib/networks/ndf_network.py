import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from lib.config import cfg

from . import embedder


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.actvn = nn.ReLU()

        self.residualxyz_fc_1 = nn.Conv1d(128 + 63, 128, 1)
        self.residualxyz_fc_2 = nn.Conv1d(128, 3, 1)

        self.pose_fc_1 = nn.Conv1d(72, 256, 1)
        self.pose_fc_2 = nn.Conv1d(256, 128, 1)

        input_ch = 63
        pose_ch = 128
        D = 7
        W = 256
        self.skips = [4]
        self.pts_linears = nn.ModuleList(
            [nn.Conv1d(input_ch + pose_ch, W, 1)] + [
                nn.Conv1d(W, W, 1) if i not in
                self.skips else nn.Conv1d(W + input_ch + pose_ch, W, 1)
                for i in range(D - 1)
            ])
        self.alpha_fc = nn.Conv1d(W, 1, 1)

        self.feature_fc = nn.Conv1d(W, W, 1)
        self.latent_fc = nn.Conv1d(W + pose_ch, W, 1)
        self.view_fc = nn.Conv1d(283, W // 2, 1)
        self.rgb_fc = nn.Conv1d(W // 2, 3, 1)

    def calculate_density_color(self, nf_pts, viewdir, pose):
        pose = pose.transpose(1, 2)
        pose_1 = self.actvn(self.pose_fc_1(pose))
        pose_2 = self.actvn(self.pose_fc_2(pose_1))

        embed_pts = embedder.xyz_embedder(nf_pts)
        embed_pts = embed_pts.transpose(1, 2)

        xyzfeatures = torch.cat((embed_pts, pose_2), dim=1)
        xyzfeatures = self.actvn(self.residualxyz_fc_1(xyzfeatures))
        xyzresidual = self.residualxyz_fc_2(xyzfeatures)
        xyzdeform = nf_pts + 0.001 * xyzresidual.transpose(1, 2)

        xyzdeform = embedder.xyz_embedder(xyzdeform)
        input_pts = xyzdeform.transpose(1, 2)

        net = torch.cat((input_pts, pose_2), dim=1)

        for i, l in enumerate(self.pts_linears):
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net, pose_2), dim=1)
        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        features = torch.cat((features, pose_2), dim=1)
        features = self.latent_fc(features)
        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir.transpose(1, 2)
        features = torch.cat((features, viewdir), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        return raw
