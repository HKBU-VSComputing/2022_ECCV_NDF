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
        W_direction = 0
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
        self.view_fc = nn.Conv1d(W+W_direction, W // 2, 1)
        self.rgb_fc = nn.Conv1d(W // 2, 3, 1)

    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def get_grid_coords(self, pts, sp_input):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = sp_input['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        #? out_sh 是什么.是volume有几个voxel
        #TODO check this out_sh, 为什么要除以它
        #! 可能是，切换到左下角之类的
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def calculate_density(self, nf_pts, pose):
        nf_pts = embedder.xyz_embedder(nf_pts)
        input_pts = nf_pts.transpose(1, 2)
        net = input_pts

        pose = pose.transpose(1, 2)

        pose_1 = self.actvn(self.pose_fc_1(pose))
        pose_2 = self.actvn(self.pose_fc_2(pose_1))

        net = torch.cat((input_pts, pose_2), dim=1)

        for i, l in enumerate(self.pts_linears):
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net, pose_2), dim=1)
        alpha = self.alpha_fc(net)
        alpha = alpha.transpose(1, 2)
        return alpha

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
        # xyzdeform = nf_pts

        xyzdeform = embedder.xyz_embedder(xyzdeform)
        input_pts = xyzdeform.transpose(1, 2)

        # embed()

        # 0.001*xyzresidual[..., torch.where(nf_pts!=-1)[1][::3]]

        net = torch.cat((input_pts, pose_2), dim=1)

        for i, l in enumerate(self.pts_linears):
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net, pose_2), dim=1)
        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        features = torch.cat((features, pose_2), dim=1)
        features = self.latent_fc(features)
        # viewdir = embedder.view_embedder(viewdir)
        # viewdir = viewdir.transpose(1, 2)
        # features = torch.cat((features, viewdir), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        return raw

    def forward(self, sp_input, grid_coords, viewdir, light_pts):
        coord = sp_input['coord']
        out_sh = sp_input['out_sh']
        batch_size = sp_input['batch_size']

        p_features = grid_coords.transpose(1, 2)
        grid_coords = grid_coords[:, None, None]

        code = self.c(torch.arange(0, 6890).to(p_features.device))
        xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)

        xyzc_features = self.xyzc_net(xyzc, grid_coords)

        net = self.actvn(self.fc_0(xyzc_features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        latent = self.latent(sp_input['latent_index'])
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = viewdir.transpose(1, 2)
        light_pts = light_pts.transpose(1, 2)
        features = torch.cat((features, viewdir, light_pts), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        return raw


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(16, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
