import imp

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg


def _wrapper_renderer(cfg, network):
    module = cfg.renderer_module
    path = cfg.renderer_path
    renderer_wrapper = imp.load_source(module, path).Renderer(network)
    return renderer_wrapper


class NetworkWrapper(nn.Module):
    def __init__(self, net, cfg=None):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.renderer = _wrapper_renderer(cfg, self.net)

        self.img2mse = lambda x, y: torch.mean((x - y)**2)
        self.acc_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0

        mask = batch['mask_at_box']
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        if cfg.get('shell_loss') == True and img_loss <= 0.005:

            near_mask = batch['biggerIndex']
            mask_shell = near_mask.reshape(-1)
            density = F.relu(ret['raw'].view(
                -1, 4)[mask_shell][:, -1])
            shell_loss = density.mean()
            loss += 0.1 * shell_loss
            scalar_stats.update({'shell_loss': shell_loss})

        scalar_stats.update({
            'loss': loss,
        })
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
