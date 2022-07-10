import torch
from lib.config import cfg
from OpenGL.GL import *

from .nerf_net_utils import *


class Renderer:
    def __init__(self, net):
        self.net = net

    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]

        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])

        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        # used for feature interpolation
        sp_input['bounds'] = batch['bounds']
        sp_input['R'] = batch['R']
        sp_input['Th'] = batch['Th']

        # used for color function
        sp_input['latent_index'] = batch['latent_index']

        return sp_input

    def get_density_color(self, wpts, viewdir, raw_decoder):
        raw = raw_decoder(wpts, viewdir)
        return raw

    def get_pixel_value(self,
                        ray_o,
                        ray_d,
                        near,
                        far,
                        resultLocation,
                        z_vals,
                        pose,
                        biggerIndex=None):

        wpts = resultLocation

        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        pose = pose.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        # wpts = wpts.float()
        wpts_raw = self.net.calculate_density_color(wpts, viewdir, pose)
        wpts_raw[torch.where(torch.sum(wpts, -1) == -3)] = 0

        raw = wpts_raw.reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)

        ret = {
            'raw': raw.view(n_batch, n_pixel, -1, 4),
            'rgb_map': rgb_map.view(n_batch, n_pixel, -1),
            'disp_map': disp_map.view(n_batch, n_pixel),
            'acc_map': acc_map.view(n_batch, n_pixel),
            'weights': weights.view(n_batch, n_pixel, -1),
            'depth_map': depth_map.view(n_batch, n_pixel)
        }

        return ret

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        pose = batch['poseSMPL']
        resultLocation, z_vals = batch['resultLocation'], batch['z_vals']
        sh = ray_o.shape

        B, nrRay, nrSample, channel = resultLocation.shape
        pose = pose[:, None, None].expand([B, nrRay, nrSample, pose.shape[-1]])
        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 1024 * 1

        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            resultLocation_chunk = resultLocation[:, i:i + chunk]
            pose_chunk = pose[:, i:i + chunk]
            z_vals_chunk = z_vals[:, i:i + chunk]

            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               resultLocation_chunk,
                                               z_vals_chunk, pose_chunk)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}
        return ret
