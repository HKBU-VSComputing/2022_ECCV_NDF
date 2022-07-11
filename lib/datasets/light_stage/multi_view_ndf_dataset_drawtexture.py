import os

import cv2
import imageio
import numpy as np
import torch.utils.data as data
from IPython import embed
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils.IT_utils import \
    get_sampling_points_drawtexture as get_sampling_points
from lib.utils.IT_utils import read_obj_uv

DEBUG = False
# Name is for implicit texture


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split, current_epoch=0):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        try:
            if len(cfg.test_view) == 0:
                test_view = [
                    i for i in range(num_cams) if i not in cfg.training_view
                ]
                if len(test_view) == 0:
                    test_view = [0]
            else:
                test_view = cfg.test_view
        except:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
        view = cfg.training_view if split == 'train' else test_view
        if len(view) == 0:
            view = [0]

        # prepare input images
        i = 0
        i = i + cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose:
            i = (i + cfg.num_train_frame) * i_intv
            ni = cfg.num_novel_pose_frame
            if self.human == 'CoreView_390':
                i = 0
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.num_cams = len(view)

        self.nrays = cfg.N_rand
        self.thickness = cfg.thickness
        self.thickness_validscale = cfg.thickness_validscale
        uvPath = cfg.uvPath
        self.erode = cfg.erode

        points, faces, UVmap, UVcoordinate, normals = read_obj_uv(uvPath)
        _, self.faces, self.UVmap, self.UVcoordinate = points, faces, UVmap, UVcoordinate

    def get_mask(self, index):
        try:
            msk_path = os.path.join(self.data_root, 'mask_cihp',
                                    self.ims[index])[:-4] + '.png'
            msk_cihp = imageio.imread(msk_path)
        except:
            try:
                msk_path = os.path.join(self.data_root, 'mask_cihp',
                                        self.ims[index][7:])[:-4] + '.png'
                msk_cihp = imageio.imread(msk_path)
            except:
                msk_path = os.path.join(self.data_root, 'mask_cihp',
                                        self.ims[index][7:])[:-4] + '.jpg'
                msk_cihp = imageio.imread(msk_path)
        if msk_cihp.shape[-1] == 3:
            msk_cihp = msk_cihp.mean(-1)
        msk = (msk_cihp != 0).astype(np.uint8)
        if self.erode != 0:
            border = self.erode
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100
        return msk

    def prepare_input_IT(self, i, xyz):
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)
        return can_bounds

    def __getitem__(self, index):

        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        img = cv2.resize(img, (cfg.W, cfg.H))
        msk = self.get_mask(index)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        if not os.path.exists(vertices_path):
            vertices_path = os.path.join(self.data_root, cfg.vertices,
                                         '{0:06d}.npy'.format(i))

            transformationMatrix_path = os.path.join(self.data_root,
                                                     cfg.transformationMatrix,
                                                     '{0:06d}.npy'.format(i))
            params_path = os.path.join(self.data_root, cfg.params,
                                       '{0:06d}.npy'.format(i))
            params = np.load(params_path, allow_pickle=True).item()
        else:
            transformationMatrix_path = os.path.join(self.data_root,
                                                     cfg.transformationMatrix,
                                                     '{}.npy'.format(i))
            params_path = os.path.join(self.data_root, cfg.params,
                                       '{}.npy'.format(i))
            params = np.load(params_path, allow_pickle=True).item()

        vertices = np.load(vertices_path)
        if vertices.shape[0] == 1:
            vertices = vertices[0]

        transformationMatrix = np.load(transformationMatrix_path)

        can_bounds = self.prepare_input_IT(i, vertices)

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_all(
            img, msk, K, R, T, can_bounds, self.nrays, self.split)

        resultLocation, z_vals, biggerIndex = get_sampling_points(
            ray_o,
            ray_d,
            near,
            far,
            vertices,
            self.faces,
            cfg.N_samples,
            self.split,
            cfg.perturb,
            self.nrays,
            transformationMatrix,
            thickness=self.thickness,
            thickness_validscale=self.thickness_validscale,
        )
        latent_index = frame_index - cfg.begin_ith_frame

        ret = {
            'poseSMPL': np.array(params['poses'][0]).astype(np.float32),
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'resultLocation': resultLocation,
            'z_vals': z_vals,
            'biggerIndex': biggerIndex
        }

        meta = {
            'frame_index': frame_index,
            'cam_ind': cam_ind,
            'latent_index': latent_index,
            'meta': img_path
        }
        ret.update(meta)
        return ret

    def __len__(self):
        return len(self.ims)
