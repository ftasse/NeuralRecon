import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset

import random
from scipy.spatial.transform import Rotation

class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)), # , 'open3d_tsdfusion'
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]
        
        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])
        
#         for i, vid in enumerate(meta['image_ids']):
#             # load images
#             imgs.append(
#                 self.read_img(
#                     os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

#             depth.append(
#                 self.read_depth(
#                     os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
#             )

#             # load intrinsics and extrinsics
#             intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
#                                                         vid)

#             intrinsics_list.append(intrinsics)
#             extrinsics_list.append(extrinsics)

        

        rotz_90 = np.eye(4); rotz_90[:3,:3] = Rotation.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix()
        rsteps = random.randint(0,1)*2
        random_downsample = random.randint(0,2)
        
        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
            )

            # load image intrinsics (not depth) and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                                        vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)
            
            K = intrinsics_list[-1].copy()
            cam_pose = extrinsics_list[-1].copy()
            depth_cur = depth[-1].copy()
            if rsteps > 0:
                for rstep in range(rsteps):
                    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
                    cam_pose = np.matmul(cam_pose, rotz_90)
                    imgs[-1] = imgs[-1].rotate(90, expand = True)
                    depth_cur = cv2.rotate(depth_cur, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    K = np.eye(K.shape[0]); K[0,0], K[1,1], K[0,2], K[1,2] = fy, fx, cy, (imgs[-1].size[1]-1)-cx
            if random_downsample:
                sc = (2*random_downsample)
                K[:2,:] /= sc
                imgs[-1] = imgs[-1].resize((imgs[-1].size[0]//sc,imgs[-1].size[1]//sc))
            intrinsics_list[-1] = K
            extrinsics_list[-1] = cam_pose
            depth[-1] = depth_cur

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        
        # print(imgs[-1].size, depth[-1].shape, intrinsics.shape, extrinsics.shape, tsdf_list[0].shape, tsdf_list[1].shape)
        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }

        if self.transforms is not None:
            items = self.transforms(items)
        return items
