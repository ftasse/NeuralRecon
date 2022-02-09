import sys
sys.path.append('.')

import time
import glob
import shutil
from tools.tsdf_fusion.fusion import *
import pickle
import argparse
from tqdm import tqdm
import ray
import torch.multiprocessing
from tools.simple_loader import *

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf, using scannet with framedata saved in *.npz')
    parser.add_argument("--dataset", default='scannet')
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to raw dataset", default='/data/scannet/output/')
    parser.add_argument("--save_name", metavar="DIR",
                        help="file name", default='all_tsdf')
    parser.add_argument('--test', action='store_true',
                        help='prepare the test set')
    parser.add_argument('--max_depth', default=3., type=float,
                        help='mask out large depth values since they are noisy')
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--margin', default=3, type=int)
    parser.add_argument('--voxel_size', default=0.04, type=float)

    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--min_angle', default=15, type=float)
    parser.add_argument('--min_distance', default=0.1, type=float)

    # ray multi processes
    parser.add_argument('--n_proc', type=int, default=16, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=2, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()


args = parse_args()
args.save_path = os.path.join(args.data_path, args.save_name)
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    

import imageio
import zlib

class ScanNetDatasetNPZ(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, scene, data_path, max_depth, id_list=None):
        """
        Args:
        """
        self.n_imgs = 0
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        
            
        sensor_data_path = "%s/%s/%s_frames.npz"%(data_path, scene, scene)
        if not os.path.exists(sensor_data_path):  
            print("skip %s"%sensor_data_path, ": *.sens could not be downloaded, so *.npz does not exists")
        else:
            sensor_data = np.load(sensor_data_path, allow_pickle=1)
            self.frames = sensor_data['frames']
            self.n_imgs = len(self.frames)
            self.depth_height, self.depth_width = sensor_data["depth_height"], sensor_data["depth_width"]
            self.intrinsic_depth = sensor_data["intrinsic_depth"][:3,:3]

        if id_list is None:
            self.id_list = [i for i in range(self.n_imgs)]
        else:
            self.id_list = id_list
            
    def __len__(self):
        return self.n_imgs

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        id = self.id_list[id]
        
        frame = self.frames[id]
        color_image = cv2.resize(imageio.imread(frame['color_jpg_data']),
                                 (self.depth_width, self.depth_height), interpolation=cv2.INTER_AREA)
        depth_im = np.frombuffer(zlib.decompress(frame['depth_zlibushort_data']), 
                              dtype=np.uint16).reshape(self.depth_height, self.depth_width)
        depth_im = depth_im.astype(np.float32)/1000.0
        depth_im[depth_im > self.max_depth] = 0
        cam_pose = (frame['camera_to_world'])
        
        return cam_pose, depth_im, color_image


def save_tsdf_full(args, scene_path, cam_intr, depth_list, cam_pose_list, color_list, save_mesh=False):
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    vol_bnds = np.zeros((3, 2))

    n_imgs = len(depth_list.keys())
    if n_imgs > 200:
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_id = np.array(list(depth_list.keys()))[ind]
    else:
        image_id = depth_list.keys()
    for id in image_id:
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol_list = []
    for l in range(args.num_layers):
        tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=args.voxel_size * 2 ** l, margin=args.margin))

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for id in depth_list.keys():
        if id % 100 == 0:
            print("{}: Fusing frame {}/{}".format(scene_path, str(id), str(n_imgs)))
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        if len(color_list) == 0:
            color_image = None
        else:
            color_image = color_list[id]

        # Integrate observation into voxel volume (assume color aligned with depth)
        for l in range(args.num_layers):
            tsdf_vol_list[l].integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    tsdf_info = {
        'vol_origin': tsdf_vol_list[0]._vol_origin,
        'voxel_size': tsdf_vol_list[0]._voxel_size,
    }
    tsdf_path = os.path.join(args.save_path, scene_path)
    if not os.path.exists(tsdf_path):
        os.makedirs(tsdf_path)

    with open(os.path.join(args.save_path, scene_path, 'tsdf_info.pkl'), 'wb') as f:
        pickle.dump(tsdf_info, f)

    for l in range(args.num_layers):
        tsdf_vol, color_vol, weight_vol = tsdf_vol_list[l].get_volume()
        np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_tsdf_layer{}'.format(str(l))), tsdf_vol)

    if save_mesh:
        try:
            for l in [0]: # range(args.num_layers):
                print("Saving mesh to mesh{}.ply...".format(str(l)))
                verts, faces, norms, colors = tsdf_vol_list[l].get_mesh()

                meshwrite(os.path.join(args.save_path, scene_path, 'mesh_layer{}.ply'.format(str(l))), verts, faces, norms,
                      colors)

                # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
                # print("Saving point cloud to pc.ply...")
                # point_cloud = tsdf_vol_list[l].get_point_cloud()
                # pcwrite(os.path.join(args.save_path, scene_path, 'pc_layer{}.ply'.format(str(l))), point_cloud)
        except: pass

def save_fragment_pkl(args, scene, cam_intr, depth_list, cam_pose_list, color_list):
    fragments = []
    print('segment: process scene {}'.format(scene))

    # gather pose
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.inf
    vol_bnds[:, 1] = -np.inf

    all_ids = []
    ids = []
    all_bnds = []
    count = 0
    last_pose = None
    for id in depth_list.keys():
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        if count == 0:
            ids.append(id)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.inf
            vol_bnds[:, 1] = -np.inf
            last_pose = cam_pose
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                # Compute camera view frustum and extend convex hull
                view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    all_bnds.append(vol_bnds)
                    ids = []
                    count = 0

    with open(os.path.join(args.save_path, scene, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)

    # save fragments
    for i, bnds in enumerate(all_bnds):
        # if not os.path.exists(os.path.join(args.save_path, scene, 'fragments', str(i))):
        #    os.makedirs(os.path.join(args.save_path, scene, 'fragments', str(i)))
        if not os.path.exists(os.path.join(args.save_path, scene, 'keyframes')):
            os.makedirs(os.path.join(args.save_path, scene, 'keyframes'))
        fragments.append({
            'scene': scene,
            'fragment_id': i,
            'image_ids': all_ids[i],
            'vol_origin': tsdf_info['vol_origin'],
            'voxel_size': tsdf_info['voxel_size']
        })
        frag_frames = []
        frag_frames_path = os.path.join(args.save_path, scene, 'keyframes/fragment_%04d.npz'%i)
        for k in all_ids[i]:
            frag_frames.append(dict(
                color =  cv2.imencode('.jpg', cv2.cvtColor(color_list[k],cv2.COLOR_RGB2BGR))[1].tobytes(),
                depth =  cv2.imencode('.png', (depth_list[k]*1000).astype(np.uint16))[1].tobytes(),
                cam_pose=np.stack(cam_pose_list[k]),
                vol_bnds=vol_bnds, cam_intr=cam_intr, frame_id = k
            ))
        np.savez(frag_frames_path, frag_frames)

    with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)

@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_with_single_worker(args, scannet_files):
    for scene in tqdm(scannet_files):
        if os.path.exists(os.path.join(args.save_path, scene, 'fragments.pkl')):
            continue
        print('read from disk')

        depth_all = {}
        cam_pose_all = {}
        color_all = {}

        if args.dataset == 'scannet':
            if os.path.exists("%s/%s/%s_frames.npz"%(args.data_path, scene, scene)):
                dataset = ScanNetDatasetNPZ(scene, args.data_path, args.max_depth)
                n_imgs = dataset.n_imgs
                if (n_imgs == 0): continue
                cam_intr = dataset.intrinsic_depth
            else:
                n_imgs = len(os.listdir(os.path.join(args.data_path, scene, 'color')))
                intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
                cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
                dataset = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth)
            
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
                                                 batch_sampler=None, num_workers=args.loader_num_workers)

        for id, (cam_pose, depth_im, color_image) in enumerate(dataloader):
            if id % 100 == 0:
                print("{}: read frame {}/{}".format(scene, str(id), str(n_imgs)))

            if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                continue
            depth_all.update({id: depth_im})
            cam_pose_all.update({id: cam_pose})
            color_all.update({id: color_image})

        scene_save_path = os.path.join(args.save_path, scene)
        if not os.path.exists(scene_save_path): os.makedirs(os.path.join(args.save_path, scene))
        save_tsdf_full(args, scene, cam_intr, depth_all, cam_pose_all, color_all, save_mesh=not False)
        save_fragment_pkl(args, scene, cam_intr, depth_all, cam_pose_all, color_all)


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


def generate_pkl(args):
    all_scenes = sorted(os.listdir(args.save_path))
    # todo: fix for both train/val/test
    if not args.test:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    if not os.path.exists(os.path.join(args.save_path, 'splits')):
        os.makedirs(os.path.join(args.save_path, 'splits'))
        for split in splits:
            shutil.copyfile(os.path.join(args.data_path, '../scannetv2_%s.txt'%split),
                        os.path.join(args.save_path, 'splits/scannetv2_%s.txt'%split))
    for split in splits:
        fragments = []
        with open(os.path.join(args.save_path, 'splits', 'scannetv2_{}.txt'.format(split))) as f:
            split_files = f.readlines()
        for scene in all_scenes:
            if 'scene' not in scene:
                continue
            if scene + '\n' in split_files:
                with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'rb') as f:
                    frag_scene = pickle.load(f)
                fragments.extend(frag_scene)

        with open(os.path.join(args.save_path, 'fragments_{}.pkl'.format(split)), 'wb') as f:
            pickle.dump(fragments, f)


if __name__ == "__main__":
    ## Example
    # python tools/tsdf_fusion/generate_gt_xstreem.py --data_path=/training_data/scannet --save_name all_tsdf
    
    all_proc = args.n_proc * args.n_gpu

    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    if args.dataset == 'scannet':
        if not args.test:
            args.data_path = os.path.join(args.data_path, 'scans')
        else:
            args.data_path = os.path.join(args.data_path, 'scans_test')
        files = sorted(os.listdir(args.data_path))
        files = [x for x in files if "scene" in x.split('/')[-1]]
    else:
        raise NameError('error!')
    
    files = split_list(files, all_proc)

    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(process_with_single_worker.remote(args, files[w_idx]))

    results = ray.get(ray_worker_ids)

    if args.dataset == 'scannet':
        generate_pkl(args)
