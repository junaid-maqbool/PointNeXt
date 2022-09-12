import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from ..data_util import crop_pc, voxelize
from ..build import DATASETS


@DATASETS.register_module()
class TELSTRA(Dataset):
    classes = ['antenna', 'transceiver_junction', 'head_frame_mount', 'shelter', 'background']
    num_classes = 5
    # num_per_class = np.array([41682, 4560, 18015, 118050, 3346605], dtype=np.int32)
    # num_per_class = np.array([100, 30, 60, 300], dtype=np.int32)
    num_per_class = np.array([100, 45, 65, 150, 400], dtype=np.int32)

    class2color = {'antenna':                [0, 255, 0],
                   'transceiver_junction':   [0, 0, 255],
                   'head_frame_mount':       [0, 255, 255],
                   'shelter':                [255, 255, 0],
                   'background':             [50, 50, 50]}

    cmap = [*class2color.values()]
    """S3DIS dataset, loading the subsampled entire room as input without block/sphere subsampling.
    Args:
        data_root (str, optional): Defaults to 'data/S3DIS/s3disfull'.
        test_area (int, optional): Defaults to 5.
        voxel_size (float, optional): the voxel size for donwampling. Defaults to 0.04.
        voxel_max (_type_, optional): subsample the max number of point per point cloud. Set None to use all points.  Defaults to None.
        split (str, optional): Defaults to 'train'.
        transform (_type_, optional): Defaults to None.
        loop (int, optional): split loops for each epoch. Defaults to 1.
        presample (bool, optional): wheter to downsample each point cloud before training. Set to False to downsample on-the-fly. Defaults to False.
        variable (bool, optional): where to use the original number of points. The number of point per point cloud is variable. Defaults to False.
        n_shifted (int, optional): the number of shifted coordinates to be used. Defaults to 1 to use the height.
    """
    def __init__(self,
                 data_root: str = 'data/S3DIS/s3disfull',
                 test_area: int = 5,
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 split: str = 'train',
                 transform=None,
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 n_shifted: int = 0,
                 shuffle: bool = True
                 ):

        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle
        self.n_shifted = n_shifted

        raw_root = os.path.join(data_root, 'raw')
        self.raw_root = raw_root
        data_list = sorted(os.listdir(raw_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [
                item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [
                item for item in data_list if 'Area_{}'.format(test_area) in item]

        np.random.seed(0)
        self.data = []
        for item in tqdm(self.data_list, desc=f'Loading S3DISFull {split} split on Test Area {test_area}'):
            data_path = os.path.join(raw_root, item + '.npy')
            cdata = np.load(data_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            self.data.append(cdata)
        npoints = np.array([len(data) for data in self.data])
        logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' %
                     (self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data_path = os.path.join(
            self.raw_root, self.data_list[data_idx] + '.npy')
        cdata = np.load(data_path).astype(np.float32)
        cdata[:, :3] -= np.min(cdata[:, :3], 0)
        coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
        # Note: random=True randomly samples voxel_max points, random=False uses random norm ball crop to select
        # voxel_max points
        # variable=True and batch_size=1, then variable number of points are allowed
        coord, feat, label = crop_pc(
            coord, feat, label, self.split, self.voxel_size, self.voxel_max,
            downsample=False, random=False, variable=self.variable, shuffle=self.shuffle)
        label = label.squeeze(-1).astype(np.long)
        data = {'pos': coord, 'x': feat, 'y': label}
        # pre-process.
        if self.transform is not None:
            data = self.transform(data)
        data['x'] = torch.cat((data['x'], torch.from_numpy(
            coord[:, 3-self.n_shifted:3].astype(np.float32))), dim=-1)
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop


"""debug 
from openpoints.dataset import vis_multi_points
import copy
old_data = copy.deepcopy(data)
if self.transform is not None:
    data = self.transform(data)
vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
"""
