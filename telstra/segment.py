from typing import Dict
from pathlib import Path
from tower_equipment_segmentation.util.dir_util import create_dir
from tower_equipment_segmentation.dataset_converter.dataset_converter import DatasetConverter
from tower_equipment_segmentation.util.dataset_util import get_site_ref_ids_in_dataset_dir, \
    get_pointnext_s3dis_pcd_npy_file_path_for_site_ref_id_and_area_idx, get_pcd_csv_file_path_for_site_ref_id
from tower_equipment_segmentation.dvo.point_cloud_asset_segmentation import PointCloudAssetSegmentation
from tower_equipment_segmentation.dvo.known_class_labels import KnownClassLabels
from tower_equipment_segmentation.util.logging_util import setup_logger

import torch
import argparse
import numpy as np
from tqdm import tqdm

from openpoints.utils import set_random_seed, load_checkpoint, EasyConfig
from openpoints.dataset import build_dataloader_from_cfg, get_scene_seg_features
from openpoints.dataset.data_util import crop_pc
from openpoints.transforms.transforms_factory import build_transforms_from_cfg
from openpoints.models import build_model_from_cfg


logger = setup_logger(__name__)

class_label_to_class_idx_map: Dict[str, int] = {KnownClassLabels.antenna: 0,
                                                KnownClassLabels.transceiver_junction: 1,
                                                KnownClassLabels.head_frame_mount: 2,
                                                KnownClassLabels.shelter: 3,
                                                KnownClassLabels.aviation_light: 3,
                                                KnownClassLabels.background: 3}


def _get_model_and_data_loader_from_configs(config: EasyConfig):
    model = build_model_from_cfg(config.model).to(torch.device('cuda:0'))
    data_loader = build_dataloader_from_cfg(config.get('val_batch_size', config.batch_size),
                                            config.dataset, config.dataloader,
                                            datatransforms_cfg=config.datatransforms,
                                            split='val', distributed=False)
    logger.info(f"length of validation dataset: {len(data_loader.dataset)}")
    num_classes = data_loader.dataset.num_classes if hasattr(data_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert config.num_classes == num_classes
    logger.info(f"number of classes of the dataset: {num_classes}")
    config.classes = data_loader.dataset.classes if hasattr(data_loader.dataset, 'classes') else np.range(num_classes)
    config.cmap = np.array(data_loader.dataset.cmap) if hasattr(data_loader.dataset, 'cmap') else None
    best_epoch, best_val = load_checkpoint(model, pretrained_path=config.pretrained_path)
    logger.info(f"Loaded model weights for best epoch: {best_epoch} with best val miou: {best_val}")
    model.eval()  # set model to eval mode
    return model, data_loader


def _get_pointnext_input_point_clouds_from_dataset(dataset_dir: Path, voxel_max: int, variable: bool,
                                                   shuffle: bool, n_shifted: int = 0):
    for site_ref_id in get_site_ref_ids_in_dataset_dir(dataset_dir=dataset_dir):
        pcd_np_fp = get_pointnext_s3dis_pcd_npy_file_path_for_site_ref_id_and_area_idx(
            dataset_dir=dataset_dir, site_ref_id=site_ref_id, area_index=2)
        cdata = np.load(str(pcd_np_fp)).astype(np.float32)
        cdata[:, :3] -= np.min(cdata[:, :3], 0)
        coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
        coord, feat, label = crop_pc(
            coord, feat, label, 'val', voxel_max,
            downsample=False, random=False, variable=variable, shuffle=shuffle)
        label = label.squeeze(-1).astype(np.long)
        data = {'pos': coord, 'x': feat, 'y': label}
        data_transform = build_transforms_from_cfg('val', model_cfg.datatransforms)
        # pre-process.
        data = data_transform(data)
        if n_shifted > 0:
            data['x'] = torch.cat((data['x'], torch.from_numpy(
                coord[:, 3-n_shifted:3].astype(np.float32))), dim=-1)
        yield site_ref_id, data


def _save_logits_as_asset_seg_predictions_to_asset_seg_pcd_csv_for_site_ref_id(
        dataset_dir: Path, logits: np.ndarray, site_ref_id: str, overwrite_existing_predictions: bool = True):
    pcd_asset_seg_fp = get_pcd_csv_file_path_for_site_ref_id(dataset_dir=dataset_dir, site_ref_id=site_ref_id)
    pcd_asset_seg = PointCloudAssetSegmentation.from_csv_file(csv_file_path=pcd_asset_seg_fp)
    class_labels_to_confidence_scores = {KnownClassLabels.antenna: logits[0],
                                         KnownClassLabels.transceiver_junction: logits[1],
                                         KnownClassLabels.head_frame_mount: logits[2],
                                         KnownClassLabels.shelter: logits[3],
                                         KnownClassLabels.background: logits[4]}
    if pcd_asset_seg.class_labels_to_conf_scores is None or overwrite_existing_predictions:
        pcd_asset_seg.class_labels_to_conf_scores = class_labels_to_confidence_scores
    else:
        pcd_asset_seg.class_labels_to_conf_scores.update(class_labels_to_confidence_scores)

    pcd_asset_seg.to_csv_file(pcd_asset_seg_fp)


@torch.no_grad()
def segment_point_cloud_assets_for_point_cloud_csvs_in_dataset(
        dataset_dir: Path, model_config: EasyConfig, voxel_max: int,
        variable: bool = True, shuffle: bool = False, random_seed: int = 1):

    set_random_seed(random_seed, deterministic=model_config.deterministic)
    torch.backends.cudnn.enabled = True

    # Convert pcd asset seg csvs to pointnext s3dis format
    tmp_dataset_dir = Path('tmp/pointnext_dataset')
    create_dir(tmp_dataset_dir, delete_if_existing=True)
    converter = DatasetConverter()
    converter.convert_tower_asset_segmentation_dataset_to_pointnext_s3dis_format(
        src_dataset_dir=dataset_dir, dst_dataset_dir=tmp_dataset_dir,
        class_label_to_class_idx_map=class_label_to_class_idx_map, area_index=2)
    model, data_loader = _get_model_and_data_loader_from_configs(config=model_config)
    for site_ref_id, data in tqdm(_get_pointnext_input_point_clouds_from_dataset(
            dataset_dir=tmp_dataset_dir, voxel_max=voxel_max, variable=variable, shuffle=shuffle)):
        logger.info(f'Predicting tower asset segmentation labels for for site: {site_ref_id}')
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].unsqueeze(0)
            data[key] = data[key].cuda(non_blocking=True)
        if len(data['x'].shape) > 2:
            data['x'] = data['x'].transpose(1, 2)
        if model_config.model.get('in_channels', None) is None:
            model_config.model.in_channels = model_config.model.encoder_args.in_channels
        data['x'] = get_scene_seg_features(model_config.model.in_channels, data['pos'], data['x'])
        logits = model(data)
        logits = torch.softmax(logits, dim=1)
        logits = logits.detach().cpu().squeeze(0).numpy()
        _save_logits_as_asset_seg_predictions_to_asset_seg_pcd_csv_for_site_ref_id(
            dataset_dir=dataset_dir, logits=logits, site_ref_id=site_ref_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--dataset_dir', type=Path, required=True, help='source dataset dir')
    parser.add_argument('--variable', type=bool, required=False, default=True, help='config file')
    parser.add_argument('--shuffle', type=bool, required=False, default=False, help='config file')
    parser.add_argument('--voxel_max', type=int, required=False, default=600_000, help='config file')
    parser.add_argument('--seed', type=int, required=False, default=1, help='random seed')
    # Loading model configs
    args, opts = parser.parse_known_args()
    model_cfg = EasyConfig()
    model_cfg.load(args.cfg, recursive=True)
    # running PointNeXt semantic segmentation model and saving results in point cloud asset segmentation csvs
    segment_point_cloud_assets_for_point_cloud_csvs_in_dataset(
        dataset_dir=args.dataset_dir, model_config=model_cfg, voxel_max=args.voxel_max,
        variable=args.variable, shuffle=args.shuffle, random_seed=args.seed)
