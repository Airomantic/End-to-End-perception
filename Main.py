import torch
import torch.utils.data
from nuscenes.nuscenes import NuScenes
from NuscenesData import FuturePredictionDataset
from geometry import cumulative_warp_features_reverse, cumulative_warp_features
from visualisation import visualise_output
import numpy as np
import imageio
LIFT_X_BOUND = [-50.0, 50.0, 0.5]  #  Forward
LIFT_Y_BOUND = [-50.0, 50.0, 0.5]
spatial_extent = (LIFT_X_BOUND[1], LIFT_Y_BOUND[1])
def prepare_future_labels(batch):
    labels = {}
    receptive_field=3
    segmentation_labels = batch['segmentation']
    hdmap_labels = batch['hdmap']
    future_egomotion = batch['future_egomotion']
    gt_trajectory = batch['gt_trajectory']

    # present frame hd map gt
    labels['hdmap'] = hdmap_labels[:, receptive_field - 1].long().contiguous()

    # gt trajectory
    labels['gt_trajectory'] = gt_trajectory

    # Warp labels to present's reference frame
    segmentation_labels_past = cumulative_warp_features(
        segmentation_labels[:, :receptive_field].float(),
        future_egomotion[:, :receptive_field],
        mode='nearest', spatial_extent=spatial_extent,
    ).long().contiguous()[:, :-1]
    segmentation_labels = cumulative_warp_features_reverse(
        segmentation_labels[:, (receptive_field - 1):].float(),
        future_egomotion[:, (receptive_field - 1):],
        mode='nearest', spatial_extent=spatial_extent,
    ).long().contiguous()
    labels['segmentation'] = torch.cat([segmentation_labels_past, segmentation_labels], dim=1)


    pedestrian_labels = batch['pedestrian']
    pedestrian_labels_past = cumulative_warp_features(
        pedestrian_labels[:, :receptive_field].float(),
        future_egomotion[:, :receptive_field],
        mode='nearest', spatial_extent=spatial_extent,
    ).long().contiguous()[:, :-1]
    pedestrian_labels = cumulative_warp_features_reverse(
        pedestrian_labels[:, (receptive_field - 1):].float(),
        future_egomotion[:, (receptive_field - 1):],
        mode='nearest', spatial_extent=spatial_extent,
    ).long().contiguous()
    labels['pedestrian'] = torch.cat([pedestrian_labels_past, pedestrian_labels], dim=1)

    # Warp instance labels to present's reference frame

    gt_instance = batch['instance']
    instance_center_labels = batch['centerness']
    instance_offset_labels = batch['offset']
    gt_instance_past = cumulative_warp_features(
        gt_instance[:, :receptive_field].float().unsqueeze(2),
        future_egomotion[:, :receptive_field],
        mode='nearest', spatial_extent=spatial_extent,
    ).long().contiguous()[:, :-1, 0]
    gt_instance = cumulative_warp_features_reverse(
        gt_instance[:, (receptive_field - 1):].float().unsqueeze(2),
        future_egomotion[:, (receptive_field - 1):],
        mode='nearest', spatial_extent=spatial_extent,
    ).long().contiguous()[:, :, 0]
    labels['instance'] = torch.cat([gt_instance_past, gt_instance], dim=1)

    instance_center_labels_past = cumulative_warp_features(
        instance_center_labels[:, :receptive_field],
        future_egomotion[:, :receptive_field],
        mode='nearest', spatial_extent=spatial_extent,
    ).contiguous()[:, :-1]
    instance_center_labels = cumulative_warp_features_reverse(
        instance_center_labels[:, (receptive_field - 1):],
        future_egomotion[:, (receptive_field - 1):],
        mode='nearest', spatial_extent=spatial_extent,
    ).contiguous()
    labels['centerness'] = torch.cat([instance_center_labels_past, instance_center_labels], dim=1)

    instance_offset_labels_past = cumulative_warp_features(
        instance_offset_labels[:, :receptive_field],
        future_egomotion[:, :receptive_field],
        mode='nearest', spatial_extent=spatial_extent,
    ).contiguous()[:, :-1]
    instance_offset_labels = cumulative_warp_features_reverse(
        instance_offset_labels[:, (receptive_field - 1):],
        future_egomotion[:, (receptive_field - 1):],
        mode='nearest', spatial_extent=spatial_extent,
    ).contiguous()
    labels['offset'] = torch.cat([instance_offset_labels_past, instance_offset_labels], dim=1)

    instance_flow_labels = batch['flow']
    instance_flow_labels_past = cumulative_warp_features(
        instance_flow_labels[:, :receptive_field],
        future_egomotion[:, :receptive_field],
        mode='nearest', spatial_extent=spatial_extent,
    ).contiguous()[:, :-1]
    instance_flow_labels = cumulative_warp_features_reverse(
        instance_flow_labels[:, (receptive_field - 1):],
        future_egomotion[:, (receptive_field - 1):],
        mode='nearest', spatial_extent=spatial_extent,
    ).contiguous()
    labels['flow'] = torch.cat([instance_flow_labels_past, instance_flow_labels], dim=1)

    return labels

dataroot = 'v1.0-mini/'
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False) #创建NuScenes类，方便调用官方的函数
traindata = FuturePredictionDataset(nusc, 0) #调用构造e2e数据集函数

nworkers = 1
trainloader = torch.utils.data.DataLoader(
    traindata, batch_size=1, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
) #构造torch dataloader类，用于变成迭代器，训练模型


for batch in trainloader:
    print(batch)
    image = batch['image']
    intrinsics = batch['intrinsics']
    extrinsics = batch['extrinsics']
    future_egomotion = batch['future_egomotion']
    command = batch['command']
    trajs = batch['sample_trajectory']
    target_points = batch['target_point']
    B = len(image)

    # Warp labels
    labels = prepare_future_labels(batch)
    visualisation_video = visualise_output(labels)
    save_res = np.transpose(visualisation_video[0], (0, 3, 2, 1))
    imageio.mimsave('test.gif', save_res, fps=2)
#
#     return trainloader

