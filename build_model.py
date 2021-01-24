from mmcv.utils import Registry, build_from_cfg
from torch.nn import GroupNorm, LayerNorm
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import functools
import copy

from mmdet.models.backbones.resnet import BasicBlock, Bottleneck

from mmcv.cnn import ConvModule

from mmcv.cnn import constant_init, kaiming_init, bias_init_with_prob, normal_init
from torch.nn.modules.batchnorm import _BatchNorm
import inspect
import os.path as osp
import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mmcv.cnn import normal_init
from mmcv.runner import force_fp32
from mmcv.utils import _BatchNorm, _InstanceNorm
from mmcv.utils.ext_loader import check_ops_exist

from mmdet.core import (anchor_inside_flags,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')
ANCHOR_GENERATORS = Registry('Anchor generator')
OPTIMIZERS = Registry('optimizer')
OPTIMIZER_BUILDERS = Registry('optimizer builder')

FILTER_SIZE_MAP = {
    1: 32,
    2: 64,
    3: 128,
    4: 256,
    5: 256,
    6: 256,
    7: 256,
}

# The fixed SpineNet architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, block_fn, (input_offset0, input_offset1), is_output).
SPINENET_BLOCK_SPECS = [
    (2, Bottleneck, (None, None), False),  # init block
    (2, Bottleneck, (None, None), False),  # init block
    (2, Bottleneck, (0, 1), False),
    (4, BasicBlock, (0, 1), False),
    (3, Bottleneck, (2, 3), False),
    (4, Bottleneck, (2, 4), False),
    (6, BasicBlock, (3, 5), False),
    (4, Bottleneck, (3, 5), False),
    (5, BasicBlock, (6, 7), False),
    (7, BasicBlock, (6, 8), False),
    (5, Bottleneck, (8, 9), False),
    (5, Bottleneck, (8, 10), False),
    (4, Bottleneck, (5, 10), True),
    (3, Bottleneck, (4, 10), True),
    (5, Bottleneck, (7, 12), True),
    (7, Bottleneck, (5, 14), True),
    (6, Bottleneck, (12, 14), True),
]

SCALING_MAP = {
    '49S': {
        'endpoints_num_filters': 128,
        'filter_size_scale': 0.65,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '49': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '96': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 2,
    },
    '143': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 1.0,
        'block_repeats': 3,
    },
    '190': {
        'endpoints_num_filters': 512,
        'filter_size_scale': 1.3,
        'resample_alpha': 1.0,
        'block_repeats': 4,
    },
}

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.utils import print_log

# from mmdet.core.visualization import imshow_det_bboxes
from mmdet.utils import get_root_logger


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_scale=0.5,
                      font_size=13,
                      win_name='',
                      fig_size=(15, 10),
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_scale (float): Font scales of texts. Default: 0.5
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        fig_size (tuple): Figure size of the pyplot figure. Default: (15, 10)
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    warnings.warn('"font_scale" will be deprecated in v2.9.0,'
                  'Please use "font_size"')
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).copy()

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
        else:
            # specify  color
            mask_colors = [
                              np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
                          ] * (
                                  max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    img = np.ascontiguousarray(img)

    plt.figure(win_name, figsize=fig_size)
    plt.title(win_name)
    plt.axis('off')
    ax = plt.gca()

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    if out_file is not None:
        dir_name = osp.abspath(osp.dirname(out_file))
        mmcv.mkdir_or_exist(dir_name)
        plt.savefig(out_file)
        if not show:
            plt.close()
    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
            plt.close()
    return mmcv.rgb2bgr(img)


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = get_root_logger()
            print_log(f'load model from: {pretrained}', logger=logger)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_scale=0.5,
                    font_size=13,
                    win_name='',
                    fig_size=(15, 10),
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_scale (float): Font scales of texts. Default: 0.5
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            fig_size (tuple): Figure size of the pyplot figure.
                Default: (15, 10)
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_scale=font_scale,
            font_size=font_size,
            win_name=win_name,
            fig_size=fig_size,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img


from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn as nn


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


class BlockSpec(object):
    """A container class that specifies the block configuration for SpineNet."""

    def __init__(self, level, block_fn, input_offsets, is_output):
        self.level = level
        self.block_fn = block_fn
        self.input_offsets = input_offsets
        self.is_output = is_output


def build_block_specs(block_specs=None):
    """Builds the list of BlockSpec objects for SpineNet."""
    if not block_specs:
        block_specs = SPINENET_BLOCK_SPECS
    return [BlockSpec(*b) for b in block_specs]


class Resample(nn.Module):
    def __init__(self, in_channels, out_channels, scale, block_type, norm_cfg=dict(type="BN"), alpha=1.0):
        super(Resample, self).__init__()
        self.scale = scale
        new_in_channels = int(in_channels * alpha)
        if block_type == Bottleneck:
            in_channels *= 4
        self.squeeze_conv = ConvModule(in_channels, new_in_channels, 1, norm_cfg=norm_cfg)
        if scale < 1:
            self.downsample_conv = ConvModule(new_in_channels, new_in_channels, 3, padding=1, stride=2,
                                              norm_cfg=norm_cfg)
        self.expand_conv = ConvModule(new_in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def _resize(self, x):
        if self.scale == 1:
            return x
        elif self.scale > 1:
            return F.interpolate(x, scale_factor=self.scale, mode='nearest')
        else:
            x = self.downsample_conv(x)
            if self.scale < 0.5:
                new_kernel_size = 3 if self.scale >= 0.25 else 5
                x = F.max_pool2d(x, kernel_size=new_kernel_size, stride=int(0.5 / self.scale),
                                 padding=new_kernel_size // 2)
            return x

    def forward(self, inputs):
        feat = self.squeeze_conv(inputs)
        feat = self._resize(feat)
        feat = self.expand_conv(feat)
        return feat


class Merge(nn.Module):
    """Merge two input tensors"""

    def __init__(self, block_spec, norm_cfg, alpha, filter_size_scale):
        super(Merge, self).__init__()
        out_channels = int(FILTER_SIZE_MAP[block_spec.level] * filter_size_scale)
        if block_spec.block_fn == Bottleneck:
            out_channels *= 4
        self.block = block_spec.block_fn
        self.resample_ops = nn.ModuleList()
        for spec_idx in block_spec.input_offsets:
            spec = BlockSpec(*SPINENET_BLOCK_SPECS[spec_idx])
            in_channels = int(FILTER_SIZE_MAP[spec.level] * filter_size_scale)
            scale = 2 ** (spec.level - block_spec.level)
            self.resample_ops.append(
                Resample(in_channels, out_channels, scale, spec.block_fn, norm_cfg, alpha)
            )

    def forward(self, inputs):
        assert len(inputs) == len(self.resample_ops)
        parent0_feat = self.resample_ops[0](inputs[0])
        parent1_feat = self.resample_ops[1](inputs[1])
        target_feat = parent0_feat + parent1_feat
        return target_feat


def make_res_layer(**kwargs):
    return ResLayer(**kwargs)


import torch
import torch.nn as nn

from mmdet.core import bbox2result

from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list


from inspect import signature

import torch

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms


class BBoxTestMixin(object):
    """Mixin class for test time augmentation of bboxes."""

    def merge_aug_bboxes(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_factors = []  # score_factors for NMS
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            bbox_outputs = self.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            # bbox_outputs of some detectors (e.g., ATSS, FCOS, YOLOv3)
            # contains additional element to adjust scores before NMS
            if len(bbox_outputs) >= 3:
                aug_factors.append(bbox_outputs[2])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_factors = torch.cat(aug_factors, dim=0) if aug_factors else None
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            score_factors=merged_factors)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels, self.num_classes)
        return bbox_results


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]


@DETECTORS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)


@BACKBONES.register_module()
class SpineNet(nn.Module):
    """Class to build SpineNet backbone"""

    def __init__(self,
                 arch,
                 in_channels=3,
                 output_level=[3, 4, 5, 6, 7],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', require_grad=True),
                 zero_init_residual=True,
                 activation='relu'):
        super(SpineNet, self).__init__()
        self._block_specs = build_block_specs()[2:]
        self._endpoints_num_filters = SCALING_MAP[arch]['endpoints_num_filters']
        self._resample_alpha = SCALING_MAP[arch]['resample_alpha']
        self._block_repeats = SCALING_MAP[arch]['block_repeats']
        self._filter_size_scale = SCALING_MAP[arch]['filter_size_scale']
        self._init_block_fn = Bottleneck
        self._num_init_blocks = 2
        self.conv_cfg = conv_cfg

        self.norm_cfg = norm_cfg

        self.zero_init_residual = zero_init_residual
        assert min(output_level) > 2 and max(output_level) < 8, "Output level out of range"
        self.output_level = output_level
        self._make_stem_layer(in_channels)
        self._make_scale_permuted_network()
        self._make_endpoints()

    def _make_stem_layer(self, in_channels):
        """Build the stem network."""
        # Build the first conv and maxpooling layers.
        self.conv1 = ConvModule(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.init_block1 = ResLayer(self._init_block_fn, 64, int(FILTER_SIZE_MAP[2] * self._filter_size_scale),
                                    self._block_repeats, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

        self.init_block2 = ResLayer(self._init_block_fn, int(FILTER_SIZE_MAP[2] * self._filter_size_scale) * 4,
                                    int(FILTER_SIZE_MAP[2] * self._filter_size_scale), self._block_repeats,
                                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

    def _make_endpoints(self):
        self.endpoint_convs = nn.ModuleDict()
        for block_spec in self._block_specs:
            if block_spec.is_output:
                in_channels = int(FILTER_SIZE_MAP[block_spec.level] * self._filter_size_scale) * 4
                self.endpoint_convs[str(block_spec.level)] = ConvModule(in_channels,
                                                                        self._endpoints_num_filters,
                                                                        kernel_size=1,
                                                                        norm_cfg=self.norm_cfg,
                                                                        act_cfg=None)

    def _make_scale_permuted_network(self):
        self.merge_ops = nn.ModuleList()
        self.scale_permuted_blocks = nn.ModuleList()
        for spec in self._block_specs:
            self.merge_ops.append(
                Merge(spec, self.norm_cfg, self._resample_alpha, self._filter_size_scale)
            )
            channels = int(FILTER_SIZE_MAP[spec.level] * self._filter_size_scale)
            in_channels = channels * 4 if spec.block_fn == Bottleneck else channels
            self.scale_permuted_blocks.append(
                ResLayer(spec.block_fn, in_channels, channels, self._block_repeats, conv_cfg=self.conv_cfg,
                         norm_cfg=self.norm_cfg))

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, input):
        feat = self.maxpool(self.conv1(input))
        feat1 = self.init_block1(feat)
        feat2 = self.init_block2(feat1)
        block_feats = [feat1, feat2]
        output_feat = {}
        num_outgoing_connections = [0, 0]

        for i, spec in enumerate(self._block_specs):
            target_feat = self.merge_ops[i]([block_feats[feat_idx] for feat_idx in spec.input_offsets])
            # Connect intermediate blocks with outdegree 0 to the output block.
            if spec.is_output:
                for j, (j_feat, j_connections) in enumerate(
                        zip(block_feats, num_outgoing_connections)):
                    if j_connections == 0 and j_feat.shape == target_feat.shape:
                        target_feat += j_feat
                        num_outgoing_connections[j] += 1
            target_feat = F.relu(target_feat, inplace=True)
            target_feat = self.scale_permuted_blocks[i](target_feat)
            block_feats.append(target_feat)
            num_outgoing_connections.append(0)
            for feat_idx in spec.input_offsets:
                num_outgoing_connections[feat_idx] += 1
            if spec.is_output:
                output_feat[spec.level] = target_feat

        return [self.endpoint_convs[str(level)](output_feat[level]) for level in self.output_level]


@HEADS.register_module()
class AnchorHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 # anchor_generator=dict(
                 #     type='AnchorGenerator',
                 #     scales=[8, 16, 32],
                 #     ratios=[0.5, 1.0, 2.0],
                 #     strides=[4, 8, 16, 32, 64]),
                 # bbox_coder=dict(
                 #     type='DeltaXYWHBBoxCoder',
                 #     clip_border=True,
                 #     target_means=(.0, .0, .0, .0),
                 #     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 # reg_decoded_bbox=False,
                 # loss_cls=dict(
                 #     type='CrossEntropyLoss',
                 #     use_sigmoid=True,
                 #     loss_weight=1.0),
                 # loss_bbox=dict(
                 #     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,**kwargs):
        anchor_generator = dict(type='AnchorGenerator',
                                scales=[8,16,32], ratios=[0.5,1.,2.],
                                strides=[4, 8, 16, 32, 64])
        bbox_coder = dict(type='DeltaXYWHBBoxCoder', clip_border=True, target_means=(.0, .0, .0, .0),
                          target_stds=(1.0, 1.0, 1.0, 1.0))

        reg_decoded_bbox = False
        loss_cls=dict(type='CrossEntropyLoss',use_sigmoid=True,loss_weight=1.0)
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        # TODO better way to determine whether sample or not
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.anchor_generator = build_anchor_generator(anchor_generator)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list,)
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)


@HEADS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        anchor_generator = dict(type='AnchorGenerator', octave_base_scale=kwargs['octave_base_scale'],
                                scales_pre_octave=kwargs['scales_per_octave'], rations=kwargs['anchor_ratios'],
                                strides=kwargs['anchor_strides'])

        kwargs.pop('octave_base_scale')
        kwargs.pop('scales_per_octave')
        kwargs.pop('anchor_ratios')
        kwargs.pop('anchor_strides')
        bbox_coder = dict(type='DeltaXYWHBBoxCoder', clip_border=True, target_means=kwargs['target_means'],
                          target_stds=kwargs['target_stds']),
        kwargs.pop('target_means')
        kwargs.pop('target_stds')

        reg_decoded_bbox = False
        loss_cls = kwargs['loss_cls'],
        loss_bbox = kwargs['loss_bbox']
        kwargs.pop('loss_cls')
        kwargs.pop('loss_bbox')
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator, bbox_coder=bbox_coder, reg_decoded_bbox=reg_decoded_bbox,
            loss_cls=loss_cls, loss_bbox=loss_bbox,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred


@HEADS.register_module()
class RetinaSepBNHead(AnchorHead):
    """"RetinaHead with separate BN.

    In RetinaHead, conv/norm layers are shared across different FPN levels,
    while in RetinaSepBNHead, conv layers are shared across different FPN
    levels, but BN layers are separated.
    """

    def __init__(self,
                 num_classes,
                 num_ins,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_ins = num_ins
        anchor_generator = dict(type='AnchorGenerator', octave_base_scale=kwargs['octave_base_scale'],
                                scales_pre_octave=kwargs['scales_per_octave'], rations=kwargs['anchor_ratios'],
                                strides=kwargs['anchor_strides'])
        kwargs.pop('octave_base_scale')
        kwargs.pop('scales_per_octave')
        kwargs.pop('anchor_ratios')
        kwargs.pop('anchor_strides')
        bbox_coder = dict(type='DeltaXYWHBBoxCoder', clip_border=True, target_means=kwargs['target_means'],
                          target_stds=kwargs['target_stds']),
        kwargs.pop('target_means')
        kwargs.pop('target_stds')

        reg_decoded_bbox = False
        loss_cls = kwargs['loss_cls'],
        loss_bbox = kwargs['loss_bbox']
        kwargs.pop('loss_cls')
        kwargs.pop('loss_bbox')
        print("-" * 25, 'RetinaSepBNHead')
        for k, v in kwargs.items():
            print(k, v)
        super(RetinaSepBNHead, self).__init__(num_classes, in_channels, anchor_generator=anchor_generator,
                                              **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.num_ins):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        for i in range(self.stacked_convs):
            for j in range(1, self.num_ins):
                self.cls_convs[j][i].conv = self.cls_convs[0][i].conv
                self.reg_convs[j][i].conv = self.reg_convs[0][i].conv
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs[0]:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs[0]:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        for i, x in enumerate(feats):
            cls_feat = feats[i]
            reg_feat = feats[i]
            for cls_conv in self.cls_convs[i]:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs[i]:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.retina_cls(cls_feat)
            bbox_pred = self.retina_reg(reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox


@ANCHOR_GENERATORS.register_module()
class AnchorGenerator(object):
    """Standard anchor generator for 2D anchor-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_sizes, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        feat_h, feat_w = featmap_size
        # convert Tensor to int, so that we can covert to ONNX correctlly
        feat_h = int(feat_h)
        feat_w = int(feat_w)
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps.
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str



@OPTIMIZER_BUILDERS.register_module()
class DefaultOptimizerConstructor:
    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        self.base_lr = optimizer_cfg.get('lr')
        self.base_wd = optimizer_cfg.get('weight_decay')
        self._validate_cfg()

    def _validate_cfg(self):
        if not isinstance(self.paramwise_cfg, dict):
            raise TypeError('paramwise_cfg should be None or a dict, '
                            f'but got {type(self.paramwise_cfg)}')

        if 'custom_keys' in self.paramwise_cfg:
            if not isinstance(self.paramwise_cfg['custom_keys'], dict):
                raise TypeError(
                    'If specified, custom_keys must be a dict, '
                    f'but got {type(self.paramwise_cfg["custom_keys"])}')
            if self.base_wd is None:
                for key in self.paramwise_cfg['custom_keys']:
                    if 'decay_mult' in self.paramwise_cfg['custom_keys'][key]:
                        raise ValueError('base_wd should not be None')

        # get base lr and weight decay
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in self.paramwise_cfg
                or 'norm_decay_mult' in self.paramwise_cfg
                or 'dwconv_decay_mult' in self.paramwise_cfg):
            if self.base_wd is None:
                raise ValueError('base_wd should not be None')

    def _is_in(self, param_group, param_group_list):
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
                isinstance(module, torch.nn.Conv2d)
                and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(f'{prefix} is duplicate. It is skipped since '
                              f'bypass_duplicate={bypass_duplicate}')
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == 'bias' and not (is_norm or is_dcn_module):
                    param_group['lr'] = self.base_lr * bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group[
                            'weight_decay'] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group[
                            'weight_decay'] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == 'bias' and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group[
                            'weight_decay'] = self.base_wd * bias_decay_mult
            params.append(param_group)

        if check_ops_exist():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        print(optimizer_cfg)
        # if no paramwise option is specified, just use the global setting
        if self.paramwise_cfg!={}:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params = list(optimizer_cfg.keys())
        print(params)
        self.add_params(params, model)
        optimizer_cfg['params'] = params
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@OPTIMIZERS.register_module()
class SGD(DefaultOptimizerConstructor):
    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        super(DefaultOptimizerConstructor,self).__init__(
        optimizer_cfg=optimizer_cfg,paramwise_cfg=paramwise_cfg)
    def _validate_cfg(self):
        if not isinstance(self.paramwise_cfg, dict):
            raise TypeError('paramwise_cfg should be None or a dict, '
                            f'but got {type(self.paramwise_cfg)}')

        if 'custom_keys' in self.paramwise_cfg:
            if not isinstance(self.paramwise_cfg['custom_keys'], dict):
                raise TypeError(
                    'If specified, custom_keys must be a dict, '
                    f'but got {type(self.paramwise_cfg["custom_keys"])}')
            if self.base_wd is None:
                for key in self.paramwise_cfg['custom_keys']:
                    if 'decay_mult' in self.paramwise_cfg['custom_keys'][key]:
                        raise ValueError('base_wd should not be None')

        # get base lr and weight decay
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in self.paramwise_cfg
                or 'norm_decay_mult' in self.paramwise_cfg
                or 'dwconv_decay_mult' in self.paramwise_cfg):
            if self.base_wd is None:
                raise ValueError('base_wd should not be None')

    def _is_in(self, param_group, param_group_list):
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
                isinstance(module, torch.nn.Conv2d)
                and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(f'{prefix} is duplicate. It is skipped since '
                              f'bypass_duplicate={bypass_duplicate}')
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == 'bias' and not (is_norm or is_dcn_module):
                    param_group['lr'] = self.base_lr * bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group[
                            'weight_decay'] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group[
                            'weight_decay'] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == 'bias' and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group[
                            'weight_decay'] = self.base_wd * bias_decay_mult
            params.append(param_group)

        if check_ops_exist():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        print(optimizer_cfg)
        # if no paramwise option is specified, just use the global setting
        if self.paramwise_cfg!={}:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params = list(optimizer_cfg.keys())
        print(params)
        self.add_params(params, model)
        optimizer_cfg['params'] = params
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)



def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss
def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def build(cfg, registry, default_args=None):
    return build_from_cfg(cfg, registry, default_args)

def build_anchor_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)

def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)


def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    print(constructor_type)
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer