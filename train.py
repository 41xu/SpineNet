import argparse
import copy
import os
# osp: os.path
import time
import warnings
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info,init_dist
from mmcv.utils import Registry
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import set_random_seed

from build_model import build_model,build_optimizer
from mmdet.utils import collect_env,get_root_logger
from mmdet.core import DistEvalHook, EvalHook
import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook)

from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='./config/spinenet_49S_B_8gpu.py', help='train config file path')
    parser.add_argument('--work-dir', default='./work_dirs/spinenet_49S_B/', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from',default='./checkpoints', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        # default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        # default='0',
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def main():
    args=parse_args()
    # print(args)
    cfg=Config.fromfile(args.config)
    # print(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark',False):
        # print("set cudnn_benchmark")
        torch.backends.cudnn.benchmark=True
    # set work_dir
    if args.work_dir is not None:
        # print("set work_dir")
        cfg.work_dir=args.work_dir
    elif cfg.get('work_dir',None) is not None:
        cfg.work_dir=os.path.join('./work_dirs',os.path.splitext(os.path.basename(args.config))[0])
    # set resume_from
    if args.resume_from is not None:
        cfg.resume_from=args.resume_from
    # set gpu
    if args.gpu_ids is not None:
        cfg.gpu_ids=args.gpu_ids
    else:
        cfg.gpu_ids=range(1) if args.gpus is None else range(args.gpus)
    # set distributed env
    if args.launcher=='none':
        distributed=False
    else:
        # 其实这里默认single gpu，我感觉这里直接设置ids=ids就可以，word_size可以省略
        distributed=True
        # init_dist(args.launcher,**cfg.dist_params)
        _,word_size=get_dist_info()
        #print(word_size)
        cfg.gpu_ids=range(word_size)
    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config
    cfg.dump(os.path.join(cfg.work_dir,os.path.basename(args.config)))
    # init logger
    timestamp=time.strftime("%Y%m%d_%H%M%S",time.localtime())
    log_file=os.path.join(cfg.work_dir,f'{timestamp}.log')
    logger=get_root_logger(log_file=log_file,log_level=cfg.log_level)
    # init something tobe logged
    meta=dict()
    env_info_dict=collect_env()
    env_info='\n'.join([(f'{k}:{v}') for k,v in env_info_dict.items()])
    dash_line='-'*60+'\n'
    logger.info('Environment info:\n'+dash_line+env_info+'\n'+dash_line)
    meta['env_info']=env_info
    meta['config']=cfg.pretty_text
    logger.info(f'Distributed training:{distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random sees
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic:{args.deterministic}')
        set_random_seed(args.seed,deterministic=args.deterministic)
    cfg.seed=args.seed
    meta['seed']=args.seed
    meta['exp_name']=os.path.basename(args.config)

    model=build_model(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    datasets=[build_dataset(cfg.data.train)]

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta=dict(mmdet_version=__version__+get_git_hash()[:7],CLASSES=datasets[0].CLASSES)
    # model.CLASSES=datasets[0].CLASSES
    print(model)
    train_detector(model,datasets,cfg,distributed=distributed,validate=(not args.no_validate),meta=meta,timestamp=timestamp)


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    print(cfg)
    print("-"*20)
    print(cfg.optimizer)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)




if __name__ == '__main__':
    main()
