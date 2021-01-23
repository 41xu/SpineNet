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
from mmdet.apis import set_random_seed,train_detector
from mmdet.datasets import build_dataset
#from mmdet.models import build_detector
from build_model import build_model
from mmdet.utils import collect_env,get_root_logger


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
        default='pytorch',
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
        print("set cudnn_benchmark")
        torch.backends.cudnn.benchmark=True
    # set work_dir
    if args.work_dir is not None:
        print("set work_dir")
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
    print(datasets,datasets[0])
    model['CLASSES']=datasets[0]['CLASSES']
    train_detector(model,datasets,cfg,distributed=distributed,validate=(not args.no_validate),meta=meta,timestamp=timestamp)






if __name__ == '__main__':
    main()
