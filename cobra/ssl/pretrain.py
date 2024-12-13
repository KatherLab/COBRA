"""
Adapted from the official MoCoV3 implementation: https://github.com/facebookresearch/moco-v3
@Article{chen2021mocov3,
author  = {Xinlei Chen* and Saining Xie* and Kaiming He},
title   = {An Empirical Study of Training Self-Supervised Vision Transformers},
journal = {arXiv preprint arXiv:2104.02057},
year    = {2021},
}
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch 
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import os
import builtins
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import argparse
import yaml
from jinja2 import Environment, FileSystemLoader
from pprint import pprint
from datetime import datetime
import math

from cobra.ssl.model import MoCo
from cobra.ssl.data import FeatDataset, get_pat_dict


def main(args,cfg):

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args,cfg))


def main_worker(gpu, ngpus_per_node, args,cfg):

    args.gpu = gpu
    args.rank = gpu
    print(f"{args.gpu=}")
    if (args.gpu != 0 or args.rank != 0): 
        def print_pass(*args):
            pass
        builtins.print = print_pass

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

    print("=> creating model...")

    model = MoCo(embed_dim=cfg["model"]["dim"], c_dim=cfg["model"]["l_dim"],
                 num_heads=cfg["model"]["nr_heads"],
                 gpu_id=args.gpu,T=cfg["ssl"]["moco_t"],
                 nr_mamba_layers=cfg["model"]["nr_mamba_layers"],dropout=cfg["model"]["dropout"]) 
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    
    # infer learning rate before changing batch size
    args.lr = float(cfg["ssl"]["lr"]) * cfg["ssl"]["batch_size"] / 256
    
    args.batch_size = int(cfg["ssl"]["batch_size"] / args.world_size)
    args.workers = int((cfg["ssl"]["workers"] + ngpus_per_node - 1) / ngpus_per_node)
    args.warmup = cfg["ssl"]["warmup_epochs"]
    model_params = sum(p.numel() for p in model.base_enc.parameters())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
   
    optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=cfg["ssl"]["weight_decay"])
    scaler = torch.amp.GradScaler('cuda')

    dataset = FeatDataset(pat_dict=get_pat_dict(cfg),num_feats=cfg["general"]["nr_feats"],
                          feat_len=1536)

    print(f"number of training samples = {len(dataset)}")
    print(f"# base_enc model params: {model_params}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, 
                        num_workers=args.workers,drop_last=True, pin_memory=True,)
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    cudnn.benchmark = True

    model.train()

    iters_per_epoch = len(loader)

    #print_freq = 10

    for e in tqdm(range(args.start_epoch,cfg["ssl"]["epochs"]),disable=args.rank!=0):
        
        t_loss = 0.0

        for i, (x1, sizes1, x2, sizes2) in enumerate(tqdm(loader,leave=False,disable=args.rank!=0)):
        
            lr = adjust_learning_rate(optimizer, e + i / iters_per_epoch, args,cfg)
            moco_m = adjust_moco_momentum(e + i / iters_per_epoch, cfg)

            x1 = x1.to(torch.float32).cuda()
            x2 = x2.to(torch.float32).cuda()

            sizes1 = sizes1.cuda()
            sizes2 = sizes2.cuda()
            loss = model(x1,x2,sizes_1=sizes1,sizes_2=sizes2,m=moco_m)
            t_loss += loss.item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if args.rank == 0:
            tqdm.write(f"Epoch {e+1}; loss: {t_loss/len(loader):.4f}; lr: {lr:.5f}")
            if (e+1)%5==0:
                state = {
                    'epoch': e + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 
                }
                torch.save(state,os.path.join(cfg["general"]["paths"]["chkpt_dir"],
                                                        f'{cfg["model"]["model_name"]}-{e+1}.pth.tar'))

def adjust_learning_rate(optimizer, epoch, args,cfg):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg["ssl"]["warmup_epochs"]:
        lr = args.lr * epoch / cfg["ssl"]["warmup_epochs"] 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - cfg["ssl"]["warmup_epochs"]) / (cfg["ssl"]["epochs"] - cfg["ssl"]["warmup_epochs"])))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_moco_momentum(epoch,cfg):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / cfg["ssl"]["epochs"])) * (1. - cfg["ssl"]["moco_m"])
    return m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cobra-training.')

    # Add the command-line argument for the config path
    parser.add_argument('--config', type=str, default='config.yml', 
                        help='Path to the config file')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:23459', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg_data = yaml.safe_load(f)

    template_env = Environment(loader=FileSystemLoader(searchpath='./'))
    template = template_env.from_string(str(cfg_data))
    # Render the template with the values from the config_data
    cfg = yaml.safe_load(template.render(**cfg_data))

    exp_str = datetime.now().strftime("%Y-%m-%d-%H:%M")

    cfg["general"]["paths"]["out_dir"] = os.path.join(cfg["general"]["paths"]["out_dir"],exp_str)
    cfg["general"]["paths"]["chkpt_dir"] = os.path.join(cfg["general"]["paths"]["out_dir"],"checkpoints")
    pprint(cfg)

    #for k in cfg.keys():
    for path in cfg["general"]["paths"].values():
            Path(path).mkdir(parents=True,exist_ok=True)

    with open(os.path.join(cfg["general"]["paths"]["out_dir"],"config.yml"),"w") as f:
        yaml.dump(cfg,f,sort_keys=False,default_flow_style=False)
    
    main(args,cfg)
