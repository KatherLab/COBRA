import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
from glob import glob
from tqdm import tqdm
import pathlib
from concurrent.futures import ThreadPoolExecutor


class FeatDataset(Dataset):
    def __init__(self,pat_dict,num_feats=600,feat_len=1536):
        self.pat_dict = pat_dict
        self.pat_list = list(self.pat_dict.keys())
        
        print(f"Found {len(self.pat_list)} patient ids.")
        self.num_feats = num_feats
        self.feat_len = feat_len

    def __len__(self):
        return len(self.pat_list)

    def __getitem__(self, idx):
        
        pat = self.pat_list[idx]
        
        idx1 = np.random.randint(0,len(self.pat_dict[pat]))
        idx2 = np.random.randint(0,len(self.pat_dict[pat]))
        
        try:
            with h5py.File(self.pat_dict[pat][idx1],'r') as f:
                feats1 = f["feats"][:]
        except Exception as e:
            print(f"Error reading {self.pat_dict[pat][idx1]}")
            raise e
        
        try:
            with h5py.File(self.pat_dict[pat][idx2],'r') as f:
                feats2 = f["feats"][:]
        except Exception as e:
            print(f"Error reading {self.pat_dict[pat][idx2]}")
            raise e
    
        assert len(feats1.shape)==2, f"{feats1.shape=}!"
        assert len(feats2.shape)==2, f"{feats2.shape=}!"
        
        orig_size_1 = feats1.shape[-1]
        orig_size_2 = feats2.shape[-1]
        
        with torch.no_grad():
            feats1 = self.pad_or_sample(torch.tensor(feats1),self.num_feats,self.feat_len)
            feats2 = self.pad_or_sample(torch.tensor(feats2),self.num_feats,self.feat_len)

        assert len(feats1.shape)==2 and feats1.shape[0]==self.num_feats and feats1.shape[1]==self.feat_len, f"{feats1.shape=}"
        assert len(feats2.shape)==2 and feats2.shape[0]==self.num_feats and feats2.shape[1]==self.feat_len, f"{feats2.shape=}"

        return feats1, torch.tensor(orig_size_1), feats2, torch.tensor(orig_size_2)

    def pad_or_sample(self, x: torch.Tensor,n=1024,k=1536) -> torch.Tensor:
        length = x.shape[0]
        x = x[torch.randperm(len(x))][:n]
        if length < n:
            repeats = (n - length) // length
            tmp = x
            for _ in range(repeats):
                x = torch.cat([x,tmp[torch.randperm(length)]])
            resample_size = (n - length) % length
            if resample_size > 0:
                x = torch.cat([x,x[torch.randperm(len(x))][:resample_size]])
        feat_len = x.shape[1]
        if k-feat_len>0:
            pad_size = k-feat_len
            x = torch.cat([x,torch.zeros(n,pad_size)],dim=1)
        return x


def check_file(f):
    try:
        with h5py.File(f,'r') as h5f:
            if "feats" not in h5f:
                raise KeyError(f"'feats' not found in {f}")
            #feats = f["feats"][:]
    except Exception as e:
        print(f"Error reading {f}")
        raise e
    #assert len(feats.shape)==2, f"{feats.shape=}!"

def get_pat_dict(cfg,num_cores=14):
    
    pat_dict = {}
    print(f'FMs: {cfg["general"]["fms"]}')
    #for c in tqdm(cfg["general"]["feat_cohorts"]):
    for c in tqdm(os.listdir(os.path.join(cfg["general"]["feat_base_paths"][0],cfg["general"]["fms"][0])),leave=False):
        for fm in tqdm(cfg["general"]["fms"],leave=False):
            for feat_base_path in cfg["general"]["feat_base_paths"]:
                feat_path = os.path.join(feat_base_path,fm,c)
                feat_path = os.path.join(feat_path, [f for f in os.listdir(feat_path) if "stamp" in f and os.path.isdir(os.path.join(feat_path, f))][0])
                feat_files = glob(os.path.join(feat_path,"*.h5"))
                assert len(feat_files)>0, f"couldnt find any feat files in path {feat_path}"
                def process_file(f):
                    pat_id = pathlib.Path(f).stem[:12]
                    check_file(f)
                    return pat_id, f

                # num_cores = cfg.get("num_cores", None)
                with ThreadPoolExecutor(max_workers=num_cores) as executor:
                    results = list(tqdm(executor.map(process_file, feat_files), total=len(feat_files), leave=False))

                for pat_id, f in results:
                    if pat_id in pat_dict:
                        pat_dict[pat_id].append(f)
                    else:
                        pat_dict[pat_id] = [f]
    print(f"Found {sum([len(list(v)) for _,v in pat_dict.items()])} feature paths")
    return pat_dict