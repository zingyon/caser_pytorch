import copy
import random
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import HfArgumentParser
import os
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()

def make_dataset_class(datas):
    uids = []
    sids = []
    
    for uid, sid in datas.items():
        uids.append(uid)
        sids.append([s+1 for s in sid])
    
    return Dataset.from_dict({
        "uid": uids,
        "sid": sids
    })


# def create_datasets(input_file, min_rating:int=4, min_sc:int=0, min_uc:int=5, split:str="leave_one_out"):
#     df = pd.read_csv(input_file, sep='::', header=None)
#     df.columns = ['uid', 'sid', 'rating', 'timestamp']
#     df = df[df["rating"] >= min_rating]
    
#     if min_sc > 0:
#         item_sizes = df.groupby('sid').size()
#         good_items = item_sizes.index[item_sizes >= min_sc]
#         df = df[df['sid'].isin(good_items)]

#     if min_uc > 0:
#         user_sizes = df.groupby('uid').size()
#         good_users = user_sizes.index[user_sizes >= min_uc]
#         df = df[df['uid'].isin(good_users)]


#     umap = {u: i for i, u in enumerate(set(df['uid']))}
#     smap = {s: i for i, s in enumerate(set(df['sid']))}
#     df['uid'] = df['uid'].map(umap)
#     df['sid'] = df['sid'].map(smap)
    
#     user_count = len(umap)
        
#     if split == 'leave_one_out':
    
#         user_group = df.groupby('uid')
#         user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
#         train, val, test = {}, {}, {}
#         for user in range(user_count):
#             items = user2items[user]
#             train[user], val[user], test[user] = items[:-2], items[:-1], items[:]
#         #test set에서 맨 마지막 mask된 1개(sid)를 유추하는데, train set의 sid들을 test set의 마지막 앞(SID)까지 채워줌
    
#     elif split == 'holdout':
#         eval_set_size = 500 # 추후 조정 필요

#         # Generate user indices
#         permuted_index = np.random.permutation(user_count)
#         train_user_index = permuted_index[:-2*eval_set_size]
#         val_user_index   = permuted_index[-2*eval_set_size: -eval_set_size]
#         test_user_index  = permuted_index[-eval_set_size:]

#         # Split DataFrames
#         train_df = df.loc[df['uid'].isin(train_user_index)]
#         val_df   = df.loc[df['uid'].isin(val_user_index)]
#         test_df  = df.loc[df['uid'].isin(test_user_index)]

#         # DataFrame to dict => {uid : list of sid's}
#         train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
#         val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
#         test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        
#     else:
#         raise NotImplementedError


#     train_dataset = make_dataset_class(train)
#     val_dataset = make_dataset_class(val)
#     test_dataset = make_dataset_class(test)
    
#     return train_dataset, val_dataset, test_dataset, smap


def create_datasets(input_file, min_rating:int=4, min_sc:int=0, min_uc:int=5, split:str="leave_one_out"):
    with open(input_file, "r") as f:
      data = f.readlines()
        
    if split == 'leave_one_out':
        smap = []
        train, val, test = {}, {}, {}
        for user in range(len(data)):
            data[user] = data[user][1:]
            items = [int(i) for i in data[user].strip("\n").split(" ") if i]
            train[user], val[user], test[user] = items[:-2], items[:-1], items[:]
            smap += items
        #test set에서 맨 마지막 mask된 1개(sid)를 유추하는데, train set의 sid들을 test set의 마지막 앞(SID)까지 채워줌
    
    elif split == 'holdout':
        eval_set_size = 500 # 추후 조정 필요

        # Generate user indices
        permuted_index = np.random.permutation(user_count)
        train_user_index = permuted_index[:-2*eval_set_size]
        val_user_index   = permuted_index[-2*eval_set_size: -eval_set_size]
        test_user_index  = permuted_index[-eval_set_size:]

        # Split DataFrames
        train_df = df.loc[df['uid'].isin(train_user_index)]
        val_df   = df.loc[df['uid'].isin(val_user_index)]
        test_df  = df.loc[df['uid'].isin(test_user_index)]

        # DataFrame to dict => {uid : list of sid's}
        train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        
    else:
        raise NotImplementedError


    train_dataset = make_dataset_class(train)
    val_dataset = make_dataset_class(val)
    test_dataset = make_dataset_class(test)
    
    return train_dataset, val_dataset, test_dataset, list(set(smap))



def tuncate_and_padding(dataset, is_train, max_len):
    
    def _map(x):
        seq_var = x["sid"][:-1]
        target = x["sid"][-1]
        user_var = x["uid"]
        
        length = min(len(seq_var), max_len)
        
        seq_var = seq_var[-max_len:] + [0] * (max_len - length)
        
        
        x["seq_var"] = seq_var
        x["user_var"] = user_var
        x["labels"] = target
        
        return x
    
    
    
    dataset = dataset.map(_map, batched=False, remove_columns=dataset.column_names, num_proc=4)
    return dataset