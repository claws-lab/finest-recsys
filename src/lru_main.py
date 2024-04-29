#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2024-04-26
#FINEST: Stabilizing Recommendations by Rank-Preserving Fine-Tuning
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import torch
import torch.nn as nn
import numpy as np
import pandas as  pd
import copy
import time
import argparse
import os
import rbo
from lru_model import *
from lru_trainer import *
from utils import *
from collections import Counter

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help='path of the dataset')
    parser.add_argument('--gpu',default='0',type=str,help='GPU# will be used')
    parser.add_argument('--attack_kind',type=str,default='deletion', help = "deletion, Replacement, Injection attack")
    parser.add_argument('--output',type=str, default = 'lstm_output.txt', help = "Output file path")
    parser.add_argument('--perturbation_ratio', default=0.001, type = float, help='training data perturbation ratio for testing the fine-tuning performance')
    parser.add_argument('--simulation_ratio', default=0.01, type = float, help='simulated perturbation ratio for fine-tuning')
    parser.add_argument('--training_epoch', default=50, type = int, help='number of training epochs')
    parser.add_argument('--fine_tuning_epoch', default=50, type = int, help='number of fine-tuning epochs')
    parser.add_argument('--topk', default=100, type = int, help='number of items that will be used for regularization')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for training')
    parser.add_argument('--fine_tune_lr', default=0.0001, type=float, help='learning rate for fine-tuning using FINEST')
    parser.add_argument('--bert_max_len', type=int, default=50)
    parser.add_argument('--bert_hidden_units', type=int, default=128)
    parser.add_argument('--bert_num_blocks', type=int, default=2)
    parser.add_argument('--bert_num_heads', type=int, default=2)
    parser.add_argument('--bert_head_size', type=int, default=32)
    parser.add_argument('--bert_dropout', type=float, default=0.2)
    parser.add_argument('--bert_attn_dropout', type=float, default=0.2)
    parser.add_argument('--bert_mask_prob', type=float, default=0.2)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
  
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
 
    output_path = args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    raw_data = pd.read_csv(args.data_path, sep='\t', header=None)
    # to ensure the input data format is (user, item, timestamp)
    data = raw_data.values[:,-3:] 
    
    unique_users = sorted(list(set(data[:, 0])))
    unique_items = sorted(list(set(data[:, 1])))
    user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
    item_dic = {item:idx for (idx,item) in enumerate(unique_items)}
    for (idx, row) in enumerate(data):
        user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
        data[idx,0],data[idx,1] = int(user),int(item)
 
    original_perf,fine_tuning_perf = [],[]
    f = open(output_path,'w')

    # filter out users with less than 10 interactions and create training/test data
    original_data,test_len = filter_and_split(data)

    # number of training data perturbations for testing the fine-tuning performance of FINEST
    num_pert = int(np.count_nonzero(original_data[:,3]==0)*args.perturbation_ratio)

    # original indices are needed to append to the original data to track what training data is perturbed during the simulation
    original_data = np.append(original_data,np.arange(0,original_data.shape[0]).reshape(-1,1),axis=1)

    original_dataset = data_partition(original_data)
    [user_train,  user_test, usernum, itemnum, timenum,user_map,item_map] = original_dataset 
    args.num_users = usernum
    args.num_items = itemnum

    # initialize the LRURec model
    model = LRU(args)
    trainer = LRUTrainer(usernum, itemnum, args,device,np.random.randint(0,99999999),original_dataset,model).to(device)    
    original_trainer = copy.deepcopy(trainer)

    print('\nPre-training of LRURec + fine-tuning with FINEST on the original training data\n')
    [original_logits,reference_list,competitive_items, original_perf] = trainer.traintest(original_logits=-1, reference_list=-1,competitive_items=-1, args=args,data = original_data, dataset = original_dataset,mode="original_training")
    
    #perform random training data perturbation
    candidates = []
    for i in range(original_data.shape[0]):
        if original_data[i,3]==0:
            candidates.append(i)

    chosen = np.random.choice(candidates,size = num_pert,replace=False)
    if args.attack_kind=='deletion':
        perturbed_data = np.delete(original_data,chosen,axis=0)
    elif args.attack_kind=='insertion':
        values = []
        for idx in chosen:
            user,item,timee = int(original_data[idx,0]),int(original_data[idx,1]),original_data[idx,2]
            replacement = np.random.choice(list(set(original_data[:,1]))) 
            values.append([user,replacement,timee-1,0])
        perturbed_data = np.insert(original_data,chosen,values,axis=0)
    else:
        perturbed_data = copy.deepcopy(original_data)
        for idx in chosen:
            replacement = np.random.choice(list(set(original_data[:,1])))
            perturbed_data[idx,1] = replacement
 
    dataset = data_partition(perturbed_data)
    [user_train,  user_test, usernum, itemnum, timenum,user_map,item_map] = dataset 


    print('\nRetraining of LRURec + fine-tuning with FINEST on the perturbed training data\n')
    [temp1,temp2,temp3,perturbed_perf] = original_trainer.traintest(original_logits=original_logits, reference_list=reference_list,competitive_items=competitive_items, args=args,data = perturbed_data, dataset = dataset,mode="retraining")
    
    print('[FINEST performance on original training data] MRR = {}\tRecall@10 = {}\tNDCG@10 = {}'.format(original_perf[0],original_perf[1],original_perf[2]),file=f,flush=True)
    print('[FINEST performance on perturbed data perturbation] MRR = {}\tRecall@10 = {}\tNDCG@10 = {}\nRBO = {}\tJaccard@10 = {}'.format(perturbed_perf[0],perturbed_perf[1],perturbed_perf[2],perturbed_perf[3],perturbed_perf[4]),file=f,flush=True)

if __name__ == "__main__":
    main()
