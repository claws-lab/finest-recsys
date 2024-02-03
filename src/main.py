#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2024-02-01
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
from LSTM import *
from collections import Counter

def filter_and_split(data,args):

    (users,counts) = np.unique(data[:,0],return_counts = True)
    
    users = users[counts>=10]

    sequence_dic,pert_dic =  {int(user):[] for user in set(data[:,0])}, {int(user):[] for user in set(data[:,0])}
    
    user_dic = {int(user):idx for (idx,user) in enumerate(users)}
    new_data = []
    for i in range(data.shape[0]):
        if int(data[i,0]) in user_dic:
            new_data.append([int(data[i,0]),int(data[i,1]),data[i,2],0])

    new_data = np.array(new_data)

    for i in range(new_data.shape[0]):
        sequence_dic[int(new_data[i,0])].append([i,int(new_data[i,1]),new_data[i,2]])

    for user in sequence_dic.keys():
        cur_test = int(0.1*len(sequence_dic[user]))
        for i in range(cur_test):
            interaction = sequence_dic[user].pop()
            new_data[interaction[0],3] = 1
    
    new_data = new_data[np.argsort(new_data[:,2]),:]
    print(data.shape,new_data.shape)
    return new_data

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

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
 
    output_path = args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    raw_data = pd.read_csv(args.data_path, sep='\t', header=None)
    # to ensure the input data format is (user, item, timestamp)
    data = raw_data.values[:,-3:] 
    # window size used for LSTM prediction
    look_back = 50
    
    unique_users = sorted(list(set(data[:, 0])))
    unique_items = sorted(list(set(data[:, 1])))
    user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
    item_dic = {item:idx for (idx,item) in enumerate(unique_items)}
    for (idx, row) in enumerate(data):
        user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
        data[idx,0],data[idx,1] = int(user),int(item)
 
    original_perf,fine_tuning_perf = [],[]
    f = open(output_path,'w')

    original_data = filter_and_split(data,args)

    # number of training data perturbations for testing the fine-tuning performance of FINEST
    num_pert = int(np.count_nonzero(original_data[:,3]==0)*args.perturbation_ratio)

    # original indices are needed to append to the original data to track what training data is perturbed during the simulation
    original_data = np.append(original_data,np.arange(0,original_data.shape[0]).reshape(-1,1),axis=1)

    model = LSTM(data = original_data,input_size=128, output_size=len(unique_items)+1, hidden_dim=128, n_layers=1, device=device,seed=0,look_back=look_back).to(device)
    model.LSTM.flatten_parameters()
    original_model = copy.deepcopy(model)
    #train a recommendation model with the original training data
    [original_logits,reference_list,competitive_items, temp] = model.traintest(original_logits=-1, reference_list=-1,competitive_items=-1, args=args,data = original_data,mode="original_training")
    
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
    
    #train a recommendation model with the perturbed training data
    original_model.LSTM.flatten_parameters()
    [temp1,temp2,temp3,original_perf] = original_model.traintest(original_logits=original_logits, reference_list=-1,competitive_items=-1, args=args,data = perturbed_data,mode="retraining")
    
    #fine-tune the trained recommendation model with the perturbed training data
    [temp1,temp2,temp3, fine_tuning_perf] =  model.traintest(original_logits = original_logits, reference_list=reference_list,competitive_items=competitive_items, args=args,data = perturbed_data,mode="fine_tuning")
   
    print('[Normal training performance against training data perturbation] RBO = {}\tJaccard@10 = {}\tMRR = {}\tHITS@10 = {}'.format(original_perf[2],original_perf[3],original_perf[0],original_perf[1]),file=f,flush=True)
    print('[FINEST fine-tuning performance against training data perturbation] RBO = {}\tJaccard@10 = {}\tMRR = {}\tHITS@10 = {}'.format(fine_tuning_perf[2],fine_tuning_perf[3],fine_tuning_perf[0],fine_tuning_perf[1]),file=f,flush=True)

if __name__ == "__main__":
    main()
