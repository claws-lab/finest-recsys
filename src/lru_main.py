#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.1
#@date       2024-07-05
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
    parser.add_argument('--attack_type',type=str,default='random', help = "Random or CASPER attack")
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
 
    f = open(output_path,'w')
    avg_original_perf,avg_perturbed_perf = [[],[],[]],[[],[],[],[],[]]
    
    for seed in range(3):
        print('\nExperiment with random seed: {}\n'.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        
        # filter out users with less than 10 interactions and create training/test data
        original_data,test_len = filter_and_split(data)

        # number of training data perturbations for testing the fine-tuning performance of FINEST
        num_pert = int(np.count_nonzero(original_data[:,3]==0)*args.perturbation_ratio)

        #CASPER attack implementation. Refer to CASPEr paper for details
        if args.attack_type=='casper':
            in_degree, num_child = np.zeros(original_data.shape[0]),np.zeros(original_data.shape[0])
            cutoff = args.bert_max_len
            user_dic,item_dic = defaultdict(list),defaultdict(list)
            edges = defaultdict(list)
            count = 0
            user_seq = {user:[0 for i in range(cutoff)] for user in set(original_data[:,0])}
            for i in range(original_data.shape[0]):
                in_degree[i]=-1
                if original_data[i,3]==0:
                    count += 1
                    user,item = int(original_data[i,0]),int(original_data[i,1])
                    user_dic[user].append(i)
                    item_dic[item].append(i)
                    user_seq[user] = user_seq[user][1:]+[i]
                    in_degree[i] = 0

            valid = {}
            for user in user_seq:
                current_seq = user_seq[user]
                for i in range(cutoff):
                    valid[current_seq[i]]=1

            for user in user_dic.keys():
                cur_list = user_dic[user]
                for i in range(len(cur_list)-1):
                    j,k = cur_list[i],cur_list[i+1]
                    if j in valid and k in valid:
                        in_degree[k] += 1
                        edges[j].append(k)

            for item in item_dic.keys():
                cur_list = item_dic[item]
                for i in range(len(cur_list)-1):
                    j,k = cur_list[i],cur_list[i+1]
                    if j in valid and k in valid:
                        in_degree[k] += 1
                        edges[j].append(k)

            queue = []
            for i in range(original_data.shape[0]):
                if in_degree[i] == 0 and  i in valid:
                    queue.append(i)
            
            while len(queue)!=0:
                root = queue.pop(0)
                check = np.zeros(original_data.shape[0])
                check[root]=1
                q2 = [root]
                count2 = 1
                while len(q2)!=0:
                    now = q2.pop(0)
                    for node in edges[now]:
                        if check[node]==0:
                            check[node]=1
                            q2.append(node)
                            count2 += 1
                num_child[root] = count2


        # original indices are needed to append to the original data to track what training data is perturbed during the simulation
        original_data = np.append(original_data,np.arange(0,original_data.shape[0]).reshape(-1,1),axis=1)

        original_dataset = data_partition(original_data)
        [user_train,  user_test, usernum, itemnum, timenum,user_map,item_map] = original_dataset 
        args.num_users = usernum
        args.num_items = itemnum


        # initialize the LRURec model
        model = LRU(args)
        trainer = LRUTrainer(usernum, itemnum, args,device,original_dataset,model).to(device)    
        original_trainer = copy.deepcopy(trainer)

        print('\nPre-training of LRURec + fine-tuning with FINEST on the original training data\n')
        [original_logits,reference_list,competitive_items, original_perf] = trainer.traintest(original_logits=-1, reference_list=-1,competitive_items=-1, args=args,data = original_data, dataset = original_dataset,mode="original_training",seed=seed)
        
        #perform random training data perturbation
        candidates = []
        for i in range(original_data.shape[0]):
            if original_data[i,3]==0:
                candidates.append(i)

        if args.attack_type =='casper':
            chosen = np.argsort(num_child)[-num_pert:]
        else:
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
        [temp1,temp2,temp3,perturbed_perf] = original_trainer.traintest(original_logits=original_logits, reference_list=reference_list,competitive_items=competitive_items, args=args,data = perturbed_data, dataset = dataset,mode="retraining",seed=seed)
        
        for i in range(3):
            avg_original_perf[i].append(original_perf[i])
        
        for i in range(5):
            avg_perturbed_perf[i].append(perturbed_perf[i])
        
    print('\n[Averaged metrics of FINEST]\n MRR = {}\tRecall@10 = {}\tNDCG@10 = {}\nRBO = {}\tJaccard@10 = {}'.format(np.average(avg_original_perf[0]),np.average(avg_original_perf[1]),np.average(avg_original_perf[2]),np.average(avg_perturbed_perf[3]),np.average(avg_perturbed_perf[4])),file=f,flush=True)

if __name__ == "__main__":
    main()
