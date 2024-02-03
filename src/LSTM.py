#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2024-02-01
#FINEST: Stabilizing Recommendations by Rank-Preserving Fine-Tuning
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict,Counter
import os
import time
import rbo
import copy

def train_test_data_per_user(data):

    train,test = defaultdict(list),defaultdict(list)
    for (idx,row) in enumerate(data):
        user,item,time,index = int(row[0]),int(row[1]),row[2],row[4]
        if row[3]==0:
            train[user].append([item,index])   
        else:
            test[user].append([item,index])
                                                                
    return train,test

class LSTM(nn.Module):
    def __init__(self, data, input_size, output_size, hidden_dim, look_back, n_layers=1, device="cpu",seed=0):
        super(LSTM, self).__init__()
        
        self.data = data
        self.num_items = output_size
        self.device = device 
        self.emb_length = input_size
        self.item_emb = nn.Embedding(self.num_items, self.emb_length,padding_idx=0)

        # training mini-batch is constructed user-wise, not instance-wise.
        self.batch_size = 16
        
        self.random_seed = seed
        self.look_back = look_back 

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # LSTM Layer
        self.LSTM = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.LSTM(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach(), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach())
        return hidden

    def traintest(self, original_logits,reference_list,competitive_items,args,data,mode):
 
        (original_train,test) = train_test_data_per_user(self.data)
         
        (train,dummy) = train_test_data_per_user(data)
        num_user = len(train)
        train_num = sum([len(train[user]) for user in train.keys()])
        print("train #={}".format(train_num))
        test_num = sum([len(test[user]) for user in test.keys()])
        print("test #={}".format(test_num))

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        learning_rate = 1e-3
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        prev_loss = 2147483647
        start_time = time.time()

        training_indices = [idx for idx in range(data.shape[0]) if data[idx,3]==0]
        # for perturbation simulation
        pert_size = int(len(training_indices)*args.simulation_ratio)
        occurence_count = Counter(self.data[:,1])
        popular_item = occurence_count.most_common(1)[0][0] 
        least_popular_item = occurence_count.most_common()[-1][0] 

        # logits contains recommendation probabilities for test instances
        logits = np.zeros((test_num,self.num_items))

        MRR,HITS,avg_RBO,avg_jaccard = 0,0,0,0
        if mode=="original_training" or mode=="retraining":
            print("normal training of a recommendation model......")
            if mode=="original_training":
                reference_list = torch.zeros((data.shape[0],args.topk),dtype=torch.long).to(self.device)
                competitive_items = torch.zeros((data.shape[0],args.topk),dtype=torch.long).to(self.device)
            else:
                ranking_original = np.argsort(-original_logits,axis=1)
            total_epochs = args.training_epoch
        else:
            print("fine-tuning of a recommendation model......") 
            ranking_original = np.argsort(-original_logits,axis=1)
            total_epochs = args.fine_tuning_epoch
 
        loss_margin = nn.MarginRankingLoss(margin=0.1)
        ones = torch.ones(args.topk*self.batch_size*self.look_back).to(self.device)
        user_list = list(train.keys())

        for epoch in range(total_epochs):
            if mode=="fine_tuning":
                # perturbation simulation
                chosen_indices = np.random.choice(training_indices,size=pert_size,replace=False)
                sample_mode = np.random.randint(0,3)
                #deletion perturbation
                if sample_mode==0:
                    new_data = np.delete(data,chosen_indices,axis=0)
                #item replacement perturbation with the least popular item in the dataset
                elif sample_mode==1:
                    new_data = copy.deepcopy(data)
                    new_data[chosen_indices,1] = int(least_popular_item)
                    # -1 indicates this instance is a simulated one
                    new_data[chosen_indices,-1] = -1
                else:
                    values = copy.deepcopy(data[chosen_indices,:])
                    values[:,1] = int(least_popular_item)
                    # insert instance right before the existing instance
                    values[:,2] -= 1
                    # -1 indicates this instance is a simulated one 
                    values[:,-1] = -1
                    new_data = np.insert(data,chosen_indices,values,axis=0)
             
                (train,dummy) = train_test_data_per_user(new_data)


            train_loss=0
            # user-wise mini-batch configuration
            for iteration in range((num_user-1)//self.batch_size+1):
                st_idx,ed_idx = iteration*self.batch_size, (iteration+1)*self.batch_size
                if ed_idx>num_user:
                    ed_idx = num_user
                        
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                
                seqs,labels,indices = [],[],[]
                for i in range(st_idx,ed_idx):
                    user = user_list[i]
                    seq_all = train[user]

                    seq,label,index = [],[],[]
                    for j in range(len(seq_all)-1):
                        seq.append(seq_all[j][0])
                        label.append(seq_all[j+1][0])
                        index.append(seq_all[j+1][1])
                    
                    #take only recent #look_back interactions per user
                    seq = seq[:self.look_back]
                    label = label[:self.look_back]
                    index = index[:self.look_back]

                    mask_len = self.look_back-len(seq)
                    #padding
                    seq = [0]*mask_len+seq
                    label = [0]*mask_len+label
                    index = [-1]*mask_len+index

                    seqs.append(seq)
                    labels.append(label)
                    indices.append(index)

                seqs,labels,indices = torch.LongTensor(seqs).to(self.device),torch.LongTensor(labels).to(self.device),np.array(indices)
                pred = self(self.item_emb(seqs))
                output = pred.clone()
                pred = pred.view(-1, pred.size(-1))
                labels = labels.view(-1)
                loss = criterion(pred,labels)

                if mode=="fine_tuning": 
                    # rank-preserving regularization for all training instances
                    for i in range(ed_idx-st_idx):
                        # ignore padding in the sequence
                        cur_indices = np.nonzero(indices[i]!=-1)[0]
                        original_indices = indices[i][cur_indices]

                        # top-k items from the reference rank list that we want to preserve 
                        topk_items = reference_list[original_indices]
                        # the logits of non-padding instances
                        valid_output = output[i][cur_indices]

                        #loss1 is preserving the relative ordering of the topk_items
                        first,second = torch.gather(valid_output,1,topk_items[:,:-1]).flatten(),torch.gather(valid_output,1,topk_items[:,1:]).flatten()
                        loss1 = loss_margin(first,second,ones[:first.shape[0]])

                        #loss2 is forcing the topk_items logits are higher than the competitive items (i.e., top-(K+1) to top-2K items) 
                        first,second = torch.gather(valid_output,1,topk_items).flatten(),torch.gather(valid_output,1,competitive_items[original_indices]).flatten()
                        loss2 = loss_margin(first,second,ones[:first.shape[0]])

                        loss += (loss1+loss2) 
                elif mode=="original_training" and epoch==args.training_epoch-1:
                    # obtain reference rank list for all training istances
                    for i in range(ed_idx-st_idx):
                        # ignore padding in the sequence 
                        cur_indices = np.nonzero(indices[i]!=-1)[0]
                        original_indices = indices[i][cur_indices]

                        # compute top-2K items for each training instance
                        ranking = torch.topk(output[i][cur_indices],k=args.topk*2,dim=1)[1]
                        reference_list[original_indices] = ranking[:,:args.topk]
                        competitive_items[original_indices] = ranking[:,args.topk:]

                loss.backward()  # Does backpropagation and calculates gradients
                train_loss += loss.item()
                optimizer.step()  # Updates the weights accordingly
 
            if (epoch==total_epochs-1):
                # evaluation phase
                MRR,HITS,count = 0,0,0
                with torch.no_grad():
                    # user-wise mini-batch configuration 
                    for iteration in range((num_user-1)//128 + 1):
                        st_idx, ed_idx = iteration * 128, (iteration + 1) * 128
                        if ed_idx > num_user:
                            ed_idx = num_user

                        seqs,labels,indices = [],[],[]
                        for i in range(st_idx,ed_idx):
                            user = user_list[i]
                            seq_all = original_train[user]
                            seq = []
                            for j in range(len(seq_all)):
                                seq.append(seq_all[j][0])
                            
                            # take only recent #look_back interactions per user
                            seq = seq[:self.look_back]
                            mask_len = self.look_back-len(seq)
                            # padding
                            seq = [0]*mask_len+seq
                            
                            seq_all = test[user]
                            for j in seq_all:
                                seqs.append(seq)
                                labels.append(j[0])
                                seq = seq[1:]+[j[0]]

                        seqs,labels = torch.LongTensor(seqs).to(self.device),torch.LongTensor(labels).to(self.device)
                        cur_logits = self(self.item_emb(seqs))[:,-1,:].detach().cpu().numpy()
                        
                        logits[count:count+cur_logits.shape[0]] = cur_logits
                        count+=cur_logits.shape[0]
                        for i in range(cur_logits.shape[0]):
                            rank = np.count_nonzero(cur_logits[i]>cur_logits[i,labels[i]])+1
                            #compute next-item prediction metrics
                            MRR += 1/rank
                            HITS += (1 if rank<=10 else 0)

                    MRR /= test_num
                    HITS /= test_num        
                    print("Epoch {}\tTrain Loss: {}\tTest MRR&Recall@10 = {},{}\tElapsed time: {}".format(epoch+1, train_loss/train_num, MRR, HITS,time.time() - start_time))

                    if mode!="original_training":
                        all_RBO,all_jaccard = [],[]
                        ranking_now = np.argsort(-logits,axis=1)
                        for i in range(test_num):
                            #compute rank list senstiviity metrics
                            RBO = rbo.RankingSimilarity(ranking_original[i,:], ranking_now[i,:]).rbo()
                            jaccard = np.intersect1d(ranking_original[i,:10],ranking_now[i,:10]).shape[0]/np.union1d(ranking_original[i,:10],ranking_now[i,:10]).shape[0]
                            all_RBO.append(RBO)
                            all_jaccard.append(jaccard)
               
                        avg_RBO,avg_jaccard = np.average(all_RBO),np.average(all_jaccard)
                        print('Test RBO = {}\tTest Jaccard = {}'.format(avg_RBO,avg_jaccard)) 
 
                start_time = time.time()
        
        return [logits,reference_list,competitive_items,[MRR,HITS,avg_RBO,avg_jaccard]]

