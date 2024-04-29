#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2024-04-26
#FINEST: Stabilizing Recommendations by Rank-Preserving Fine-Tuning
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import torch
import time
import math
from collections import defaultdict,Counter
import numpy as np
import rbo
import copy

class BERTTrainer(torch.nn.Module):
    def __init__(self, args,device,original_data,bert_model):
        super().__init__()
        self.args = args
        self.device = device 
        self.num_item = len(np.unique(original_data[:,1]))
        self.random_seed = np.random.randint(0,2147483647)
        self.model = bert_model
        self.original_data = original_data

    def clip_gradients(self, limit=5):
        for p in self.parameters():
            torch.nn.utils.clip_grad_norm_(p, 5)
        
    def train_test_generator(self,data,user_map,item_map,flag):
         
        if flag==0:
            umap = {u: i+1 for i, u in enumerate(np.unique(data[:,0]))}
            smap = {s: i+1 for i, s in enumerate(np.unique(data[:,1]))}
        else:
            umap = user_map
            smap = item_map

        train, test = defaultdict(list),defaultdict(list)
        for i in range(data.shape[0]):
            user,item,oidx = umap[data[i,0]],smap[data[i,1]],data[i,-1]
            if data[i,3]==0:
                train[user].append([item,oidx])
            else:
                test[user].append([item,oidx])
           
        return train,test,umap,smap
 
    def traintest(self,original_logits,reference_list,competitive_items,args,data,mode):
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.random_seed)

        # loss function
        ce = torch.nn.CrossEntropyLoss(ignore_index=0)

        # create original training data (unperturbed), current training data (might be perturbed), and test data for BERT4Rec
        original_train,test,umap,smap = self.train_test_generator(self.original_data,-1,-1,0)
        train,temp,umap,smap = self.train_test_generator(data,umap,smap,1)
        self.num_user,self.num_item = len(umap),len(smap)
        
        train_num = sum([len(train[user]) for user in train.keys()])
        print("train instances #={}".format(train_num))
        test_num = sum([len(test[user]) for user in test.keys()])
        print("test instances #={}".format(test_num))

        max_len = args.bert_max_len
        mask_prob = args.bert_mask_prob
        mask_token = self.num_item + 1
    
        original_train_users,train_users,test_users = len(original_train),len(train),len(test)
        #user-wise mini-batch formulation
        num_batch = ((train_users-1) // self.args.train_batch_size) +1 
        print(self.num_user,self.num_item)

        prev_loss = 2147483647
        start_time = time.time()

        training_indices = [idx for idx in range(data.shape[0]) if data[idx,3]==0]
        # for perturbation simulation
        pert_size = int(len(training_indices)*args.simulation_ratio)
        occurence_count = Counter(self.original_data[:,1])
        popular_item = occurence_count.most_common(1)[0][0] 
        least_popular_item = occurence_count.most_common()[-1][0] 

        # logits contains recommendation probabilities for test instances
        logits = np.zeros((test_num,self.num_item))

        if mode=="original_training":
            reference_list = torch.zeros((data.shape[0],args.topk),dtype=torch.long).to(self.device)
            competitive_items = torch.zeros((data.shape[0],args.topk),dtype=torch.long).to(self.device)
        else:
            ranking_original = np.argsort(-original_logits,axis=1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr) 
        total_epochs = args.training_epoch+args.fine_tuning_epoch
        original_user_list = sorted(list(original_train.keys()))
        user_list = sorted(list(train.keys()))
        loss_margin = torch.nn.MarginRankingLoss(margin=0.1)
        ones = torch.ones(args.topk*args.train_batch_size*max_len).to(self.device)
        
        for epoch in range(total_epochs):
            if epoch==args.training_epoch:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=args.fine_tune_lr) 

            # perturbation simulation
            if epoch>=args.training_epoch:
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
                          
                train,temp,umap,smap = self.train_test_generator(new_data,umap,smap,1)
                self.num_user,self.num_item,train_users = len(umap),len(smap),len(train)
                num_batch = ((train_users-1) // args.train_batch_size)+1
                user_list = sorted(list(train.keys()))

            #different training order per epoch
            shuffled = np.arange(train_users)
            np.random.shuffle(shuffled)
            total_loss = 0

            # user-wise mini-batch configuration
            for step in range(num_batch):
                st_idx,ed_idx = step*self.args.train_batch_size, (step+1)*self.args.train_batch_size
                if ed_idx>train_users:
                    ed_idx = train_users
                cur_len = ed_idx-st_idx
 
                optimizer.zero_grad()

                seqs,labels,indices = [],[],[]
                for i in range(st_idx,ed_idx):
                    user = user_list[shuffled[i]]
                    seq = train[user]

                    tokens = []
                    label = []
                    index = []
                    start_index = 0 if len(seq)<=max_len else len(seq)-max_len
                    # constructing training sequences for BERT4Rec
                    for j in range(start_index,len(seq)):
                        s,idx = seq[j][0],seq[j][1]
                        
                        prob = np.random.random()
                        if prob < mask_prob:
                            prob /= mask_prob
                            if prob < 0.8:
                                tokens.append(mask_token)
                            elif prob < 0.9:
                                tokens.append(np.random.randint(1, self.num_item))
                            else:
                                tokens.append(s)
                            label.append(s)
                        else:
                            tokens.append(s)
                            label.append(0)
                        index.append(idx)

                    mask_len = max_len - len(tokens)

                    #padding
                    tokens = [0] * mask_len + tokens
                    label = [0] * mask_len + label
                    index = [-1] * mask_len + index

                    seqs.append(tokens)
                    labels.append(label)
                    indices.append(index)

                seqs,labels,indices = torch.LongTensor(seqs).to(self.device),torch.LongTensor(labels).to(self.device),np.array(indices)
                pred = self.model(seqs)
                output = pred.clone()
                pred = pred.view(-1, pred.size(-1))
                labels = labels.view(-1)
                # compute the next-item prediction loss function
                loss = ce(pred, labels)
                
                if epoch>=args.training_epoch: 
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

                        loss += 1.0*(loss1+loss2) 
                
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

                total_loss += loss.item()
                loss.backward()
                self.clip_gradients(5)
                optimizer.step()
            
            if (epoch+1)%10==0:
                print('Epoch = {}\tTraining loss = {}'.format(epoch,total_loss))

            if (epoch==total_epochs-1):
                self.model.eval()
                test_num_batch = ((original_train_users-1)//args.test_batch_size)+1
                MRR,HITS,NDCG, avg_RBO,avg_jaccard,count = 0,0,0,0,0,0
                with torch.no_grad():
                    for step in range(test_num_batch):
                        st_idx,ed_idx = step*self.args.test_batch_size, (step+1)*self.args.test_batch_size
                        if ed_idx>original_train_users:
                            ed_idx = original_train_users
                        cur_len = ed_idx-st_idx
         
                        users,seqs,labels,indices = [],[],[],[]
                        for i in range(st_idx,ed_idx):
                            user = original_user_list[i]
                            if user not in test:
                                continue

                            # we split the train/test data by the timestamp (first 90% = train/remaining 10% = test) 
                            # to obtain test sequences, we need to build sequences with training data first
                            seq = original_train[user]
                            tokens = []
                            start_index = 0 if len(seq)<=max_len else len(seq)-max_len
                            for j in range(start_index,len(seq)):
                                s,idx = seq[j][0],seq[j][1]
                                tokens.append(s)
                            
                            mask_len = max_len - len(tokens)
                            tokens = [0] * mask_len + tokens

                            # append test data to the previous training sequences to form final ttest data
                            seq = test[user]
                            for j in range(len(seq)):
                                s,idx = seq[j][0],seq[j][1]
                                tokens = tokens[1:] + [mask_token]
                                users.append(user)
                                seqs.append(copy.deepcopy(tokens))
                                labels.append(s)
                                tokens[-1] = s
                       
                        if len(seqs)==0:
                            continue

                        seqs,labels = torch.LongTensor(seqs).to(self.device),np.array(labels,dtype=int).reshape(-1,1)
                        scores = self.model(seqs)
                        # we only need the next-item prediction result
                        scores = scores[:, -1, :]
                        softmax = torch.nn.Softmax(dim=1)
                        cur_logits = softmax(scores).cpu().numpy()

                        logits[count:count+cur_logits.shape[0]] = cur_logits[:,1:-1]
                        count+=cur_logits.shape[0]
                        for i in range(cur_logits.shape[0]):
                            rank = np.count_nonzero(cur_logits[i]>cur_logits[i,labels[i]])+1
                            MRR += 1/rank
                            HITS += (1 if rank<=10 else 0)
                            NDCG += (1/math.log2(rank+1) if rank<=10 else 0)

                MRR/=test_num
                HITS/=test_num
                NDCG/=test_num

                print("Epoch {}\tTest MRR&Recall@10&NDCG@10 = {},{},{}\tElapsed time: {}".format(epoch+1, MRR, HITS,NDCG,time.time() - start_time))

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
                    print('[Stability Metrics] Avg. RBO = {}\tAvg. Jaccard@10 = {}'.format(avg_RBO,avg_jaccard)) 

    
        return [logits,reference_list,competitive_items,[MRR,HITS,NDCG,avg_RBO,avg_jaccard]]

