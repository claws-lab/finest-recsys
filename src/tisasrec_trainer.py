import numpy as np
import torch
import math
import sys
import time
from utils import *
from tisasrec_model import *
import rbo
from collections import Counter,defaultdict
from torch.nn import functional as F

class TiSASRec_Trainer(torch.nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, user_num, item_num, args, device, dataset,model):
        super(TiSASRec_Trainer, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.device = device
        self.dataset=dataset
        self.model = model

    def clip_gradients(self, limit=5):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), limit)
 
    def traintest(self,original_logits,reference_list,competitive_items,data,dataset,args,mode,seed):

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
 
        # create original training data (unperturbed), current training data (might be perturbed), and test data for LRURec
        [train,  test, usernum, itemnum, timenum,  user_map,item_map] = dataset
        [original_train,test,usernum,itemnum, timenum, user_map,item_map] = self.dataset
       
        train_num = sum([len(train[user]) for user in train.keys()])
        print("train instances #={}".format(train_num))
        test_num = sum([len(test[user]) for user in test.keys()])
        print("test instances #={}".format(test_num))
 
        original_user_list = sorted(list(original_train.keys()))
        user_list = sorted(list(train.keys()))
        maxlen = args.maxlen

        #user-wise mini-batch formulation
        num_batch = ((len(train)-1) // args.train_batch_size)+1
        print(usernum,itemnum)

        ce = torch.nn.CrossEntropyLoss(ignore_index=0)
        training_indices = [idx for idx in range(data.shape[0]) if data[idx,3]==0]
        item_indices = list(range(1, itemnum+1))

        # for perturbation simulation
        pert_size = int(len(training_indices)*args.simulation_ratio)
        occurence_count = Counter(data[:,1])
        popular_item = occurence_count.most_common(1)[0][0] 
        least_popular_item = occurence_count.most_common()[-1][0] 

        # logits contains recommendation probabilities for test instances
        logits = np.zeros((test_num,self.item_num))

        relation_matrix = Relation(train, usernum, args.maxlen, args.time_span)

        bce_criterion = torch.nn.BCEWithLogitsLoss()

        if mode=="original_training":
            reference_list = torch.zeros((data.shape[0],args.topk),dtype=torch.long).to(self.device)
            competitive_items = torch.zeros((data.shape[0],args.topk),dtype=torch.long).to(self.device)
        else:
            ranking_original = np.argsort(-original_logits,axis=1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr) 
        total_epochs = args.training_epoch+args.fine_tuning_epoch
        loss_margin = torch.nn.MarginRankingLoss(margin=0.1)
        ones = torch.ones(args.topk*args.maxlen).to(self.device)
        start_time = time.time() 

        for epoch in range(total_epochs):
            if epoch==args.training_epoch:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=args.fine_tune_lr) 
            # perturbation simulation
            if epoch>=args.training_epoch:
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
           
                new_dataset = data_partition(new_data)
                [train,  temp1,temp2,temp3,temp4,temp5,temp6] = new_dataset  
                user_list = sorted(list(train.keys()))
                num_batch = ((len(train)-1) // args.train_batch_size)+1

            total_loss = 0
            shuffled = np.arange(len(train))
            
            #different training order per epoch
            np.random.shuffle(shuffled)

            # user-wise mini-batch configuration
            for step in range(num_batch):
                st_idx,ed_idx = step*args.train_batch_size, (step+1)*args.train_batch_size
                if ed_idx>len(train):
                    ed_idx = len(train)

                cur_len = ed_idx-st_idx
                u = np.zeros((cur_len,1),dtype=np.int32)
                seq = np.zeros((cur_len,args.maxlen),dtype=np.int32)
                pos = np.zeros((cur_len,args.maxlen),dtype=np.int32)
                neg = np.zeros((cur_len,args.maxlen),dtype=np.int32)
                index = np.zeros((cur_len,args.maxlen),dtype=np.int32)
                time_matrix = np.zeros((cur_len,args.maxlen,args.maxlen),dtype=np.int32)
    
                # constructing training sequences for TiSASRec
                for i in range(st_idx,ed_idx):
                    cur_idx = st_idx-i
                    user = user_list[shuffled[i]]
                    seqt = np.zeros([maxlen], dtype=np.int32)
                    post = np.zeros([maxlen], dtype=np.int32)
                    negt = np.zeros([maxlen], dtype=np.int32)
                    indext = np.zeros([maxlen], dtype=np.int32)
                    nxt = train[user][-1][0]

                    idx = maxlen - 1
                    ts = {x[0]:1 for x in train[user]}
                    item_list = [item for item in range(1,itemnum+1) if item not in ts]
                    for j in reversed(train[user][:-1]):
                        seqt[idx] = j[0]
                        post[idx] = nxt
                        indext[idx] = j[2]
                        if nxt != 0: negt[idx] = j[3]
                        nxt = j[0]
                        idx -= 1
                        if idx == -1: break
                    
                    u[cur_idx] = user
                    seq[cur_idx] = seqt
                    pos[cur_idx] = post
                    neg[cur_idx] = negt
                    index[cur_idx] = indext
                    time_matrix[cur_idx] = relation_matrix[user]

                pos_logits, neg_logits = self.model(u, seq, time_matrix, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
                optimizer.zero_grad()
                indices = np.where(pos != 0)

                # compute the next-item prediction loss function
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in self.model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in self.model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in self.model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in self.model.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                for param in self.model.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                
                output = self.model.predict2(u, seq, time_matrix, item_indices)
                if epoch>=args.training_epoch: 
                    # rank-preserving regularization for all training instances
                    for i in range(ed_idx-st_idx):
                        # ignore padding in the sequence
                        valid_indices = [idx for idx in range(maxlen) if pos[i,idx]!=0 and index[i,idx]!=-1]
                        original_indices = index[i,valid_indices]
                        # top-k items from the reference rank list that we want to preserve 
                        topk_items = reference_list[original_indices]
                        # the logits of non-padding instances
                        valid_output = output[i,valid_indices,:]

                        #loss1 is preserving the relative ordering of the topk_items
                        first,second = torch.gather(valid_output,1,topk_items[:,:-1]).flatten(),torch.gather(valid_output,1,topk_items[:,1:]).flatten()
                        loss1 = loss_margin(first,second,ones[:first.shape[0]])

                        #loss2 is forcing the topk_items logits are higher than the competitive items (i.e., top-(K+1) to top-2K items) 
                        first,second = torch.gather(valid_output,1,topk_items).flatten(),torch.gather(valid_output,1,competitive_items[original_indices]).flatten()
                        loss2 = loss_margin(first,second,ones[:first.shape[0]])
                        
                        loss += 0.1*(loss1+loss2) 
    
                elif mode=="original_training" and epoch==args.training_epoch-1:
                    # obtain reference rank list for all training istances
                    for i in range(ed_idx-st_idx):
                        # ignore padding in the sequence
                        valid_indices = [idx for idx in range(maxlen) if pos[i,idx]!=0 and index[i,idx]!=-1]
                        original_indices = index[i,valid_indices]

                        # compute top-2K items for each training instance
                        ranking = torch.topk(output[i,valid_indices,:],k=args.topk*2,dim=1)[1]
                        reference_list[original_indices] = ranking[:,:args.topk]
                        competitive_items[original_indices] = ranking[:,args.topk:]
               
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
       
            if epoch%10==0:
                print("training loss in epoch {}: {}".format(epoch, total_loss))

            if epoch==total_epochs-1:
                self.model.eval()
                test_num_batch = ((len(original_train)-1)//args.test_batch_size)+1
                MRR,HITS,NDCG, avg_RBO,avg_jaccard,count = 0,0,0,0,0,0
                with torch.no_grad():
                    for step in range(test_num_batch):
                        st_idx,ed_idx = step*args.test_batch_size, (step+1)*args.test_batch_size
                        if ed_idx>len(original_train):
                            ed_idx = len(original_train)
                        cur_len = ed_idx-st_idx
         
                        test_users,test_seqs,test_matrix_seqs, test_labels = [],[],[],[]
                        for i in range(st_idx,ed_idx):
                            user = original_user_list[i]
                            if user not in test:
                                continue

                            # we split the train/test data by the timestamp (first 90% = train/remaining 10% = test) 
                            # to obtain test sequences, we need to build sequences with training data first
                            seq = original_train[user]
                            test_seq,time_seq = [],[]
                            start_index = 0 if len(seq)<=maxlen else len(seq)-maxlen
                            for j in range(start_index,len(seq)):
                                cur_item,timestamp = seq[j][0],seq[j][1]
                                test_seq.append(cur_item)
                                time_seq.append(timestamp)
                            
                            mask_len = maxlen - len(test_seq)
                            test_seq = [0] * mask_len + test_seq
                            time_seq = [0] * mask_len + time_seq

                            # append test data to the previous training sequences to form final ttest data
                            seq = test[user]
                            for j in range(len(seq)):
                                cur_item,timestamp = seq[j][0],seq[j][1]
                                test_users.append(user)
                                test_seqs.append(test_seq)
                                time_matrix = computeRePos(np.array(time_seq), args.time_span)
                                test_matrix_seqs.append(time_matrix)
                                test_labels.append(cur_item-1)
                                test_seq = test_seq[1:] + [cur_item]
                                time_seq = time_seq[1:] + [timestamp]

                        if len(test_seqs)==0:
                            continue
                       
                        u_array,matrix_array = np.array(test_users,dtype=int).reshape(-1,1), np.array(test_matrix_seqs,dtype=int)
                        seq_array = np.array(test_seqs,dtype=int)
                        cur_logits = self.model.predict(u_array, seq_array, matrix_array, item_indices).detach().cpu().numpy()
                        logits[count:count+cur_logits.shape[0]] = cur_logits
                        count+=cur_logits.shape[0]
                        for i in range(cur_logits.shape[0]):
                            target = int(test_labels[i])
                            rank = np.count_nonzero(cur_logits[i]>cur_logits[i,target])+1
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
