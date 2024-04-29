import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def filter_and_split(data):

    (users,counts) = np.unique(data[:,0],return_counts = True)
    
    users = users[counts>=10]

    sequence_dic,pert_dic =  {int(user):[] for user in set(data[:,0])}, {int(user):[] for user in set(data[:,0])}
 
    user_dic = {int(user):idx for (idx,user) in enumerate(users)}
    new_data = []
    for i in range(data.shape[0]):
        if int(data[i,0]) in user_dic:
            new_data.append([int(data[i,0]),int(data[i,1]),data[i,2],0,0])

    new_data = np.array(new_data)
    items = np.unique(new_data[:,1])
    for i in range(new_data.shape[0]):
        new_data[i,-1] = np.random.choice(items)
        sequence_dic[int(new_data[i,0])].append([i,int(new_data[i,1]),new_data[i,2]])
   
    test_len = 0
    for user in sequence_dic.keys():
        cur_test = int(0.1*len(sequence_dic[user]))
        for i in range(cur_test):
            interaction = sequence_dic[user].pop()
            new_data[interaction[0],3] = 1
        test_len += cur_test

    new_data = new_data[np.argsort(new_data[:,2]),:]
    print(data.shape,new_data.shape)
    return new_data,test_len

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def computeRePos(time_seq, time_span):
    
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix

def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set: # float as map key?
        time_map[time] = int(round(float(time-time_min)))
    return time_map

def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
            item_set.add(item[3])

    user_set = sorted(user_set)
    item_set = sorted(item_set)
    
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1
    
    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]],x[2],item_map[x[3]],x[4]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1),x[2],x[3],x[4]], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max), user_map,item_map

def data_partition(data):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = defaultdict(list)
    user_test = defaultdict(list)
    
    time_set = set()
    for (idx,row) in enumerate(data):
        u, i, timestamp,test,neg_item,original_idx = row[0],row[1],row[2],row[3],row[4],row[5]
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        test = int(test)
        time_set.add(timestamp)
        User[u].append([i, timestamp,original_idx,neg_item,test])
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum,user_map,item_map = cleanAndsort(User, time_map)

    for user in User:
        for interaction in User[user]:
            if interaction[4]==0:
                user_train[user].append(interaction[:4])
            else:
                user_test[user].append(interaction[:3])
    return [user_train, user_test, usernum, itemnum, timenum,user_map,item_map]


