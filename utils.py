import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

import pandas as pd
import random
import copy
from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import scipy.stats



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer
        
        
def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    
"""load data from the processed dataset"""
def load_data(data_dir):
    df = pd.read_csv(data_dir + "df.csv")
    user_features = pd.read_csv(data_dir + "user_features.csv")
    video_features = pd.read_csv(data_dir + "video_features.csv")
    
    return df, user_features, video_features

    
def read_bin_dict(file_dir):
    with open(file_dir, 'r') as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content] 
    content_dict = {}
    for line in content:
        key, value = line.split(':')
        content_dict[key] = [float(item) for item in value.split(',')]
        content_dict[key][-1] = int(content_dict[key][-1])
        
    return content_dict


"""modify the float column value of dataframe by reindexing the value with bin groups"""
def preprocess_float_df(df, column_name, column_bins):
    value_bins = linspace(column_bins[0], column_bins[1], column_bins[2]).tolist()
    df[column_name] = df[column_name].apply(np.log1p)
    value_bins.insert(0, df[column_name].min()-1)
    value_bins.append(df[column_name].max()+1)
    labels = range(column_bins[2]+1)
    df[column_name] = pd.cut(x = df[column_name], 
                      bins = value_bins, 
                      labels = labels, 
                      include_lowest = True)
    
    return df

""""wrap the dataframe preprocess function for float feature values"""
def wrap_preprocess_float_df(df, column_name_value_bins_dict):
    new_df = copy.deepcopy(df)
    for column_name in column_name_value_bins_dict.keys():
        column_bins = column_name_value_bins_dict[column_name]
        new_df = preprocess_float_df(new_df, column_name, column_bins)
    
    return new_df


"""read the feature names from txt file"""
def read_feature_names(file_dir):
    feature_names = []
    with open(file_dir, 'r') as f:
        for line in f:
            feature_names.append(line.strip('\n'))
            
    return feature_names
    

"""replace the video id=0 with the new id and add the video attribute to the id=0"""
def process_video_feature_df(video_feature, user_df, selected_video_features):
    zero_replace_value = video_feature['video_id'].max()+1
    video_feature['video_id'] = video_feature['video_id'].replace(0, zero_replace_value)
    user_df['video_id'] = user_df['video_id'].replace(0, zero_replace_value)
    
    added_line_values = [0 for i in range(video_feature.shape[1])]
    video_feature.loc[len(video_feature.index)] = added_line_values
    for feature_name in selected_video_features:
        feature_unique_value_count = video_feature[feature_name].max() + 1
        video_feature.loc[video_feature['video_id']==0, feature_name]=feature_unique_value_count
    video_feature.sort_values(by=["video_id"], ascending=[True], inplace=True, ignore_index=True)
    return video_feature, user_df


# train/val/test data generation
def data_partition_leave_one_split(user_df):
    """create three user-item interaction dicts, the key is user ID, the value is item ID list"""
    User_data = defaultdict(list)
    User_label = defaultdict(list)
    user_train_data = {}
    user_valid_data = {}
    user_test_data = {}
    user_train_label = {}
    user_valid_label = {}
    user_test_label = {}
    
    for user in user_df['user_id'].unique():
        User_data[user] = user_df.loc[user_df['user_id'] == user, 'video_id'].values.tolist()
        User_label[user] = user_df.loc[user_df['user_id'] == user, 'is_click'].values.tolist()
    
    for user in User_data:
        user_train_data[user] = User_data[user][:-2]
        user_train_label[user] = User_label[user][:-2]
        user_valid_data[user] = []
        user_valid_data[user].append(User_data[user][-2])
        user_valid_label[user] = []
        user_valid_label[user].append(User_label[user][-2])
        user_test_data[user] = []
        user_test_data[user].append(User_data[user][-1])
        user_test_label[user] = []
        user_test_label[user].append(User_label[user][-1])
    
    return user_train_data, user_train_label, user_valid_data, user_valid_label, user_test_data, user_test_label


def process_train_data(user_train_data, user_train_label, maxlen):
    user_ids = list(user_train_data.keys())
    train_data_dict = defaultdict(list)
    for user in user_ids:
        seq_data = np.zeros([maxlen], dtype=np.int32)
        pos_data = np.zeros([maxlen], dtype=np.int32)
        pos_label = np.zeros([maxlen], dtype=np.float32)
        nxt_data = user_train_data[user][-1]
        nxt_label = user_train_label[user][-1]
        
        idx = maxlen - 1
        for i in reversed(user_train_data[user][:-1]):
            """seq is the input sequence"""
            seq_data[idx] = i
            """pos is the next interaction of current interaction"""
            pos_data[idx] = nxt_data
            nxt_data = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i in reversed(user_train_label[user][:-1]):
            pos_label[idx] = nxt_label
            nxt_label = i
            idx -= 1
            if idx == -1: break
        
        train_data_dict[user] = [user, seq_data, pos_data, pos_label]
    
    return train_data_dict


def process_valid_data_leave_one_split(user_train_data, user_valid_data, user_valid_label, maxlen):
    user_ids = list(user_train_data.keys())
    valid_data_dict = defaultdict(list)
    for u in user_ids:
        user_indices, log_seqs, item_indices, labels = [], [], [], []
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train_data[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        item_idx = [user_valid_data[u][0]]
        user_indices.append(u)
        log_seqs.append(seq)
        item_indices.append(item_idx)
        labels.append(user_valid_label[u][0])
        valid_data_dict[u] = [np.array(user_indices), np.array(log_seqs), np.array(item_indices), labels]
    
    return valid_data_dict


def process_test_data_leave_one_split(user_train_data, user_valid_data, user_test_data, user_test_label, maxlen):
    user_ids = list(user_train_data.keys())
    test_data_dict = defaultdict(list)
    for u in user_ids:
        user_indices, log_seqs, item_indices, labels = [], [], [], []
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        """the validation sample is the input sequence of test phase"""
        seq[idx] = user_valid_data[u][0]
        idx -= 1
        for i in reversed(user_train_data[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        item_idx = [user_test_data[u][0]]
        user_indices.append(u)
        log_seqs.append(seq)
        item_indices.append(item_idx)
        labels.append(user_test_label[u][0])
        test_data_dict[u] = [np.array(user_indices), np.array(log_seqs), np.array(item_indices), labels]
        
    return test_data_dict


def construct_uexpert_parm_name_dict(model, num_blocks, num_experts, switch_hidden_layers):
    parm_name_dict = defaultdict(list)
    for param_name in model.selected_user_features:
        parm_name_dict['emb_user_names'].append('embedding_user_features.'+param_name+'.weight')
    for param_name in model.selected_video_features:
        parm_name_dict['emb_item_names'].append('embedding_video_features.'+param_name+'.weight')
    parm_name_dict['pos_emb_names'] = ['pos_emb.weight']
    for idx in range(num_blocks):
        parm_name_dict['attention_names'].append('attention_layernorms.'+str(idx)+'.weight')
        parm_name_dict['attention_names'].append('attention_layernorms.'+str(idx)+'.bias')
        parm_name_dict['attention_names'].append('attention_layers.'+str(idx)+'.in_proj_weight')
        parm_name_dict['attention_names'].append('attention_layers.'+str(idx)+'.in_proj_bias')
        parm_name_dict['attention_names'].append('attention_layers.'+str(idx)+'.out_proj.weight')
        parm_name_dict['attention_names'].append('attention_layers.'+str(idx)+'.out_proj.bias')
        parm_name_dict['attention_names'].append('forward_layernorms.'+str(idx)+'.weight')
        parm_name_dict['attention_names'].append('forward_layernorms.'+str(idx)+'.bias')
        parm_name_dict['uexpert_names'].append('forward_layers.'+str(idx)+'.uexpert'+'.0.weight')
        parm_name_dict['uexpert_names'].append('forward_layers.'+str(idx)+'.uexpert'+'.0.bias')
        parm_name_dict['uexpert_names'].append('forward_layers.'+str(idx)+'.uexpert'+'.3.weight')
        parm_name_dict['uexpert_names'].append('forward_layers.'+str(idx)+'.uexpert'+'.3.bias')
        for ind in range(num_experts):
            parm_name_dict['expert_names'].append('forward_layers.'+str(idx)+'.experts.'+str(ind)+'.0.weight')
            parm_name_dict['expert_names'].append('forward_layers.'+str(idx)+'.experts.'+str(ind)+'.0.bias')
            parm_name_dict['expert_names'].append('forward_layers.'+str(idx)+'.experts.'+str(ind)+'.3.weight')
            parm_name_dict['expert_names'].append('forward_layers.'+str(idx)+'.experts.'+str(ind)+'.3.bias')
        for ind in range(switch_hidden_layers):
            parm_name_dict['switch_names'].append('forward_layers.'+str(idx)+'.switch.'+str(ind)+'.weight')
            parm_name_dict['switch_names'].append('forward_layers.'+str(idx)+'.switch.'+str(ind)+'.bias')
    parm_name_dict['last_layernorm_names'] = ['last_layernorm.weight', 'last_layernorm.bias']
    for idx in range(len(model.fc_layers)):
        parm_name_dict['fc_layers_names'].append('fc_layers.'+str(idx)+'.weight')
        parm_name_dict['fc_layers_names'].append('fc_layers.'+str(idx)+'.bias')
    parm_name_dict['affine_output_names'] = ['affine_output.weight', 'affine_output.bias']
    
    return parm_name_dict


def init_group_params(model, config):
    model_param = model.state_dict()
    group_params = {}
    for idx in range(config['num_blocks']):
        for ind in range(config['num_experts']):
            idx_ind_expert_weight_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.weight'
            idx_ind_expert_bias_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.bias'
            idx_ind_expert_weight_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.weight'
            idx_ind_expert_bias_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.bias'
            group_params[idx_ind_expert_weight_0_name] = copy.deepcopy(model_param[idx_ind_expert_weight_0_name].data.cpu())
            group_params[idx_ind_expert_bias_0_name] = copy.deepcopy(model_param[idx_ind_expert_bias_0_name].data.cpu())
            group_params[idx_ind_expert_weight_3_name] = copy.deepcopy(model_param[idx_ind_expert_weight_3_name].data.cpu())
            group_params[idx_ind_expert_bias_3_name] = copy.deepcopy(model_param[idx_ind_expert_bias_3_name].data.cpu())
    
    return group_params


def count_all_users_expert(group_participant_params, num_blocks, num_experts):
    all_users_expert_list = []
    for idx in range(num_blocks):
        block_user_num_list = []
        for ind in range(num_experts):
            idx_ind_expert_weight_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.weight'
            block_user_num_list.append(len(group_participant_params[idx_ind_expert_weight_0_name]))
        all_users_expert_list.append(block_user_num_list)
        
    return all_users_expert_list