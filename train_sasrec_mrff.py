import pandas as pd
import numpy as np
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from model import SASRecEngine
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--num_round', type=int, default=500)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
parser.add_argument('--clients_sample_num', type=int, default=1)
parser.add_argument('--public_param_names', type=str, default='emb_item_names, pos_emb_names, attention_names, switch_names, last_layernorm_names, fc_layers_names, affine_output_names')
parser.add_argument('--private_param_names', type=str, default='emb_user_names, uexpert_names')
parser.add_argument('--num_experts', type=int, default=4)
parser.add_argument('--switch_hidden_dims', type=str, default='')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--dataset', type=str, default='KuaiSAR-R')
parser.add_argument('--split_method', type=str, default='leave_one_split')
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--latent_dim', default=8, type=int)
parser.add_argument('--position_dim', default=8, type=int)
parser.add_argument('--pred_layers', type=str, default="32")
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_regularization', type=float, default=0.)
parser.add_argument('--client_rep', type=str, default='con')# con, avg, con_ue, avg_ue
parser.add_argument('--device', default='cpu', type=str)
args = parser.parse_args()

# Transform string argument into list.
config = vars(args)
config['private_param_names'] = [str(item) for item in config['private_param_names'].split(',')]
config['public_param_names'] = [str(item) for item in config['public_param_names'].split(',')]
if len(config['pred_layers']) == 0:
    config['pred_layers'] = []
elif len(config['pred_layers']) == 1:
    config['pred_layers'] = [int(config['pred_layers'])]
else:
    config['pred_layers'] = [int(item) for item in config['pred_layers'].split(',')]
if len(config['switch_hidden_dims']) == 0:
    config['switch_hidden_dims'] = []
elif len(config['switch_hidden_dims']) == 1:
    config['switch_hidden_dims'] = [int(config['switch_hidden_dims'])]
else:
    config['switch_hidden_dims'] = [int(item) for item in config['switch_hidden_dims'].split(',')]
    
# Logging.
path = 'log/'+config['dataset']+'/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)

# Load the data
logging.info('Loading the data')
dataset_dir = './dataset/' + config['dataset'] + '/'
df, user_features, video_features = load_data(dataset_dir)
df.sort_values(by=["user_id", "timestamp"], ascending=[True, True], inplace=True, ignore_index=True)

# Model.
logging.info('Define the model')
selected_user_feature_file_dir = dataset_dir + config['dataset'] + '_user_feature_dict.txt'
selected_user_features = read_feature_names(selected_user_feature_file_dir)
selected_video_feature_file_dir = dataset_dir + config['dataset'] + '_video_feature_dict.txt'
selected_video_features = read_feature_names(selected_video_feature_file_dir)
new_video_features, df = process_video_feature_df(video_features, df, selected_video_features)
embedding_dim = config['latent_dim'] * len(selected_video_features)
latent_feature_dim = embedding_dim

if config['client_rep'] == 'con':
    insert_dim = latent_feature_dim * config['maxlen']
elif config['client_rep'] == 'avg':
    insert_dim = latent_feature_dim
elif config['client_rep'] == 'con_ue':
    insert_dim = latent_feature_dim * config['maxlen'] + config['latent_dim'] * len(selected_user_features)
elif config['client_rep'] == 'avg_ue':
    insert_dim = latent_feature_dim + config['latent_dim'] * len(selected_user_features)
else:
    pass
config['switch_hidden_dims'].insert(0, insert_dim)
config['switch_hidden_dims'].append(config['num_experts'])
engine = SASRecEngine(config, user_features, new_video_features, selected_user_features, selected_video_features)

# DataLoader for training
logging.info('Split the data')
if config['split_method'] == 'leave_one_split':
#     user_train_data, user_train_label, user_valid_data, user_valid_label, user_test_data, user_test_label = data_partition_leave_one_split(df)
#     np.save(dataset_dir+'/'+config['split_method']+'_'+'user_train_data.npy', user_train_data)
#     np.save(dataset_dir+'/'+config['split_method']+'_'+'user_train_label.npy', user_train_label)
#     np.save(dataset_dir+'/'+config['split_method']+'_'+'user_valid_data.npy', user_valid_data)
#     np.save(dataset_dir+'/'+config['split_method']+'_'+'user_valid_label.npy', user_valid_label)
#     np.save(dataset_dir+'/'+config['split_method']+'_'+'user_test_data.npy', user_test_data)
#     np.save(dataset_dir+'/'+config['split_method']+'_'+'user_test_label.npy', user_test_label)
    user_train_data = np.load(dataset_dir+'/'+config['split_method']+'_'+'user_train_data.npy', allow_pickle=True).item()
    user_train_label = np.load(dataset_dir+'/'+config['split_method']+'_'+'user_train_label.npy', allow_pickle=True).item()
    user_valid_data = np.load(dataset_dir+'/'+config['split_method']+'_'+'user_valid_data.npy', allow_pickle=True).item()
    user_valid_label = np.load(dataset_dir+'/'+config['split_method']+'_'+'user_valid_label.npy', allow_pickle=True).item()
    user_test_data = np.load(dataset_dir+'/'+config['split_method']+'_'+'user_test_data.npy', allow_pickle=True).item()
    user_test_label = np.load(dataset_dir+'/'+config['split_method']+'_'+'user_test_label.npy', allow_pickle=True).item()
else:
    pass

cc = 0.0
for u in user_train_data:
    cc += len(user_train_data[u])
print('average sequence length: %.2f' % (cc / len(user_train_data)))
logging.info('Process the train data')
train_data_dict = process_train_data(user_train_data, user_train_label, config['maxlen'])

logging.info('Process the evaluation data')
if config['split_method'] == 'leave_one_split':
    valid_data_dict = process_valid_data_leave_one_split(user_train_data, user_valid_data, user_valid_label, config['maxlen'])
    test_data_dict = process_test_data_leave_one_split(user_train_data, user_valid_data, user_test_data, user_test_label, config['maxlen'])
else:
    pass

eval_auc_list = []
val_auc_list = []
test_auc_list = []
eval_logloss_list = []
val_logloss_list = []
test_logloss_list = []
best_val_auc = 0
final_test_round = 0
best_train_auc = 0
temp = 0
for round_idx in range(config['num_round']):
    logging.info('-' * 80)
    logging.info('Round {} starts !'.format(round_idx))

    logging.info('-' * 80)
    logging.info('Training phase!')
    engine.fed_train_a_round(train_data_dict, list(train_data_dict.keys()), round_idx)
    
    if (round_idx+1) > -1:
        
        logging.info('-' * 80)
        logging.info('Validating phase!')
        val_auc, val_logloss = engine.fed_evaluate(list(valid_data_dict.keys()), valid_data_dict)
        # break
        logging.info('[Validating Round {}] AUC = {:.4f}'.format(round_idx, val_auc))
        val_auc_list.append(val_auc)
        logging.info('[Validating Round {}] LogLoss = {:.4f}'.format(round_idx, val_logloss))
        val_logloss_list.append(val_logloss)

        logging.info('-' * 80)
        logging.info('Testing phase!')
        test_auc, test_logloss = engine.fed_evaluate(list(test_data_dict.keys()), test_data_dict)
        logging.info('[Testing Round {}] AUC = {:.4f}'.format(round_idx, test_auc))
        test_auc_list.append(test_auc)
        logging.info('[Testing Round {}] LogLoss = {:.4f}'.format(round_idx, test_logloss))
        test_logloss_list.append(test_logloss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_round = temp
        
        temp += 1

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
file_str = current_time + '-' + 'latent_dim: ' + str(config['latent_dim']) + '-' + 'lr: ' + str(config['lr']) + '-' + \
           'clients_sample_ratio: ' + str(config['clients_sample_ratio']) + '-' + 'clients_sample_num: ' + str(config['clients_sample_num']) + '-' + 'local_epoch: ' + str(config['local_epoch']) + '-' + 'public_param_names: ' + str(config['public_param_names']) + '-' + 'private_param_names: ' + str(config['private_param_names']) + '-' + 'num_experts: ' + str(config['num_experts']) + '-' + 'switch_hidden_dims: ' + str(config['switch_hidden_dims']) + '-' + 'alpha: ' + str(config['alpha']) + '-' + \
           'num_blocks: ' + str(config['num_blocks']) + '-' + 'num_round: ' + str(config['num_round']) + '-' + 'dataset: ' + \
           config['dataset'] + '-' + 'num_heads: ' + str(config['num_heads']) + '-' + 'split_method: ' + str(config['split_method']) + '-' + 'pred_layers: ' + str(config['pred_layers']) + '-' + 'maxlen: ' + str(config['maxlen'])+ '-' + 'position_dim: ' + str(config['position_dim']) + '-' + 'optimizer: ' + config['optimizer'] + '-' + \
           'l2_regularization: ' + str(config['l2_regularization']) + '-' + 'dropout_rate: ' + str(config['dropout_rate']) + '-' + 'client_rep: ' + str(config['client_rep']) + '-' + 'auc: ' + \
           str(test_auc_list[final_test_round]) + '-' + 'logloss: ' + str(test_logloss_list[final_test_round]) + '-' + 'best_round: ' + str(final_test_round)
file_name = "sh_result/"+config['dataset']+".txt"
with open(file_name, 'a') as file:
    file.write(file_str + '\n')

logging.info('FedSASRec')
logging.info('latent_dim: {}, num_blocks: {}, lr: {}, dataset: {}, ' 
             'num_heads: {}'.
             format(config['latent_dim'], config['num_blocks'], config['lr'],
                    config['dataset'], config['num_heads']))

logging.info('test_auc_list: {}'.format(test_auc_list))
logging.info('test_logloss_list: {}'.format(test_logloss_list))
logging.info('Best test auc: {}, test logloss: {}: {} at round {}'.format(test_auc_list[final_test_round], test_logloss_list[final_test_round], final_test_round))