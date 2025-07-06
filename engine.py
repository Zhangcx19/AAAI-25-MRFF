import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import *
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import random
import copy
from collections import defaultdict


class Engine(object):

    def __init__(self, config):
        self.config = config  # model configuration
        self.server_model_param = {}
        self.client_model_params = {}
        self.group_params = init_group_params(self.model, config)
        self.param_name_dict = construct_uexpert_parm_name_dict(self.model, self.config['num_blocks'], self.config['num_experts'], len(self.config['switch_hidden_dims'])-1)
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.all_users_expert_dict = defaultdict(list)
    
    def fed_train_single_batch(self, model_client, user_train_data, optimizer, round_id, client_num):
        user, seq_data, pos_data, pos_label = user_train_data
        indices = np.where(np.array([pos_data]) != 0)
        user = torch.LongTensor(np.array([user]))
        seq_data = torch.LongTensor(np.array([seq_data]))
        pos_data = torch.LongTensor(np.array([pos_data]))
        pos_label = torch.FloatTensor(np.array([pos_label])).cuda()
        
        optimizer.zero_grad()     
        pos_logits, block_routes, block_out_pro_max = model_client(user, seq_data, pos_data)
        loss = self.crit(pos_logits[indices], pos_label[indices])
        
        expert_reg_loss = 0
        if round_id > 0:
            for idx in range(self.config['num_blocks']):
                expert_ind = block_routes[idx]
                block_expert_reg = self.config['alpha'] * self.config['num_experts'] * (self.all_users_expert_dict[round_id-1][idx][expert_ind]/client_num) * block_out_pro_max[idx]
                expert_reg_loss += block_expert_reg
            if self.config['client_rep'] == 'avg_ue':
                loss += expert_reg_loss[0]
            else:
                loss += expert_reg_loss
        else:
            pass
            
        loss.backward()
        optimizer.step()
        
        return model_client, block_routes
    
    def aggregate_clients_params(self, round_user_params, group_participant_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in round_user_params.keys():
            # load a user's parameters.
            user_params = round_user_params[user]
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)
            
        for idx in range(self.config['num_blocks']):
            for ind in range(self.config['num_experts']):
                idx_ind_expert_weight_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.weight'
                idx_ind_expert_bias_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.bias'
                idx_ind_expert_weight_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.weight'
                idx_ind_expert_bias_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.bias'
                if len(group_participant_params[idx_ind_expert_weight_0_name]) > 0:
                    self.group_params[idx_ind_expert_weight_0_name] = torch.tensor(np.mean(group_participant_params[idx_ind_expert_weight_0_name], axis=0))
                else:
                    pass
                if len(group_participant_params[idx_ind_expert_bias_0_name]) > 0:
                    self.group_params[idx_ind_expert_bias_0_name] = torch.tensor(np.mean(group_participant_params[idx_ind_expert_bias_0_name], axis=0))
                else:
                    pass
                if len(group_participant_params[idx_ind_expert_weight_3_name]) > 0:
                    self.group_params[idx_ind_expert_weight_3_name] = torch.tensor(np.mean(group_participant_params[idx_ind_expert_weight_3_name], axis=0))
                else:
                    pass
                if len(group_participant_params[idx_ind_expert_bias_3_name]) > 0:
                    self.group_params[idx_ind_expert_bias_3_name] = torch.tensor(np.mean(group_participant_params[idx_ind_expert_bias_3_name], axis=0))
                else:
                    pass
    
    def fed_train_a_round(self, train_data_dict, user_ids, round_id):
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(len(user_ids) * self.config['clients_sample_ratio'])
            participants = random.sample(user_ids, num_participants)
        else:
            participants = random.sample(user_ids, self.config['clients_sample_num'])
        
        # store users' model parameters of current round.
        round_participant_params = {}
        group_participant_params = {}
        for idx in range(self.config['num_blocks']):
            for ind in range(self.config['num_experts']):
                idx_ind_expert_weight_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.weight'
                idx_ind_expert_bias_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.bias'
                idx_ind_expert_weight_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.weight'
                idx_ind_expert_bias_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.bias'
                group_participant_params[idx_ind_expert_weight_0_name] = []
                group_participant_params[idx_ind_expert_bias_0_name] = []
                group_participant_params[idx_ind_expert_weight_3_name] = []
                group_participant_params[idx_ind_expert_bias_3_name] = []
        
        # perform model update for each participated user.
        for user in participants:
            model_client = copy.deepcopy(self.model)
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                for key in self.server_model_param.keys():
                    user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).cuda()
                for idx in range(self.config['num_blocks']):
                    for ind in range(self.config['num_experts']):
                        idx_ind_expert_weight_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.weight'
                        idx_ind_expert_bias_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.bias'
                        idx_ind_expert_weight_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.weight'
                        idx_ind_expert_bias_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.bias'
                        user_param_dict[idx_ind_expert_weight_0_name] = copy.deepcopy(self.group_params[idx_ind_expert_weight_0_name].data).cuda()
                        user_param_dict[idx_ind_expert_bias_0_name] = copy.deepcopy(self.group_params[idx_ind_expert_bias_0_name].data).cuda()
                        user_param_dict[idx_ind_expert_weight_3_name] = copy.deepcopy(self.group_params[idx_ind_expert_weight_3_name].data).cuda()
                        user_param_dict[idx_ind_expert_bias_3_name] = copy.deepcopy(self.group_params[idx_ind_expert_bias_3_name].data).cuda()
                model_client.load_state_dict(user_param_dict)

            # Defining optimizers
            optimizer = use_optimizer(model_client, self.config)
            user_train_data = train_data_dict[user]
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                model_client, block_routes = self.fed_train_single_batch(model_client, user_train_data, optimizer, round_id, len(participants))
            
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' local parameters for personalization.
            private_param_names = self.config['private_param_names']
            private_param_names_list = []
            for name in private_param_names:
                private_param_names_list.extend(self.param_name_dict[name])
            self.client_model_params[user] = {}
            for param_name in private_param_names_list:
                self.client_model_params[user][param_name] = copy.deepcopy(client_param[param_name].data.cpu())
                
            public_param_names = self.config['public_param_names']
            public_param_names_list = []
            for name in public_param_names:
                public_param_names_list.extend(self.param_name_dict[name])    
            round_participant_params[user] = {}
            for param_name in public_param_names_list:
                round_participant_params[user][param_name] = copy.deepcopy(client_param[param_name].data.cpu())

            for idx in range(self.config['num_blocks']):
                ind =  block_routes[idx]
                idx_ind_expert_weight_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.weight'
                idx_ind_expert_bias_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.bias'
                idx_ind_expert_weight_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.weight'
                idx_ind_expert_bias_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.bias'
                group_participant_params[idx_ind_expert_weight_0_name].append(copy.deepcopy(client_param[idx_ind_expert_weight_0_name].data.cpu().numpy()))
                group_participant_params[idx_ind_expert_bias_0_name].append(copy.deepcopy(client_param[idx_ind_expert_bias_0_name].data.cpu().numpy()))
                group_participant_params[idx_ind_expert_weight_3_name].append(copy.deepcopy(client_param[idx_ind_expert_weight_3_name].data.cpu().numpy()))
                group_participant_params[idx_ind_expert_bias_3_name].append(copy.deepcopy(client_param[idx_ind_expert_bias_3_name].data.cpu().numpy()))
            
        # save all users expert information.
        self.all_users_expert_dict[round_id] = count_all_users_expert(group_participant_params, self.config['num_blocks'], self.config['num_experts'])
        
        # aggregate client models in server side.
        self.aggregate_clients_params(round_participant_params, group_participant_params)

    def fed_evaluate(self, user_ids, eval_data_dict):
        temp = 0
        labels = []
        for user in user_ids:
            user_indices, log_seqs, item_indices, user_label = eval_data_dict[user]
            user_model = copy.deepcopy(self.model)
            user_param_dict = copy.deepcopy(self.model.state_dict())
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            for key in self.server_model_param.keys():
                user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).cuda()
            for idx in range(self.config['num_blocks']):
                for ind in range(self.config['num_experts']):
                    idx_ind_expert_weight_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.weight'
                    idx_ind_expert_bias_0_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.0.bias'
                    idx_ind_expert_weight_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.weight'
                    idx_ind_expert_bias_3_name = 'forward_layers.' + str(idx)+'.experts.'+str(ind)+'.3.bias'
                    user_param_dict[idx_ind_expert_weight_0_name] = copy.deepcopy(self.group_params[idx_ind_expert_weight_0_name].data).cuda()
                    user_param_dict[idx_ind_expert_bias_0_name] = copy.deepcopy(self.group_params[idx_ind_expert_bias_0_name].data).cuda()
                    user_param_dict[idx_ind_expert_weight_3_name] = copy.deepcopy(self.group_params[idx_ind_expert_weight_3_name].data).cuda()
                    user_param_dict[idx_ind_expert_bias_3_name] = copy.deepcopy(self.group_params[idx_ind_expert_bias_3_name].data).cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                user_pred = user_model.predict(user_indices, log_seqs, item_indices)
                if temp == 0:
                    preds = user_pred
                else:
                    preds = torch.cat((preds, user_pred))
                labels.extend(user_label)
                temp += 1
                    
        auc = roc_auc_score(np.array(labels), preds.cpu().numpy())
        logloss = log_loss(np.array(labels), preds.cpu().numpy().astype("float64"))

        return auc, logloss