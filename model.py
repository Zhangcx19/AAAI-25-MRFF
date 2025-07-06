import numpy as np
import torch
from engine import Engine
from utils import get_parameter_number
import copy


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class MOEPointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, maxlen, switch_hidden_dims, dropout_rate, num_experts, config):

        super(MOEPointWiseFeedForward, self).__init__()

        self.config = config
        self.n_experts = num_experts
        self.switch_hidden_dims = switch_hidden_dims
        self.uexpert = torch.nn.Sequential(torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
                                                torch.nn.Dropout(p=dropout_rate),
                                                torch.nn.ReLU(),
                                                torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
                                                torch.nn.Dropout(p=dropout_rate))
        single_expert_net = torch.nn.Sequential(torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
                                                torch.nn.Dropout(p=dropout_rate),
                                                torch.nn.ReLU(),
                                                torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
                                                torch.nn.Dropout(p=dropout_rate))
        self.experts = torch.nn.ModuleList([single_expert_net for _ in range(num_experts)])
        self.switch = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.switch_hidden_dims[:-1], self.switch_hidden_dims[1:])):
            self.switch.append(torch.nn.Linear(in_size, out_size))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, x_real, user_embedding):
        if self.config['client_rep'] == 'con':
            x_input = x_real.view(1, -1)
        elif self.config['client_rep'] == 'avg':
            x_input = torch.mean(x_real.squeeze(), dim=0)
        elif self.config['client_rep'] == 'con_ue':
            x_input = torch.cat((x_real.view(1, -1), user_embedding), 1)
        elif self.config['client_rep'] == 'avg_ue':
            x_input = torch.cat((torch.mean(x_real.squeeze(), dim=0).unsqueeze(0), user_embedding), 1)
        else:
            pass

        for idx, _ in enumerate(range(len(self.switch))):
            x_input = self.switch[idx](x_input)
            if idx < len(self.switch) - 1:
                x_input = torch.nn.ReLU()(x_input)

        route_prob = self.softmax(x_input)
        out_prob_max, routes = torch.max(route_prob, dim=-1)
        adopted_expert = self.experts[routes.item()]
        output = adopted_expert(x.transpose(-1, -2))
        output = output.transpose(-1, -2)
        
        uexpert_output = self.uexpert(x.transpose(-1, -2))
        uexpert_output = uexpert_output.transpose(-1, -2)
        final_output = output + uexpert_output

        return final_output, routes.item(), out_prob_max

class SASRec(torch.nn.Module):
    def __init__(self, config, user_features, video_features, selected_user_features, selected_video_features):
        super(SASRec, self).__init__()

        self.config = config
        self.latent_dim = self.config['latent_dim']
        self.user_features = user_features
        self.video_features = video_features
        self.selected_user_features = selected_user_features
        self.selected_video_features = selected_video_features
        self.pred_layers = self.config['pred_layers']
        self.embedding_dim = self.latent_dim * len(self.selected_video_features)
        self.pred_dim = self.latent_dim * len(self.selected_video_features)

        """user embedding"""
        self.embedding_user_features = torch.nn.ModuleDict()
        for feature_name in self.selected_user_features:
            feature_unique_value_count = user_features[feature_name].max() + 1
            self.embedding_user_features[feature_name] = torch.nn.Embedding(num_embeddings=feature_unique_value_count,
                                                                            embedding_dim=self.latent_dim)

        """item embedding"""
        self.embedding_video_features = torch.nn.ModuleDict()
        for feature_name in self.selected_video_features:
            feature_unique_value_count = video_features[feature_name].max() + 1
            self.embedding_video_features[feature_name] = torch.nn.Embedding(
                    num_embeddings=feature_unique_value_count, embedding_dim=self.latent_dim,
                    padding_idx=video_features[feature_name].max())

        """position embedding"""
        self.pos_emb = torch.nn.Embedding(self.config['maxlen'], self.embedding_dim)
        self.latent_feature_dim = self.embedding_dim
        self.emb_dropout = torch.nn.Dropout(p=self.config['dropout_rate'])

        """attention block"""
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.latent_feature_dim, eps=1e-8)
        for _ in range(self.config['num_blocks']):
            new_attn_layernorm = torch.nn.LayerNorm(self.latent_feature_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.latent_feature_dim, self.config['num_heads'],
                                                         self.config['dropout_rate'])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.latent_feature_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = MOEPointWiseFeedForward(self.latent_feature_dim, self.config['maxlen'],
                                                    self.config['switch_hidden_dims'], self.config['dropout_rate'],
                                                    self.config['num_experts'], self.config)
            self.forward_layers.append(new_fwd_layer)

        """prediction function"""
        self.pred_layers.insert(0, self.latent_feature_dim + self.latent_dim * (
                len(self.selected_user_features) + len(self.selected_video_features)))

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.pred_layers[:-1], self.pred_layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.affine_output = torch.nn.Linear(in_features=self.pred_layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def embedding_consrtuction(self, embedding_flag, indices):
        if embedding_flag == 'user':
            for feature_idx in range(len(self.selected_user_features)):
                feature_name = self.selected_user_features[feature_idx]
                user_feature_values = self.user_features.loc[indices][feature_name].values
                user_feature_embedding = self.embedding_user_features[feature_name](
                    torch.LongTensor(user_feature_values).cuda())
                if feature_idx == 0:
                    user_embedding = user_feature_embedding
                else:
                    user_embedding = torch.cat([user_embedding, user_feature_embedding], dim=-1)
            return user_embedding
        elif embedding_flag == 'video':
            for feature_idx in range(len(self.selected_video_features)):
                feature_name = self.selected_video_features[feature_idx]
                video_feature_values = self.video_features.loc[indices][feature_name].values
                video_feature_embedding = self.embedding_video_features[feature_name](
                    torch.LongTensor(video_feature_values).cuda())
                if feature_idx == 0:
                    video_embedding = video_feature_embedding
                else:
                    video_embedding = torch.cat([video_embedding, video_feature_embedding], dim=-1)
            return video_embedding
        else:
            pass

    def input_batch_seq_emb_construction(self, log_seqs, user_embedding):
        seqs = self.embedding_consrtuction('video', log_seqs[0])
        for idx in range(1, len(log_seqs)):
            user_seq = log_seqs[idx]
            user_seq_emb = self.embedding_consrtuction('video', user_seq)
            if idx == 1:
                seqs = torch.stack([seqs, user_seq_emb], dim=0)
            else:
                seqs = torch.cat([seqs, torch.unsqueeze(user_seq_emb, dim=0)], dim=0)

        return seqs

    def pred_batch_seq_emb_construction(self, log_seqs, user_embedding):
        seqs = self.embedding_consrtuction('video', log_seqs[0])
        for idx in range(1, len(log_seqs)):
            user_seq = log_seqs[idx]
            user_seq_emb = self.embedding_consrtuction('video', user_seq)
            if idx == 1:
                seqs = torch.stack([seqs, user_seq_emb], dim=0)
            else:
                seqs = torch.cat([seqs, torch.unsqueeze(user_seq_emb, dim=0)], dim=0)

        repeated_user_embedding = user_embedding.expand(seqs.shape[0], -1)
        new_seqs = torch.cat((seqs, repeated_user_embedding), dim=1)
        return new_seqs

    def log2feats(self, log_seqs, user_embedding):
        seqs = self.input_batch_seq_emb_construction(log_seqs, user_embedding)
        seqs *= self.embedding_dim ** 0.5
        seqs = seqs.unsqueeze(0)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        if self.config['con_position_flag'] is True:
            seqs = torch.cat((seqs, self.pos_emb(torch.LongTensor(positions).to('cuda'))), dim=2)
        else:
            seqs += self.pos_emb(torch.LongTensor(positions).to('cuda'))
        seqs = self.emb_dropout(seqs)

        """search the missing positions and mark them as masks"""
        timeline_mask = torch.BoolTensor(log_seqs == 0).to('cuda')

        """set the filled positions as embedding 0"""
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        """construct a matrix whose upper triangle elements are true"""
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device='cuda'))

        block_routes = []
        block_out_pro_max = []
        """the shape of seqs is (bz, maxlen, hidden)"""
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs_real = seqs * ~timeline_mask.unsqueeze(-1)
            seqs, routes, out_prob_max = self.forward_layers[i](seqs, seqs_real, user_embedding)
            seqs *= ~timeline_mask.unsqueeze(-1)
            block_routes.append(routes)
            block_out_pro_max.append(out_prob_max)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats, block_routes, block_out_pro_max

    def forward(self, user_indices, log_seqs, pos_seqs):  # for training   
        user_embedding = self.embedding_consrtuction('user', user_indices)
        log_feats, block_routes, block_out_pro_max = self.log2feats(log_seqs, user_embedding)  # user_ids hasn't been used yet
        new_log_feats = log_feats
        pos_embs = self.pred_batch_seq_emb_construction(pos_seqs, user_embedding).unsqueeze(0)
        vector = torch.cat([new_log_feats, pos_embs], dim=-1)
        reshaped_vector = vector.view(-1, vector.shape[-1])
        for idx, _ in enumerate(range(len(self.fc_layers))):
            reshaped_vector = self.fc_layers[idx](reshaped_vector)
            reshaped_vector = torch.nn.ReLU()(reshaped_vector)
        pos_logits = self.affine_output(reshaped_vector)
        pos_pred = self.logistic(pos_logits)
        reshaped_pos_pred = pos_pred.view(len(user_indices), -1)
        return reshaped_pos_pred, block_routes, block_out_pro_max  # pos_pred, neg_pred

    def predict(self, user_indices, log_seqs, item_indices):  # for inference
        user_embedding = self.embedding_consrtuction('user', user_indices)
        log_feats, block_routes, block_out_pro_max = self.log2feats(log_seqs, user_embedding)  # user_ids hasn't been used yet
        new_log_feats = log_feats
        final_feat = new_log_feats[:, -1, :]  # only use last QKV classifier, a waste
        item_embs = self.pred_batch_seq_emb_construction(item_indices, user_embedding)  # (U, I, C)
        vector = torch.cat([final_feat, torch.squeeze(item_embs, dim=1)], dim=-1)
        reshaped_vector = vector.view(-1, vector.shape[-1])
        for idx, _ in enumerate(range(len(self.fc_layers))):
            reshaped_vector = self.fc_layers[idx](reshaped_vector)
            reshaped_vector = torch.nn.ReLU()(reshaped_vector)
        pos_logits = self.affine_output(reshaped_vector)
        pos_pred = self.logistic(pos_logits)
        return pos_pred  # pos_pred, neg_pred


class SASRecEngine(Engine):

    def __init__(self, config, user_features, video_features, selected_user_features, selected_video_features):
        self.config = config
        self.model = SASRec(self.config, user_features, video_features, selected_user_features, selected_video_features)
        self.model.cuda()
        super(SASRecEngine, self).__init__(self.config)
        print(self.model)
        print(get_parameter_number(self.model))
        print(self.model.state_dict().keys())