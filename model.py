import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from torch.autograd import Variable
import numpy as np
import random


class FedIC(nn.Module):
    def __init__(self,args, embedding,class_num):
        super(FedIC, self).__init__()
        print('FedIC')
        self.embs = nn.Embedding(339503, 200)
        if args.is_emb=='training':
            self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad = False
        self.device = args.device
        
        self.encoder = nn.GRU(args.embed_dim, args.lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.re_encoder = nn.GRU(args.embed_dim, args.lstm_hidden_dim, batch_first=True, bidirectional=True)
        
        self.x_2_prob_z = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim*2, 2)
          )
        self.z_to_fea = nn.Linear(args.lstm_hidden_dim*2, args.lstm_hidden_dim*2)
        self.ChargeClassifier = nn.Linear(args.lstm_hidden_dim*2, class_num)
        self.alpha_rationle = args.alpha_rationle


    def forward(self, documents,sent_len,bias_augmention_model=None):
        eps = 1e-8
        embed = self.embs(documents) # batch, seq_len, embed-dim
        if bias_augmention_model==None:
            mask = torch.sign(documents).float()
            batch_size = embed.size(0)
            en_outputs,_ = self.encoder(embed)  # batch seq hid
            z_logits = self.x_2_prob_z(en_outputs) # batch seq 2

            sampled_seq = F.gumbel_softmax(z_logits,hard=True,dim=2)
            sampled_seq = sampled_seq * mask.unsqueeze(2)

            sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
            sampled_num = (sampled_num == 0).to(self.device, dtype=torch.float32)  + sampled_num
            sampled_word = embed * (sampled_seq[:,:,1].unsqueeze(2))  # batch seq hid
        
            s_w_feature,_ = self.re_encoder(sampled_word)

            s_w_feature = torch.sum(s_w_feature, dim = 1)/ sampled_num.unsqueeze(1)# batch hid
            self.final_output = s_w_feature
            output = self.ChargeClassifier(s_w_feature) 

            mask_number1 = sampled_seq[:,:,1]
            infor_loss = (mask_number1.sum(-1) / (sent_len+eps) ) - self.alpha_rationle

            self.infor_loss = torch.abs(infor_loss).mean()
            regular =  torch.abs(mask_number1[:,1:] - mask_number1[:,:-1]).sum(1) / (sent_len-1+eps)
            self.regular = regular.mean()
            return output , sampled_seq[:,:,1]

        else:
            mask = torch.sign(documents).float()
            batch_size = embed.size(0)
            en_outputs,_ = self.encoder(embed)  # batch seq hid
            z_logits = self.x_2_prob_z(en_outputs) # batch seq 2

            sampled_seq = F.gumbel_softmax(z_logits,hard=True,dim=2)
            sampled_seq = sampled_seq * mask.unsqueeze(2)

            sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
            sampled_num = (sampled_num == 0).to(self.device, dtype=torch.float32)  + sampled_num
            sampled_word = embed * (sampled_seq[:,:,1].unsqueeze(2))  # batch seq hid
        
            s_w_feature,_ = self.re_encoder(sampled_word)

            s_w_feature = torch.sum(s_w_feature, dim = 1)/ sampled_num.unsqueeze(1)# batch hid
            self.final_output = s_w_feature
            output = self.ChargeClassifier(s_w_feature) 

            mask_number1 = sampled_seq[:,:,1]
            infor_loss = (mask_number1.sum(-1) / (sent_len+eps) ) - self.alpha_rationle

            self.infor_loss = torch.abs(infor_loss).mean()
            regular =  torch.abs(mask_number1[:,1:] - mask_number1[:,:-1]).sum(1) / (sent_len-1+eps)
            self.regular = regular.mean()

            # non_rationale
            non_rationale = embed * (1-sampled_seq[:,:,1].unsqueeze(2))
            new_env,_ = bias_augmention_model.projector(non_rationale)
            new_env = torch.sum(new_env, dim = 1)/ sampled_num.unsqueeze(1)

            self.output_cour = self.ChargeClassifier(s_w_feature+new_env) 
            
            return output , sampled_seq[:,:,1]


    def eval_ic_forward(self, documents,sent_len,new_embed,bias=1):
        embed = self.embs(documents)
        mask = torch.sign(documents).float()
        
        en_outputs,_ = self.encoder(embed)  # batch seq hid
        z_logits = self.x_2_prob_z(en_outputs) # batch seq 2

        sampled_seq = F.gumbel_softmax(z_logits,hard=True,dim=2)
        sampled_seq = sampled_seq * mask.unsqueeze(2)

        sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
        sampled_num = (sampled_num == 0).to(self.device, dtype=torch.float32)  + sampled_num
        sampled_word = embed * (sampled_seq[:,:,1].unsqueeze(2))  # batch seq hid
    
        s_w_feature,_ = self.re_encoder(sampled_word)

        s_w_feature = torch.sum(s_w_feature, dim = 1)/ sampled_num.unsqueeze(1)# batch hid
        self.final_output = s_w_feature
        output = self.ChargeClassifier(s_w_feature+new_embed) 
        if bias==1:
            # min MI
            ori,_ = self.re_encoder(embed)
            ori = ori.mean(1)
            mi_min = self.minimize_covariance(s_w_feature+new_embed, ori)
            return output , s_w_feature+new_embed, ori
        else:
            return output , sampled_seq[:,:,1]

    def eval_forward(self, documents,sent_len,new_embed):
        mask = torch.sign(documents).float()
        eps = 1e-8
        batch_size = new_embed.size(0)
        en_outputs,_ = self.encoder(new_embed)  # batch seq hid
        z_logits = self.x_2_prob_z(en_outputs) # batch seq 2

        sampled_seq = F.gumbel_softmax(z_logits,hard=True,dim=2)
        sampled_seq = sampled_seq * mask.unsqueeze(2)

        sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
        sampled_num = (sampled_num == 0).to(self.device, dtype=torch.float32)  + sampled_num
        sampled_word = new_embed * (sampled_seq[:,:,1].unsqueeze(2))  # batch seq hid
       
        s_w_feature,_ = self.re_encoder(sampled_word)

        s_w_feature = torch.sum(s_w_feature, dim = 1)/ sampled_num.unsqueeze(1)# batch hid
        self.final_output = s_w_feature
        output = self.ChargeClassifier(s_w_feature) 

            
        return output , sampled_seq[:,:,1]

class CLUB_NCE(nn.Module):
    def __init__(self, emb_dim = 300):
        super(CLUB_NCE, self).__init__()
        lstm_hidden_dim = emb_dim//2
        
        self.F_func = nn.Sequential(nn.Linear(lstm_hidden_dim*4, lstm_hidden_dim*2),
                                    #nn.Dropout(p=0.2),
                                    nn.ReLU(),
                                    nn.Linear(lstm_hidden_dim*2, 1),
                                    nn.Softplus())
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))#

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        upper_bound = T0.mean() - T1.mean()
        
        return lower_bound, upper_bound

    
class FedIC_projector(torch.nn.Module):

    def __init__(self,args,embedding):
        '''
            num_tasks (int): number of labels to be predicted
        '''
        super(FedIC_projector, self).__init__()
        self.embs = nn.Embedding(339503, 200)
        if args.is_emb=='training':
            self.embs.weight.data.copy_(embedding)
        self.embs.weight.requires_grad = False

        self.predictor = torch.nn.Linear(200, 200)
        self.criterion_recons = nn.MSELoss()
        self.device = args.device
        self.projector = nn.GRU(args.embed_dim, args.lstm_hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, documents,bias_model):
        embed = self.embs(documents)
        mask = torch.sign(documents).float()
        batch_size = embed.size(0)
        en_outputs,_ = bias_model.encoder(embed)  # batch seq hid
        z_logits = bias_model.x_2_prob_z(en_outputs) # batch seq 2

        sampled_seq = F.gumbel_softmax(z_logits,hard=True,dim=2)
        sampled_seq = (1-sampled_seq) * mask.unsqueeze(2)

        sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
        sampled_num = (sampled_num == 0).to(self.device, dtype=torch.float32)  + sampled_num

        non_rationale = embed * (sampled_seq[:,:,1].unsqueeze(2))  # batch seq hid
        new_env,_ = self.projector(non_rationale)
        new_env = torch.sum(new_env, dim = 1)/ sampled_num.unsqueeze(1)

        return new_env
        


