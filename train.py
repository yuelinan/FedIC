import json
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from utils import Data_Process,eval_data_types,make_kv_string
import torch
import torch.nn as nn
from model import FedIC_projector,FedIC,CLUB_NCE
import logging
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os
import argparse
from word2vec import toembedding
import copy
import prettytable as pt
from sklearn.metrics import precision_recall_fscore_support
parser = argparse.ArgumentParser(description='VMask classificer')

parser.add_argument('--model_name', type=str, default='RNP', help='model name')
parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate')

parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')

parser.add_argument('--alpha_rationle', type=float, default=0.2, help='alpha_rationle')
parser.add_argument('--infor_loss', type=float, default=0.2, help='infor_loss')
parser.add_argument('--regular', type=float, default=0.2, help='regular')

parser.add_argument('--embed_dim', type=int, default=200, help='original number of embedding dimension')
parser.add_argument('--lstm_hidden_dim', type=int, default=150, help='number of hidden dimension')
parser.add_argument('--mask_hidden_dim', type=int, default=200, help='number of hidden dimension')
parser.add_argument('--lstm_hidden_layer', type=int, default=1, help='number of hidden layers')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--types', type=str, default='legal', help='data_type')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--class_num', type=int, default=11, help='class_num')
parser.add_argument('--save_path', type=str, default='', help='save_path')
parser.add_argument('--save_name', type=str, default='model_', help='model_name')
parser.add_argument('--is_emb', type=str, default='training', help='is_emb')
parser.add_argument('--date', type=str, default='1128', help='is_emb')
parser.add_argument('--min_lr', type=float, default=5e-5)
parser.add_argument('--lr_decay', type=float, default=0.97)
parser.add_argument('--abs', type=int, default=1)
parser.add_argument('--data_type', choices=["length", "charge_num"], default="length")
parser.add_argument('--loss_type', choices=["infonce", "kl", "mse","js"], default="infonce")
parser.add_argument('--criterion_type', type=str, default='all', choices=['all','only_1','only_2'])

args = parser.parse_args()
if args.class_num==115:
    type_log = 'charge'
elif  args.class_num==99:
    type_log = 'article'
else:
    type_log = 'term'
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def random_seed():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.is_emb=='training':
    embedding = toembedding()
else:
    embedding = 1

for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))


if args.data_type == 'length':
    pass
    ###### length
    client1_idx = []
    client2_idx = []
    client3_idx = []
    client4_idx = []

    client_number = 4
    f = open('./train.json','r')
    for index,line in enumerate(f):

        line = json.loads(line)
        fact = line['fact']
        fact = fact.replace(' ','')
        json_str = json.dumps(line, ensure_ascii=False)
        if len(fact)<200:
            client1_idx.append(json_str)
        elif len(fact)>=200 and len(fact)<315:
            client2_idx.append(json_str)
        elif len(fact)>=315 and len(fact)<450:
            client3_idx.append(json_str)
        else:
            client4_idx.append(json_str)

if args.data_type == 'charge_num':
    ######## charge
    charge2id = json.load(open('./charge2id.json'))
    client1_idx = []
    client2_idx = []
    client3_idx = []
    client4_idx = []
    client_number = 4
    f = open('./train.json','r')
    a=0
    b=0
    c=0
    d=0
    for index,line in enumerate(f):
        # if index>10:break

        line = json.loads(line)
        charge = line['charge']
        charge1 = charge2id[charge[0]]
        json_str = json.dumps(line, ensure_ascii=False)
        if charge1<80 and a<=26154:
            client1_idx.append(json_str)
            a+=1
            continue
        elif charge1>28 and charge1<85 and b<=19923:
            client2_idx.append(json_str)
            b+=1
            continue
        elif charge1>30 and charge1<90 and c<=34139:
            client3_idx.append(json_str)
            c+=1
            continue
        else:
            client4_idx.append(json_str)
            d+=1
            continue

    ########

partition_dicts = [None] * client_number
data_all = [client1_idx,client2_idx,client3_idx,client4_idx]
len_all = len(client1_idx)+len(client2_idx)+len(client3_idx)+len(client4_idx)

for index,train_client in enumerate(data_all):
    train_loader = DataLoader(train_client, batch_size=args.batch_size, shuffle=True, num_workers = 0)
    partition_dicts[index] = {"train": train_loader}

client_weights = [len(client1_idx)/len_all, len(client2_idx)/len_all,len(client3_idx)/len_all,len(client4_idx)/len_all]
print(client_weights)
test_data = []
f_test = open('./test.json','r')

for index,lines in enumerate(f_test):
    # if index>10:break
    test_data.append(lines)
if args.is_emb!='training':
    test_data = test_data[0:10]
print(len(test_data))
test_dataloader = DataLoader(test_data, batch_size = args.batch_size, shuffle=False, num_workers=0, drop_last=False)


rationale_test_data = []
f_rationale_test = open('./rationale_LJP2.json','r')
for index,lines in enumerate(f_rationale_test):
    # if index>=12 and index<=12:   
    rationale_test_data.append(lines)

print(len(rationale_test_data))
rationale_test_dataloader = DataLoader(rationale_test_data, batch_size = 1, shuffle=False, num_workers=0, drop_last=False)
id2word = json.load(open('./idx2word.json', "r"))


def main():
    process = Data_Process()
    server_model = FedIC(args, embedding,args.class_num).to(args.device)
    d_model = CLUB_NCE().to(args.device)
    models = [copy.deepcopy(server_model) for idx in range(client_number)]
    d_models = [copy.deepcopy(d_model) for idx in range(client_number)]


    bias_augmention_model = FedIC_projector(args,embedding).to(args.device)
    bias_augmention_models = [copy.deepcopy(bias_augmention_model) for idx in range(client_number)]
    bias_optimizers = [optim.Adam(bias_augmention_model.parameters(), lr=args.lr, weight_decay=args.weight_decay) for bias_augmention_model in bias_augmention_models]

    def train_bias_augmention_model(args, train_loader, bias_augmention_model, bias_optimizer, bias_model, unbias_model,process,d_optimizer,Dmodel,criterion_type='all'):
        
        bias_augmention_model.train()
        Dmodel.train()


        for param in bias_model.parameters():
            param.requires_grad = False
            
        for param in unbias_model.parameters():
            param.requires_grad = False
        bias_model.train()
        unbias_model.train()
        for step, batch in enumerate(train_loader):
            # if step>5:break
            charge_label,article_label,time_label,documents,sent_lent = process.process_data(batch)
            documents = documents.to(args.device)
            sent_lent = sent_lent.to(args.device)
            charge_label = charge_label.to(args.device)
            article_label = article_label.to(args.device)
            time_label = time_label.to(args.device)
            new_embed = bias_augmention_model(documents,bias_model)
            if args.class_num==115:
                criterion_label = charge_label
            elif  args.class_num==99:
                criterion_label = article_label
            else:
                criterion_label = time_label

            pred_bias,x_samples,y_samples = bias_model.eval_ic_forward(documents,sent_lent,new_embed,bias=1)
        
            lower_bound, upper_bound = Dmodel(x_samples.detach(),y_samples.detach())
            Dloss = -lower_bound
            d_optimizer.zero_grad()
            Dloss.backward()
            d_optimizer.step()

            pred_bias,x_samples,y_samples = bias_model.eval_ic_forward(documents,sent_lent,new_embed,bias=1)
            lower_bound, upper_bound = Dmodel(x_samples,y_samples)

            pred_unbias,_ = unbias_model.eval_ic_forward(documents,sent_lent,new_embed,bias=0)

            loss_bias = criterion(pred_bias,criterion_label)
            loss_unbias = criterion(pred_unbias,criterion_label)

            if criterion_type == 'all':
                loss =  loss_unbias + loss_bias  # max MI
                loss += 0.01*upper_bound  # min MI
            elif criterion_type ==  'only_1':
                loss = loss_bias
                loss += 0.01*upper_bound
            elif criterion_type ==  'only_2':
                loss =  loss_unbias

            bias_optimizer.zero_grad()
            loss.backward()
            bias_optimizer.step()
        
        for param in bias_model.parameters():
            param.requires_grad = True
        
        for param in unbias_model.parameters():
            param.requires_grad = True


    def get_opt(args,model):
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def communication(server_model, models, client_weights):
        client_num = len(models)
        with torch.no_grad():
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                # print(client_num)
                for client_idx in range(client_num):
                    #  print(client_weights[client_idx])
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)

                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        return server_model, models
    
    for epoch in range(1, args.epochs+1):
        optimizers = [ get_opt(args, models[idx])  for idx in range(client_number)]
        d_optimizers = [ get_opt(args, d_models[idx])  for idx in range(client_number)]
        for client_idx, model in enumerate(models):
            temp_time=0
            train_loader = partition_dicts[client_idx]['train']
            optimizer = optimizers[client_idx]
            d_optimizer = d_optimizers[client_idx]
            Dmodel = d_models[client_idx]
            for i in range(3):
                if epoch==1:
                    for step,batch in enumerate(tqdm(train_loader)):
                        # if step>5:break
                        model.train()
                        optimizer.zero_grad()
                        charge_label,article_label,time_label,documents,sent_lent = process.process_data(batch)
                    
                        documents = documents.to(args.device)
                        charge_label = charge_label.to(args.device)
                        article_label = article_label.to(args.device)
                        time_label = time_label.to(args.device)
                    
                        sent_lent = sent_lent.to(args.device)
                        output, rationales = model(documents,sent_lent)
                        if args.class_num==115:
                            criterion_label = charge_label
                        elif  args.class_num==99:
                            criterion_label = article_label
                        else:
                            criterion_label = time_label
                        loss_charge = criterion(output,criterion_label)
                        loss =  loss_charge + args.infor_loss * model.infor_loss + args.regular * model.regular
                        loss.backward()
                        optimizer.step()
                else:
                    if temp_time ==0:
                        for param in bias_augmention_model.parameters():
                            param.requires_grad = True
                        train_bias_augmention_model(args, train_loader, bias_augmention_models[client_idx], bias_optimizers[client_idx], bias_models[client_idx], server_model,process,d_optimizer,Dmodel,args.criterion_type)
                        temp_time=1
                    else:
                        for step,batch in enumerate(tqdm(train_loader)):
                            # if step>5:break
                            model.train()
                            for param in bias_augmention_model.parameters():
                                param.requires_grad = False
                            optimizer.zero_grad()
                            charge_label,article_label,time_label,documents,sent_lent = process.process_data(batch)
                        
                            documents = documents.to(args.device)
                            charge_label = charge_label.to(args.device)
                            article_label = article_label.to(args.device)
                            time_label = time_label.to(args.device)
                        
                            sent_lent = sent_lent.to(args.device)
                            output, rationales = model(documents,sent_lent,bias_augmention_model)
                            if args.class_num==115:
                                criterion_label = charge_label
                            elif  args.class_num==99:
                                criterion_label = article_label
                            else:
                                criterion_label = time_label
                            loss_charge = criterion(output,criterion_label)
                            loss_cour = criterion(model.output_cour,criterion_label)
                            loss =  loss_charge + loss_cour + args.infor_loss * model.infor_loss + args.regular * model.regular
                            loss.backward()
                            optimizer.step()

        with torch.no_grad():
            # bias_models = [copy.deepcopy(bias_model) for bias_model in models]
            a = models[0]
            b = models[1]
            c = models[2]
            d = models[3]
            bias_models = [a,b,c,d]
            server_model, models = communication( server_model, models, client_weights)

        server_model.eval()
        predictions_article = []
        predictions_charge = []
        predictions_time = []

        predictions_article_rnp = []
        predictions_charge_rnp = []
        predictions_time_rnp = []

        true_article = []
        true_charge = []
        true_time = []
        for step,batch in enumerate(tqdm(test_dataloader)):
            charge_label,article_label,time_label,documents,sent_lent = process.process_data(batch)
            documents = documents.to(args.device)
            sent_lent = sent_lent.to(args.device)
            true_article.extend(article_label.numpy())
            true_charge.extend(charge_label.numpy())
            true_time.extend(time_label.numpy())

            with torch.no_grad():
                charge_out,_ = server_model(documents,sent_lent)
            charge_pred = charge_out.cpu().argmax(dim=1).numpy()
            predictions_charge.extend(charge_pred)

        dev_eval = {}
        if args.class_num==115:
            charge_acc, charge_p, charge_r, charge_f1 = eval_data_types(true_charge,predictions_charge,num_labels=args.class_num)
        if args.class_num==99:
            charge_acc, charge_p, charge_r, charge_f1 = eval_data_types(true_article,predictions_charge,num_labels=args.class_num)
        if args.class_num==11:
            charge_acc, charge_p, charge_r, charge_f1 = eval_data_types(true_time,predictions_charge,num_labels=args.class_num)

        table = pt.PrettyTable(['types ','     Acc   ','          P          ', '          R          ', '      F1          ' ]) 
        table.add_row(['charge',  charge_acc, charge_p, charge_r, charge_f1 ])

        logger.info(table)

        

        #######################
        server_model.eval()
        predictions_adc = []
        predictions_ssc = []
        predictions_dsc = []
        predictions_sc = []
        predictions_adc_charge = []

        true_adc = []
        true_ssc = []
        true_dsc = []
        true_sc = []
        percent_adcs = 0
        percent_sscs = 0
        percent_dscs = 0
        percent_adc_charges = 0
        for step,batch in enumerate(tqdm(rationale_test_dataloader)):
            rationale_adc,rationale_ssc,rationale_dsc,rationale_sc,documents,sent_lent,facts_ori = process.rationale_process(batch)
            documents = documents.to(args.device)
            sent_lent = sent_lent.to(args.device)

            true_adc.extend(rationale_adc)
            true_ssc.extend(rationale_ssc)
            true_dsc.extend(rationale_dsc)
            true_sc.extend(rationale_sc)

            with torch.no_grad():
                charge_out,mask_number = model(documents,sent_lent)

            #***********************************************************
            def get_one_hot_pred_rationale(mask_number,documents,id2word,facts_ori):
                mask_number = mask_number.squeeze(0).squeeze(-1)
            
                percent = int(mask_number.sum()) / mask_number.size(0)
                
                mask_fact = (documents.squeeze(0) * mask_number.long()).cpu().numpy()
                xx = ''
                document = documents.squeeze(0).cpu().numpy()
                mask_number = mask_number.long().cpu().numpy()
                xx = []
                pred_adc_rationale = []
                facts_ori = facts_ori[0].split(' ')
                
                sums = 0
                for index,data in enumerate(mask_number):
                    if data==1:
                        xx = [1 for k in range(len(facts_ori[index]))]
                        
                        pred_adc_rationale.extend(xx)
                        
                    else:
                        xx = [0 for k in range(len(facts_ori[index]))]
                        pred_adc_rationale.extend(xx)
                        
                
                return pred_adc_rationale,percent*100

            pred_adc_rationale_charge,percent_adc_charge = get_one_hot_pred_rationale(mask_number,documents,id2word,facts_ori)

            percent_adc_charges += percent_adc_charge
           
            predictions_adc_charge.extend(pred_adc_rationale_charge)
        
        if args.class_num==115 or args.class_num==99:
            precision_tagging_charge, recall_tagging_charge, f1_tagging_charge, _ = precision_recall_fscore_support(
            np.array(true_adc), np.array(predictions_adc_charge), labels=[1])
            log_num = 'charge or article'
        else:
            precision_tagging_charge, recall_tagging_charge, f1_tagging_charge, _ = precision_recall_fscore_support(
                np.array(true_sc), np.array(predictions_adc_charge), labels=[1])
            log_num = 'term'

        table = pt.PrettyTable(['types ', '          P          ', '          R          ', '      F1          ', '      Percent          ' ]) 
        table.add_row([log_num,  precision_tagging_charge[0], recall_tagging_charge[0], f1_tagging_charge[0], percent_adc_charges/len(rationale_test_data) ])
        logger.info(table)

        PATH = args.save_path+str(args.class_num)+'-'+args.model_name.lower()+'_'+str(epoch)
        torch.save(server_model.state_dict(), PATH)



if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    main()