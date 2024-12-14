import json
import torch
import numpy as np
def get_value(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def gen_result(res, test=False, file_path=None, class_name=None):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_value(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return micro_precision,macro_precision,macro_recall,macro_f1

def eval_data_types(target,prediction,num_labels):
    ground_truth_v2 = []
    predictions_v2 = []
    for i in target:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        ground_truth_v2.append(v)
    for i in prediction:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        predictions_v2.append(v)

    res = []
    for i in range(num_labels):
        res.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    y_true = np.array(ground_truth_v2)
    y_pred = np.array(predictions_v2)
    for i in range(num_labels):
    
        outputs1 = y_pred[:, i]
        labels1 = y_true[:, i] 
        res[i]["TP"] += int((labels1 * outputs1).sum())
        res[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        res[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        res[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())
    class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 = gen_result(res)

    return class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1

def make_kv_string(d):
    out = []
    for k, v in d.items():
        if isinstance(v, float):
            out.append("{} {:.4f}".format(k, v))
        else:
            out.append("{} {}".format(k, v))

    return " ".join(out)

class Data_Process():
    def __init__(self):
        self.word2id = json.load(open('./word2id.json', "r"))
        self.charge2id = json.load(open('./charge2id.json'))
        self.article2id = json.load(open('./article2id.json'))
        self.time2id = json.load(open('./time2id.json'))
        self.symbol = [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”"]
        self.last_symbol = ["?", "。", "？"]
        self.charge2detail = json.load(open('./charge_details.json','r'))
        self.sent_max_len = 200
        self.law = json.load(open('./law.json'))
    def transform(self, word):
        if not (word in self.word2id.keys()):
            return self.word2id["UNK"]
        else:
            return self.word2id[word]
    
    def parse(self, sent):
        result = []
        sent = sent.strip().split()
        for word in sent:
            if len(word) == 0:
                continue
            if word in self.symbol:
                continue
            result.append(word)
        return result

    def parse_rationale(self, sent):
        result = []
        sent = sent.strip().split()
        for word in sent:
            if len(word) == 0:
                continue
            result.append(word)
        return result

    def seq2tensor(self, sents, max_len=350):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, max_len)

        sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                #print(word)
                sent_tensor[s_id][w_id] = self.transform(word) 
        return sent_tensor,sent_len
    


    def rationale_process(self,data):
        fact_all = []
        charge_label = []
        article_label = []
        time_label = []
        facts_ori = []
        for index,line in enumerate(data):
            if index>100:
                line = json.loads(line)
                fact = line['fact'].split('上述 事实')[0]
                facts_ori.append(fact)
                fact_all.append(self.parse_rationale(fact))

                # fact = line['fact'].replace(" ", "")
                fact  = line['fact'].split('上述 事实')[0].replace(" ", "")
            else:
                line = json.loads(line)
                fact = line['fact']
                facts_ori.append(fact)
                fact_all.append(self.parse_rationale(fact))

                # fact = line['fact'].replace(" ", "")
                fact  = line['fact'].replace(" ", "")
            fact  = line['fact'].replace(" ", "")
            
            adc = line['adc_str'].replace(" ", "")
            ssc = line['ssc_str'].replace(" ", "")
            dsc = line['dsc_str'].replace(" ", "")
            sc =  line['ssc_str'].replace(" ", "")  + line['dsc_str'].replace(" ", "")

            rationale_adc = [0 for i in range(len(fact))]
            rationale_ssc = [0 for i in range(len(fact))]
            rationale_dsc = [0 for i in range(len(fact))]
            rationale_sc = [0 for i in range(len(fact))]
            # print(len(rationale_dsc))
            for adc_index in adc:
                rationale_adc[ fact.find(adc_index) :  fact.find(adc_index) + len(adc_index) ] = [1 for k in range(len(adc_index))]
                
            for adc_index in ssc:
                rationale_ssc[ fact.find(adc_index) :  fact.find(adc_index) + len(adc_index) ] = [1 for k in range(len(adc_index))]
            
            for adc_index in dsc:
                rationale_dsc[ fact.find(adc_index) :  fact.find(adc_index) + len(adc_index) ] = [1 for k in range(len(adc_index))]
            
            for adc_index in sc:
                rationale_sc[ fact.find(adc_index) :  fact.find(adc_index) + len(adc_index) ] = [1 for k in range(len(adc_index))]
            
            # print(len(rationale_dsc))
        documents,sent_lent = self.seq2tensor(fact_all,max_len=350)
        return rationale_adc,rationale_ssc,rationale_dsc,rationale_sc,documents,sent_lent,facts_ori

    def process_data(self,data):
        fact_all = []
        charge_label = []
        article_label = []
        time_label = []
        for index,line in enumerate(data):

            line = json.loads(line)
            # print(line)
            fact = line['fact']
            charge = line['charge']
            
            article = line['article']
            if line['meta']['term_of_imprisonment']['death_penalty'] == True or line['meta']['term_of_imprisonment']['life_imprisonment'] == True:
                time_labels = 0
            else:
                time_labels = self.time2id[str(line['meta']['term_of_imprisonment']['imprisonment'])]
  
            charge_label.append(self.charge2id[charge[0]])
            article_label.append(self.article2id[str(article[0])])

            
            time_label.append(int(time_labels))

            fact_all.append(self.parse(fact))

        article_label = torch.tensor(article_label,dtype=torch.long)
        charge_label = torch.tensor(charge_label,dtype=torch.long)
        time_label = torch.tensor(time_label,dtype=torch.long)

        documents,sent_lent = self.seq2tensor(fact_all,max_len=350)
        return charge_label,article_label,time_label,documents,sent_lent

 