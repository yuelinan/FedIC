import json
import numpy as np
import torch
from tqdm import tqdm
import random
def toembedding():
    random.seed(42)
    word2vec = json.load(open('./word2vec.json', "r"))
    word2vec['UNK'] = np.random.randn(200).tolist() 
    word2vec['Padding'] = [0. for i in range(200)]
    embedding = torch.FloatTensor(339503, 200).zero_()
    word2id = json.load(open('./word2id.json', "r"))
    idx2word = json.load(open('./idx2word.json', "r"))
    # for i in range(339503):
    for i in tqdm(range(339503)):
        # print(str(i))
        word = idx2word[str(i)]
        result = list(map(float, word2vec[word]))
        #两种写入方式一样
        embedding[i] = torch.from_numpy(np.array(result))
    return embedding