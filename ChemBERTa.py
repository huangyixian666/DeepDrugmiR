import numpy as np
import pandas as pd
import torch
import os
from torch import nn
from Mole_Bert import molebert       #pip install transformer-pytorch
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel, AutoModelForMaskedLM, T5Tokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')
def chemberta(mirna_list):
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    #
    model = BertModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    x=len(mirna_list)
    list2=[]
    for i in range(x):
        print(i)
        sequence_Example=mirna_list[i]
        print('SMILE-length:',len(sequence_Example))
        max_length = model.config.max_position_embeddings
        encoded_input = tokenizer(sequence_Example,max_length=max_length, return_tensors='pt')

        output = model(**encoded_input)
        s = output[0].data.cpu().numpy() # 数据类型转换
        list2.append(s)
    list1=[]
    for i in range(x):
        data=list2[i]
        d=data.mean(axis=1)
        feat=d[0].tolist()
        list1.append(feat)
    feature = pd.DataFrame(list1)
    return feature

