import numpy as np
import pandas as pd
import torch
import os
from torch import nn
from Mole_Bert import molebert       #pip install transformer-pytorch
from ChemBERTa import chemberta
# import transformershuggingface as ppbhuggingface
from transformers import BertModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import torch
import random
from transformers import set_seed
def chemmolefusion(mirna_list):
    seed_value = 42

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    set_seed(seed_value)
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
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
        s = output[0].data.cpu().numpy()
        list2.append(s)
    list1=[]
    for i in range(x):
        data=list2[i]
        d=data.mean(axis=1)
        feat=d[0].tolist()
        list1.append(feat)

    feature_chembert = pd.DataFrame(list1)
    feature_mole = molebert(mirna_list)

    feature_fusion = pd.concat([feature_chembert,feature_mole],axis=1)
    return feature_fusion

