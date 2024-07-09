import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import warnings
warnings.filterwarnings('ignore')
import torch
import random
from transformers import set_seed
def bert(mirna_list):
    seed_value = 42

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    set_seed(seed_value)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M",trust_remote_code=True,config=config)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", do_lower_case=False)
    x=len(mirna_list)
    list2=[]
    for i in range(x):
        print(i)
        sequence_Example=mirna_list[i]
        if len(mirna_list[i])>2000:
            sequence_Example=' '.join(mirna_list[i].split()[:1000])
        print('mirna-length:',len(sequence_Example))
        encoded_input = tokenizer(sequence_Example, return_tensors='pt')

        output = model(**encoded_input)
        s = output[0].data.cpu().numpy() # 数据类型转换
        list2.append(s)
        encoded_input=0
        sequence_Example=0
        s=0
        output=0
    list1=[]
    for i in range(x):
        data=list2[i]
        d=data.mean(axis=1)
        feat=d[0].tolist()
        list1.append(feat)

    features=pd.DataFrame(list1)
    return features