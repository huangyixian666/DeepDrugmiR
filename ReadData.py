import pandas as pd
import numpy as np
# from tensorflow.keras.datasets import mnist
# from keras.utils import np_utils
# from keras.utils.vis_utils import plot_model
# from tensorflow.keras.models import Sequential,Model
# from tensorflow.keras.layers import Dense,Input,Conv1D,LSTM,MaxPooling1D,Flatten,Dropout,concatenate,Reshape,GlobalAveragePooling1D,BatchNormalization
# from keras.layers.recurrent import SimpleRNN
# from keras import layers
# from keras.optimizers import Adam
# from keras.optimizers import adam_v2
# from CapsuleLayer import Capsule
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from bert import bert
from Mole_Bert import molebert
from ChemBERTa import chemberta
from ChemBERTa_mole_fusion import chemmolefusion
from Capsule_MPNN import *
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def newdata(dti_fname, mirna_encoder, drug_encoder):
    dti_df = pd.read_csv(dti_fname)
    dti_df['sequence'] = dti_df['sequence'].str.replace('U', 'T')

    data1 = dti_df

    mirna_list = data1['sequence'].tolist()

    length = data1['sequence'].map(lambda x: len(str(x)))
    print("Sequence_max_length:" + str(length.max()))

    if mirna_encoder == "bert":
        mirna_df = bert(mirna_list)

    drug_df_list = []

    if drug_encoder == "chemmolefusion":
        drug_df = chemmolefusion(data1['SMILES'].tolist())
        drug_df_list.append(drug_df)

    y = data1['expression'].tolist()
    return np.array(mirna_df), np.array(drug_df_list), y