# import os
# #os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import math
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ## LIMIT GPU USAGE
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)
import torch

# from keras.optimizers import adam_v2
from model import *
from Capsule_MPNN import *
from ReadData import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from keras.optimizers import Adam
import keras.backend as K


# def custom_f1(y_true, y_pred):
#     y_pred = np.argmax(y_pred, -1)
#     y_true = np.argmax(np.array(y_true)[:,1],-1)
# return f1_score(y_true,y_pred)

def custom_f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = (true_positives + K.epsilon()) / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pickle


def fitting(train_m, test_m, train_d, test_d, train_y, test_y, model_type, lr, ep, path, taxonomy,dti, batchsize=32):
    # global model_1
    adam = Adam(learning_rate=lr)
    # adam=tf.keras.optimizers.Adam(learning_rate=lr)
    parameters_string = model_type
    train_path = os.path.join(path, parameters_string)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    fw = open(model_type + '/test_m.txt', 'wb')
    pickle.dump(test_m, fw)
    fw.close()
    fw = open(model_type + '/test_d.txt', 'wb')
    pickle.dump(test_d, fw)
    fw.close()
    fw = open(model_type + '/test_y.txt', 'wb')
    pickle.dump(test_y, fw)
    fw.close()

    cv = KFold(n_splits=5, random_state=33, shuffle=True)

    if "bert_chemmolefusion_capsule" in model_type:
        param_grid = {
            "mirna_dense": [200,400],
            "batch_size": [32],
            "message_units": [64],
            "message_steps": [4],
            "num_attention_heads": [8],
            "dense_units": [512],
            "num_capsule": [2],
            "routings": [3,6],
            "kernel_size": [5,10],
        }
        fw = open(os.path.join(train_path, "search_process.txt"), "w")
        fw.close()
        best_score, best_mirna_dense, best_batch_size, best_message_units, best_message_steps, best_num_attention_heads, best_dense_units, best_num_capsule, best_routings, best_kernel_size = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for mirna_dense in param_grid["mirna_dense"]:
            for batch_size in param_grid["batch_size"]:
                for message_units in param_grid["message_units"]:
                    for message_steps in param_grid["message_steps"]:
                        for num_attention_heads in param_grid["num_attention_heads"]:
                            for dense_units in param_grid["dense_units"]:
                                for num_capsule in param_grid["num_capsule"]:
                                    for routings in param_grid["routings"]:
                                        for kernel_size in param_grid["kernel_size"]:
                                            all_acc_scores = []
                                            all_sensitivity = []
                                            all_specificity = []
                                            all_aucroc = []
                                            all_aupr = []
                                            all_f1 = []
                                            fw = open(os.path.join(train_path, "search_process.txt"), "a")
                                            fw.write(
                                                "mirna_dense,batch_size,message_units,message_steps,num_attention_heads,dense_units,num_capsule,routings,kernel_size\n")
                                            fw.write(str(mirna_dense) + "," + str(batch_size) + "," + str(
                                                message_units) + "," + str(message_steps) + "," + str(
                                                num_attention_heads) + "," + str(dense_units) + "," + str(
                                                num_capsule) + "," + str(routings) + "," + str(kernel_size) + "\n")
                                            fw.close()
                                            for i, (train_index, val_index) in enumerate(cv.split(train_m,train_d)):
                                                train_p_train, train_p_val = train_m[train_index], train_m[val_index]
                                                train_d_train, train_d_val = np.array(train_d)[train_index], np.array(train_d)[val_index]
                                                train_y_train, train_y_val = train_y[train_index], train_y[val_index]
                                                param = {"mirna_dense": mirna_dense,
                                                         "drug_dense": mirna_dense, "kernel_size": kernel_size,
                                                         "num_capsule": num_capsule, "routings": routings}
                                                model_1 = model_bert_chemmolefusion_capsule(param=param)
                                                model_1.compile(loss="binary_crossentropy", optimizer=adam,
                                                                metrics=[custom_f1, 'accuracy', 'AUC',
                                                                         tf.keras.metrics.Precision(),
                                                                         tf.keras.metrics.Recall(),
                                                                         tf.keras.metrics.TruePositives(),
                                                                         tf.keras.metrics.TrueNegatives(),
                                                                         tf.keras.metrics.FalsePositives(),
                                                                         tf.keras.metrics.FalseNegatives()])
                                                lrate = LearningRateScheduler(step_decay)
                                                Early = EarlyStopping(monitor="accuracy", mode='max', patience=50,
                                                                      verbose=1, restore_best_weights=True)
                                                model_1.fit([train_p_train,train_d_train],train_y_train, epochs=ep,
                                                            batch_size=batchsize,
                                                            verbose=2,
                                                            callbacks=[lrate, Early])  # callbacks=[lrate,Early]
                                                # pred0 = model_1.predict([train_m,train_m])
                                                pred0 = model_1.predict([train_p_val, train_d_val])
                                                pred = np.argmax(pred0, -1)
                                                confusion = metrics.confusion_matrix(np.array(train_y_val)[:, 1], pred)
                                                TN = confusion[0, 0]
                                                FP = confusion[0, 1]
                                                accuracy = accuracy_score(np.array(train_y_val)[:, 1], pred)
                                                specificity = TN / float(TN + FP)
        
                                                fpr0, tpr0, thresholds0 = metrics.roc_curve(np.array(train_y_val)[:, 1],
                                                                                            pred)
                                                f1 = f1_score(np.array(train_y_val)[:, 1], pred)
                                                auc_v = metrics.auc(fpr0, tpr0)
                                                precision0, recall, thresholds = metrics.precision_recall_curve(
                                                    np.array(train_y_val)[:, 1], pred)
                                                area = metrics.auc(recall, precision0)
                                                recall = recall_score(np.array(train_y_val)[:, 1], pred)
                                                fw = open(os.path.join(train_path, "search_process.txt"), "a")
                                                # fw.write("#####Fold" + str(i) + "\n")
                                                fw.write("accuracy:" + str(round(accuracy, 4)) + "\t")
                                                fw.write("specificity:" + str(round(specificity, 4)) + "\t")
                                                fw.write("sensitivity:" + str(round(recall, 4)) + "\t")
                                                fw.write("aucroc:" + str(round(auc_v, 4)) + "\t")
                                                fw.write("aupr:" + str(round(area, 4)) + "\t")
                                                fw.write("f1:" + str(round(f1, 4)) + "\n")
                                                all_acc_scores.append(accuracy)
                                                all_f1.append(f1)
                                                all_aupr.append(area)
                                                all_aucroc.append(auc_v)
                                                all_specificity.append(specificity)
                                                all_sensitivity.append(recall)
                                                fw.close()
        
                                            score = np.mean(all_acc_scores)
                                            if score > best_score:
                                                fw = open(os.path.join(train_path, "search_process.txt"), "a")
                                                fw.write("accuracy:" + str(score) + " > best_score:" + str(
                                                    best_score) + "\n")
                                                fw.close()
                                                best_score, best_mirna_dense, best_batch_size, best_message_units, best_message_steps, best_num_attention_heads, best_dense_units, best_num_capsule, best_routings, best_kernel_size = score, mirna_dense, batch_size, message_units, message_steps, num_attention_heads, dense_units, num_capsule, routings, kernel_size
        fw = open(os.path.join(train_path, "search_process.txt"), "a")
        fw.write(
            "best_mirna_dense,best_batch_size,best_message_units,best_message_steps,best_num_attention_heads,best_dense_units,best_num_capsule,best_routings,best_kernel_size\n")
        fw.write(str(best_mirna_dense) + "," + str(best_batch_size) + "," + str(
            best_message_units) + "," + str(best_message_steps) + "," + str(
            best_num_attention_heads) + "," + str(
            best_dense_units) + "," + str(best_num_capsule) + "," + str(
            best_routings) + "," + str(
            best_kernel_size) + "\n")
        fw.write(
            "accuracy,specificity,sensitivity,aucroc,aupr,f1\n")
        fw.write(str(best_score) + "±" + str(np.std(all_acc_scores)) + "," + str(
            np.mean(all_specificity)) + "±" + str(np.std(all_specificity)) + str(
            np.mean(all_sensitivity)) + "±" + str(np.std(all_sensitivity)) + str(
            np.mean(all_aucroc)) + "±" + str(np.std(all_aucroc)) + str(
            np.mean(all_aupr)) + "±" + str(np.std(all_aupr)) + str(
            np.mean(all_f1)) + "±" + str(np.std(all_f1)) + "\n")
        df = pd.DataFrame({"dataset":[dti],"classifier": [model_type], "accuracy": [
            str(best_score) + "±" + str(np.std(all_acc_scores))], "specificity": [
            str(np.mean(all_specificity)) + "±" + str(np.std(all_specificity))],
                           "sensitivity": [str(np.mean(all_sensitivity)) + "±" + str(
                               np.std(all_sensitivity))], "aucroc": [
                str(np.mean(all_aucroc)) + "±" + str(np.std(all_aucroc))], "aupr": [
                str(np.mean(all_aupr)) + "±" + str(np.std(all_aupr))],
                           "f1": [str(np.mean(all_f1)) + "±" + str(np.std(all_f1))]})
        df.to_csv("./result/5fold-performance.csv", mode="a", index=None)
        fw.close()
        adam = Adam(learning_rate=lr)
        param = {"mirna_dense": best_mirna_dense, "drug_dense": best_mirna_dense,
                 "kernel_size": 5, "num_capsule": 2, "routings": 3}
        model_1 = model_bert_chemmolefusion_capsule(param=param)
        model_1.compile(loss="binary_crossentropy", optimizer=adam,
                        metrics=[custom_f1, 'accuracy', 'AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                 tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                                 tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
        dti_name = dti.split('/')
        model_parh = os.path.join(train_path+'_'+dti_name[1], "{}.ckpt".format(taxonomy))
        csv_logger = CSVLogger(os.path.join(train_path, 'model_training.csv'))
        lrate = LearningRateScheduler(step_decay)
        Early = EarlyStopping(monitor='accuracy', mode='max', patience=50, verbose=1)
        checkpoint = ModelCheckpoint(filepath=model_parh, monitor='accuracy', mode='max', save_best_only=True,
                                     verbose=1)
        history = model_1.fit([train_m, train_d], train_y, batch_size=batchsize, epochs=ep,
                              callbacks=[lrate, Early, csv_logger, checkpoint], verbose=0)
        print("=============Train Over! ===========", flush=True)

    return history, model_1


def evaluate(model_parh, test_m, test_d, test_y, model_name):
    print("=============Start Evaluate! ===========", flush=True)
    model = load_model(model_parh, compile=False,
                           custom_objects={'Capsule': Capsule, 'Length': Length,'TransformerEncoderReadout': TransformerEncoderReadout})
    pred0 = model.predict([test_m, np.array(test_d)]) 

    pred = np.argmax(pred0, -1)
    confusion = metrics.confusion_matrix(np.array(test_y)[:, 1], pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    sensitivity = recall_score(np.array(test_y)[:, 1], pred)
    specificity = TN / float(TN + FP)
    precision = precision_score(np.array(test_y)[:, 1], pred)
    accuracy = accuracy_score(np.array(test_y)[:, 1], pred)
    f1 = f1_score(np.array(test_y)[:, 1], pred)
    aucroc = roc_auc_score(np.array(test_y)[:, 1], pred0[:, 1])
    # aupr=average_precision_score(np.array(test_y)[:,1],pred0[:,1])
    fpr0, tpr0, thresholds0 = metrics.roc_curve(np.array(test_y)[:, 1], pred0[:, 1])
    auc_v = metrics.auc(fpr0, tpr0)
    precision0, recall, thresholds = metrics.precision_recall_curve(np.array(test_y)[:, 1], pred0[:, 1])
    area = metrics.auc(recall, precision0)
    print("sensitivity", round(sensitivity, 3))
    print("specificity", round(specificity, 3))
    print("precision", round(precision, 3))
    print("accuracy", round(accuracy, 3))
    print("f1", round(f1, 3))
    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)
    print("aucroc:", round(auc_v, 3))
    print("aupr:", round(area, 3))

    with open(os.path.join(model_name, "performance.txt"), "w") as fw:
        fw.write(
            "Evaluation metrics" + "\t" + "sensitivity" + "\t" + "specificity" + "\t" + "precision" + "\t" + "accuracy" + "\t" + "f1" + "\t" + "aucroc" + "\t" + "aupr" + "\n")
        fw.write("Value" + "\t" + str(round(sensitivity, 3)) + "\t" + str(round(specificity, 3)) + "\t" + str(
            round(precision, 3)) + "\t" + str(round(accuracy, 3)) + "\t" + str(round(f1, 3)) + "\t" + str(
            round(auc_v, 3)) + "\t" + str(round(area, 3)) + "\n")
        df_evaluate = pd.DataFrame({"dataset":[dti],"classifier": [model_name], "accuracy": [
            str(accuracy)], "specificity": [
            str(specificity)],
                                    "sensitivity": [str(sensitivity)], "aucroc": [
                str(auc_v)], "aupr": [
                str(area)],
                                    "f1": [str(f1)]})
        df_evaluate.to_csv("./result/evaluate-performance.csv", mode="a", index=None)
    print("=============Evaluate Over! ===========", flush=True)
    return sensitivity, specificity, precision, accuracy, f1, auc_v, area


def History(history, name):
    epochs = range(len(history.history['accuracy']))
    plt.figure(dpi=300)
    plt.plot(epochs, history.history['accuracy'], '#228B8B', label='Training acc')
    plt.scatter(epochs, history.history['accuracy'], color='#228B8B', s=12)
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./' + name + "/" + name + '_acc.jpg')

    plt.figure(dpi=300)
    plt.plot(epochs, history.history['loss'], '#228B8B', label='Training loss')
    plt.scatter(epochs, history.history['loss'], color='#228B8B', s=12)
    # plt.plot(epochs,history.history['val_loss'],'#8B2222',label='Validation val_loss')
    # plt.scatter(epochs,history.history['val_loss'],color='#8B2222',s=12)
    plt.title('Training and Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./' + name + "/" + name + '_loss.jpg')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dti", help="data file")
    parser.add_argument("--mirna-descripter", "-mirna", help="miRNA descripter")
    parser.add_argument("--drug-descripter", '-d', help="drug descripter")
    parser.add_argument("--model-name", "-m", help="model name")

    parser.add_argument("--learning-rate", '-r', help="Learning late for training", default=1e-4, type=float)
    parser.add_argument("--n-epoch", '-e', help="The number of epochs for training or validation", type=int,
                        default=100)
    parser.add_argument("--batch-size", "-b", help="Batch size", default=64, type=int)
    parser.add_argument("--gpu", "-g", help="Gpu number", default=0, type=str)

    parser.add_argument("--data-prefix", "-dp", help="output data prefix", default="data", type=str)

    args = parser.parse_args()

    dti = args.dti
    mirna_descripter = args.mirna_descripter
    drug_descripter = args.drug_descripter
    model_name = args.model_name
    learning_rate = args.learning_rate
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    gpu = args.gpu
    data_prefix = args.data_prefix

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    import os
    import time

    model_name_list = model_name.split("_")
    model_name_normal = model_name_list[0] + "_" + model_name_list[1] + "_"
    dti_name = dti.split('/')
    if not os.path.exists(model_name_normal + data_prefix + '_' + dti_name[1]):
        os.makedirs(model_name_normal + data_prefix + '_' + dti_name[1])
    if os.path.exists(model_name_normal + data_prefix + '_' + dti_name[1] + "/" + model_name_normal + "miRNA.txt"):
        df = open(model_name_normal + data_prefix + '_' + dti_name[1] + "/" + model_name_normal + 'miRNA.txt', 'rb')
        mirna = pickle.load(df)
        df.close()
        df = open(model_name_normal + data_prefix + '_' + dti_name[1] + "/" + model_name_normal + 'drug.txt', 'rb')
        drug = pickle.load(df)
        df.close()
        df = open(model_name_normal + data_prefix + '_' + dti_name[1] + "/" + model_name_normal + 'y.txt', 'rb')
        y = pickle.load(df)
        df.close()
    else:
        mirna, drug, y = newdata(dti, mirna_descripter, drug_descripter)
        fw = open(model_name_normal + data_prefix + '_' + dti_name[1] + "/" + model_name_normal + "miRNA.txt", 'wb')
        pickle.dump(mirna, fw)
        fw.close()
        fw = open(model_name_normal + data_prefix + '_' + dti_name[1] + "/" + model_name_normal + "drug.txt", 'wb')
        pickle.dump(drug, fw)
        fw.close()
        fw = open(model_name_normal + data_prefix + '_' + dti_name[1] + "/" + model_name_normal + "y.txt", 'wb')
        pickle.dump(y, fw)
        fw.close()

    train_m, test_m, train_d, test_d, train_y, test_y = split(mirna, drug, y, drug_descripter, model_name, dti)
    # print(test_y)
    history1, model = fitting(train_m, test_m, train_d, test_d, train_y, test_y, model_name, learning_rate, n_epoch, ".", model_name,dti, batchsize=batch_size)
    evaluate("./" + model_name +"_"+ dti_name[1] + "/" + model_name  +".ckpt", test_m, test_d, test_y, model_name)
    History(history1, model_name)
