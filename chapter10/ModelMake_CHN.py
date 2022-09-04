import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import datetime
import ModelsPlotDisplay_CHN as model_plot
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

def Norm(in_df,no_Norm):
    op_df = in_df.copy()
    for col in op_df.columns:
        if col in no_Norm:
            continue
        else:
            col_max = max(op_df[col])
            col_min = min(op_df[col])
            op_df[col] = (op_df[col] - col_min) / (col_max - col_min)
    return op_df





country = 'CHN' #########
op_path = '%s/'%country
if not os.path.exists(op_path):
    os.makedirs(op_path) 


str_txt = op_path + 'r2_msfe_op.txt'
f = open(str_txt,mode = 'w')
f.close()

whole_data = pd.read_csv('CHN_sample_data.csv')
Xtodrop = ['stkcd','Dates','y']##删掉预测中不与要的数据列
Xtodrop_add = []
Xtodrop = Xtodrop + Xtodrop_add


all_month_list = list(whole_data['Dates'].drop_duplicates().values)
train_month_n = 24
test_month_n = 12
oos_month_n = 1

nn_op = []

for i in range(len(all_month_list)):
    if i < train_month_n + test_month_n + oos_month_n - 1:
        continue
    else:
        
        train_monthes = all_month_list[i-test_month_n-train_month_n:i-test_month_n]
        test_monthes = all_month_list[i-test_month_n:i]
        oos_month = all_month_list[i]
        print(oos_month)
        
        train_data = whole_data[whole_data['Dates'].apply(lambda x: True if x in train_monthes else False)]
        test_data = whole_data[whole_data['Dates'].apply(lambda x: True if x in test_monthes else False)]
        oos_data = whole_data[whole_data['Dates'] == oos_month]

        X_train = train_data.drop(columns = Xtodrop).iloc[:,:50]########
        y_train = train_data['y']
        
        X_test = test_data.drop(columns = Xtodrop).iloc[:,:50]###########
        y_test = test_data['y']
        
        X_oos = oos_data.drop(columns = Xtodrop).iloc[:,:50]##########
        y_oos = oos_data['y']
        
        to_Norm = pd.concat([train_data,test_data,oos_data])
        normed_data = Norm(to_Norm,Xtodrop)
        train_data_normed = normed_data[normed_data['Dates'].apply(lambda x: True if x in train_monthes else False)]
        test_data_normed = normed_data[normed_data['Dates'].apply(lambda x: True if x in test_monthes else False)]
        oos_data_normed = normed_data[normed_data['Dates'] == oos_month]
        
        X_train_normed = train_data_normed.drop(columns = Xtodrop).iloc[:,:50]###
        y_train = train_data['y']
        
        X_test_normed = test_data_normed.drop(columns = Xtodrop).iloc[:,:50]####
        y_test = test_data['y']
        
        X_oos_normed = oos_data_normed.drop(columns = Xtodrop).iloc[:,:50]##########
        y_oos = oos_data['y']
        
        nn_result = model_plot.NN(X_train_normed,y_train,X_test_normed,y_test,X_oos_normed, test_data_normed, oos_data_normed)
        nn_op.append(nn_result)


# deal with NN
method = 'NN'
all_pred = []
for i in range(len(all_month_list)):
    if i < train_month_n + test_month_n + oos_month_n - 1:
        continue
    else:

        result_idx = i - (train_month_n + test_month_n + oos_month_n - 1)

        if result_idx >= len(nn_op):
            break

        oos_month = all_month_list[i]
        temp_tuning = nn_op[result_idx]
        all_pred.append(temp_tuning)

all_pred = pd.concat(all_pred)
all_pred['Dates'] = [datetime.datetime(year = int(x//100),month = int(x%100),day = 28) for x in all_pred['Dates']]
all_pred = all_pred.sort_values(['layers','Dates'])
all_pred = all_pred.set_index(['layers'])



for l in range(1,4):

    alpha_pred = all_pred.loc[l]
    pred_toplot = alpha_pred.copy()

    msfe_monthly = alpha_pred.groupby('Dates').apply(lambda x: np.sum((x['yhat']-x['y'])**2) / len(x) )
    head_mse = msfe_monthly.sort_values().head(2)
    head_month = list(head_mse.index)
    for m in head_month:
        print(m)
        mm = str(m)[:7].replace('-','')
        m_pred = pred_toplot[pred_toplot['Dates'] == m]
        m_pred = m_pred.sort_values('yhat')
        m_pred['y'].reset_index(drop=True).plot()
        m_pred['yhat'].reset_index(drop=True).plot(title='Pred vs Real %s'%mm)
        plt.legend()
        plt.savefig(op_path + 'pred_vs_real_l%s_%s.png'%(l,mm))
        plt.clf()

        abs(m_pred['y'] - m_pred['yhat']).reset_index(drop=True).plot(title='abs error %s'%mm)
        plt.savefig(op_path + 'abs_error_l%s_%s.png' % (l, mm))
        plt.clf()

    r2 = 1-np.sum((alpha_pred['yhat']-alpha_pred['y'])**2)/np.sum(alpha_pred['y']**2)
    r2_str = '\n %s R2: (layer = %s)'%(method,l)+ str(r2)
    f = open(str_txt, mode = 'a')
    f.write(r2_str)
    f.close()

    msfe = np.sum((alpha_pred['yhat']-alpha_pred['y'])**2) / len(alpha_pred)
    msfe_str = '\n %s msfe: (layer = %s)' % (method, l) + str(msfe)
    f = open(str_txt, mode='a')
    f.write(msfe_str)
    f.close()
    
