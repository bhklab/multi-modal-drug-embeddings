from collections import defaultdict
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score,roc_auc_score
import tqdm 

hdd_coldata = pd.read_csv(
    "../../data/rawdata/HDD/colData.csv",
    usecols= ["HDD.Compound.ID","GEOM.Source.SMILES"]
    )


print(hdd_coldata.head())
tox = pd.read_csv("../../data/rawdata/HDD/Tox21.csv",index_col=0)

seed = 1234


geom_coldata = pd.read_csv("../../data/rawdata/GEOM/colData.tsv", 
                           sep = "\t", 
                           usecols = ["GEOM.Source.SMILES",	"Pubchem.CID"])


# print(geom_coldata.head())
# # geom_coldata = geom_coldata.dropna(subset=['Pubchem.CID'])
# metadata = geom_coldata.merge(hdd_coldata,on='"GEOM.Source.SMILES"')
# metadata = metadata[["HDD.Compound.ID","GEOM.Source.SMILES"]]
# print(metadata.head())

whim_type_to_file = {'Average':"../../data/rawdata/GEOM/WHIM_hp.parquet",
                     "High Prob":"../../data/rawdata/GEOM/WHIMS_avg.parquet",
                     "Weight Avg.":"../../data/rawdata/GEOM/WHIMS_wavg.parquet",
                     'FP':None}


all_res = defaultdict(list)
rs = np.random.RandomState(seed)
for task in (pbar:=tqdm.tqdm(tox.index)):
    pbar.set_description(f"Working on {task}")
    skf = StratifiedKFold(n_splits=10)
    task_df = pd.DataFrame(tox.loc[task].dropna()).reset_index()
    
    n_before = task_df.shape[0]
    task_df.columns = ["HDD.Compound.ID", 'Response']
    task_df = task_df.merge(hdd_coldata,on="HDD.Compound.ID")
    for whim in whim_type_to_file.keys():
        if WHIM!='FP':
            WHIMS = pd.read_parquet(whim_type_to_file[whim])
            whim_cols = WHIMS.columns[1:]
    
        task_whims = task_df.merge(WHIMS,on='GEOM.Source.SMILES')
        task_whims['Response']=[int(x) for x in task_whims['Response']]
        n_after = task_whims.shape[0]
        X = task_whims[whim_cols].to_numpy(copy=True)
        y = task_whims['Response']
        scaler = StandardScaler()
        res = defaultdict(list)
        # print(task_df.shape)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index,:],X[test_index,:]
            y_train, y_test = y[train_index],y[test_index]
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = LogisticRegressionCV(max_iter=100000)
            model.fit(X_train,y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test,preds)
            res['Accuracy'].append(acc)
            acc = balanced_accuracy_score(y_test,preds)
            res['Balanced Accuracy'].append(acc)
            probs = model.predict_proba(X_test)[:,1]
            roc = roc_auc_score(y_test,probs)
            res['ROC AUC'].append(roc)

        all_res['Task'].append(task)
        for k in res.keys():
            all_res[k].append(np.mean(res[k]))
        all_res['n_before'].append(n_before)
        all_res['n_after'].append(n_after)
        all_res['WHIM Type'].append(whim)

    
    
    
    
all_res = pd.DataFrame(all_res)
all_res.to_csv("../../data/results/tox21_LR.csv")
    
