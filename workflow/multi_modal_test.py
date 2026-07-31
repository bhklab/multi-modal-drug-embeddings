import pandas as pd
from collections import defaultdict
import sys
import numpy as np
from damply import dirs
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import tqdm 
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tuned_models.models import Classifier
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from unsupervised_models.auto_encoder import AutoEncoder
from rdkit import Chem
from typing import Dict


pipe = pipeline("feature-extraction", model="seyonec/ChemBERTa-zinc-base-v1")


def get_chemberta_embedding(mol:str):
    res = pipe(mol,return_tensors=True)
    
    res = res.numpy().squeeze()
    
    avg_emb = np.mean(res,axis=0).reshape(1,-1)
    return avg_emb

def create_dataset(metadata, whims):
    X = []
    metadata = metadata.reset_index()
    mol_2_idx, idx_to_mol = {},{}
    for idx, row in tqdm.tqdm(metadata.iterrows(),total=metadata.shape[0]):
        mol_2_idx[row['HDD.Compound.ID']] = idx
        idx_to_mol[idx] = row['HDD.Compound.ID']
        whim_vec = whims[row['GEOM.Source.SMILES']]
        text_emb = get_chemberta_embedding(row['SMILES'])
        vec = np.concatenate((whim_vec,text_emb),axis=1)
        X.append(list(vec.reshape(-1)))


    
    X = torch.tensor(X, dtype=torch.float32)
    return X, mol_2_idx, idx_to_mol
    
        

    
def build_whim_map(
        whim_data:pd.DataFrame
    ) ->Dict[str,np.array]:
    res = {}
    for idx, row in tqdm.tqdm(whim_data.iterrows(),total = whim_data.shape[0]):
        id = row['GEOM.Source.SMILES']
        value = np.array([row[col] for col in list(whim_data.columns[1:]) ]).reshape(1,-1)
        res[id]=value

    return res


geom_metadata = pd.read_csv(dirs.RAWDATA / "GEOM" / "colData.tsv",sep = "\t")
whim_data =  pd.read_parquet(dirs.RAWDATA / "GEOM" / "WHIM_hp.parquet")


whims =build_whim_map(whim_data)

hdd_metadata = pd.read_csv(dirs.RAWDATA / "HDD" / "colData.csv",
                           usecols=['HDD.Compound.ID','Pubchem.CID','SMILES','GEOM.Source.SMILES'])

metadata = hdd_metadata.merge(geom_metadata,on='GEOM.Source.SMILES')

# metadata = metadata.iloc[:500,:]
bs = 128

X, m2i, i2m =create_dataset(metadata,whims)
ds = TensorDataset(X)
loader = DataLoader(ds,batch_size=bs,shuffle=True)
device = "cpu"
# model = Classifier(input_dim = 768,hidden_sizes=[256,768],num_classes = y.shape[1])
num_epochs = 200
loss_fn = nn.CrossEntropyLoss()

model = AutoEncoder(in_dim =  X.shape[1],
                    latent_dim = 64, 
                    encoder_sizes = [128, 128],
                         decoder_sizes = [128, 128] )


optimizer = optim.Adam(model.parameters(), lr=0.001)               
num_empochs = 300
loss_fn = nn.MSELoss()

for ep in range(num_empochs):
    running_loss = 0.0
    for i, batch_X in enumerate(loader):
        x = batch_X[0]
        
        x_reconstruct = model(x)


        loss = loss_fn(x_reconstruct, x)
        
        running_loss+=loss.item()
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        
    epoch_loss = running_loss/len(loader.dataset)
    print(f"For epoch {ep} the loss is {epoch_loss}\n")



model.eval()
# for idex in 
        



# print(geom_metadata.head())
# print(hdd_metadata.head())
# print(whim_data.head())
# for c in whim_data.columns:
#     print(c)

# geom_info = pd.read_csv()
# def make_molecule_embedding():
#     pass



