""""
Construct embeddings of drug structure only based on neural multi-omic fusion. 

"""
import yaml
import argparse 
import pandas as pd
from mvfusion.mv_integrator import MultiViewIntegrator
from scipy import stats
from itertools import combinations
# from snf import compute
# from snf import datasets
import sys
from damply import dirs
from drug_embedders import Small_Molecule_Embedder
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from damply import dirs
from rdkit.Chem import rdFingerprintGenerator
import tqdm 
import numpy as np
from typing import Dict
import torch
from utils import unpack_parameters
import os



def main(config:Dict):
 
    outdir = dirs.RESULTS / "Response"

    lincs = pd.read_parquet(dirs.RAWDATA / "LINCS"/ "signatures.parquet")


    

    
    
   
    os.makedirs(outdir,exist_ok=True)

    fpRadius, useBondTypes, includeChirality, fpSize = unpack_parameters(config['MORGAN_PARAMETERS'])

    alignment_epochs, mu, nbhd_size, seed,emb_dim = unpack_parameters(config['INTEGRATION_PARAMETERS'])

    
    

    molData = pd.read_csv(dirs.RAWDATA /"HDD"/'colData.csv',
                      usecols=['HDD.Compound.ID','Mechanism.of.Action','SMILES','GEOM.Source.SMILES','LINCS.CMap.Name'])
    molData = molData.dropna(subset= ['Mechanism.of.Action','SMILES'])
    molData = molData.reset_index(drop=True)

    lincs = pd.merge(molData[['HDD.Compound.ID','LINCS.CMap.Name']], lincs,on='LINCS.CMap.Name')
    
    lincs = lincs.drop(columns=['LINCS.CMap.Name']).set_index('HDD.Compound.ID')
    X = lincs.values
    X = X/np.linalg.norm(X,axis=1,keepdims=True)
    lincs = pd.DataFrame(X, index=lincs.index, columns=  lincs.columns)
    # lincs = lincs.div(np.sqrt(np.square(lincs).sum(axis=1)))
    
    

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=fpRadius,
                                                       fpSize=fpSize,
                                                       useBondTypes=useBondTypes,
                                                       includeChirality=includeChirality)


    geom = pd.read_parquet(dirs.RAWDATA / "GEOM"/ "WHIM_hp.parquet")
    feat_cols = [c for c in list(geom.columns) if c[:4]=='WHIM']

    geom = geom.merge(molData,on='GEOM.Source.SMILES')
    geom = geom[['HDD.Compound.ID']+feat_cols]
    geom = geom.set_index('HDD.Compound.ID')

    fps = []
    chemberta = []
    DrugEmb = Small_Molecule_Embedder()
    for idx, row in tqdm.tqdm(molData.iterrows(),total = molData.shape[0]):
        mol = row["SMILES"]
        smiles_embedding = DrugEmb.embed_molecule(mol)
        chemberta.append(smiles_embedding)
        mol = Chem.MolFromSmiles(mol)
        fp = np.array(mfpgen.GetFingerprint(mol))
        fp = fp.astype(np.float32)
        fps.append(fp)


    fingerprints = pd.DataFrame(fps, index=molData['HDD.Compound.ID'])

    chemberta_embeddings = pd.DataFrame(chemberta, index=molData['HDD.Compound.ID'])


    integrator = MultiViewIntegrator(
                views = [fingerprints, chemberta_embeddings, geom, lincs],
                view_names=['fingerprints','chemberta','geom','lincs'],
                    metrics = ['sqeuclidean']*4,
                    neighborhood_size= nbhd_size,
                    mu= mu,
                    alignment_epochs=alignment_epochs,
                    emb_dim = emb_dim,
                    seed = seed)



    embeds_final, S_final, model = integrator.neural_integration()
    embeds_final = embeds_final.reset_index(names='HDD.Compound.ID')
    embeds_final = embeds_final.merge(
        molData[['HDD.Compound.ID','Mechanism.of.Action']],
        on = 'HDD.Compound.ID'    )

    embeds_final.to_csv(outdir / "Embeddings.csv")


    all_samples = embeds_final.index

    final_adjacency = pd.DataFrame(S_final)

    final_adjacency.to_csv(outdir / "Adjacency.csv")


if __name__ =="__main__":
    parser = argparse.ArgumentParser(
		prog = "Configuration for Structure Embeddings.",
		description = "")

    parser.add_argument("-config", help = "The configuration file for the experiment.")
    args = parser.parse_args()
    with open(dirs.CONFIG / args.config) as config_file:
        config = yaml.safe_load(config_file)

    main(config)

    
    
