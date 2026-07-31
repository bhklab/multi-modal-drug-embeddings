""""
Construct embeddings of drug structure only based on neural multi-omic fusion. 

"""

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
from ot.gromov import gromov_barycenters

molData = pd.read_csv(dirs.RAWDATA /"HDD"/'colData.csv',
                      usecols=['HDD.Compound.ID','Mechanism.of.Action','SMILES','GEOM.Source.SMILES'])
molData = molData.dropna(subset= ['Mechanism.of.Action','SMILES'])
molData = molData.reset_index(drop=True)
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4,
                                                       fpSize=1024,
                                                       useBondTypes=True,
                                                       includeChirality=True)


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


fps = pd.DataFrame(fps, index=molData['HDD.Compound.ID'])

cbs = pd.DataFrame(chemberta, index=molData['HDD.Compound.ID'])





integrator = MultiViewIntegrator(
            views = [fps, cbs, geom],
            view_names=['fp','cb','geom'],
                metrics = ['sqeuclidean']*3,
                neighborhood_size= 20,
                mu= 0.4,
                alignment_epochs=10,
                emb_dim = 100,
                seed = 30)



embeds_final, S_final, model = integrator.neural_integration()
embeds_final = embeds_final.reset_index(names='HDD.Compound.ID')
embeds_final = embeds_final.merge(
    molData[['HDD.Compound.ID','Mechanism.of.Action']],
    on = 'HDD.Compound.ID'    )

embeds_final.to_csv(dirs.RESULTS / "integration_res.csv")


all_samples = embeds_final.index
