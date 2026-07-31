import sys
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from damply import dirs
from rdkit.Chem import rdFingerprintGenerator
import heapq
import tqdm 

"""
Method 1: Look at the union of compounds where we 
have at least one measurement across possible case studies.

"""

# bioassays = pd.read_csv(dirs.RAWDATA / "HDD" / "Bioassays.csv",index_col=0)
# bioassays = bioassays.dropna(axis=1,how='all')
molData = pd.read_csv(dirs.RAWDATA / "HDD" /"colData.csv")
molData = molData.dropna(subset='SMILES')
moaData = molData.dropna(subset=['Mechanism.of.Action'])
fdaData =  molData[molData['FDA.Approved']==True]
# print(fdaData['FDA.Approved'].value_counts())

Tox21 = pd.read_csv(dirs.RAWDATA / "HDD"/ "Tox21.csv",index_col=0)
Tox21 = Tox21.dropna(axis=1,how='all')
ToxCast = pd.read_csv(dirs.RAWDATA / "HDD"/ "ToxCast.csv",index_col=0)
ToxCast = ToxCast.dropna(axis=1, how='all')
SIDER = pd.read_csv(dirs.RAWDATA / "HDD"/ "SIDER.csv",index_col=0)
ClinTox = pd.read_csv(dirs.RAWDATA / "HDD"/"ClinTox.csv")

all_cpds = set(moaData['HDD.Compound.ID'])
all_cpds = set(moaData['HDD.Compound.ID']).union(
    set(fdaData['HDD.Compound.ID']))
for df in [Tox21, ToxCast,SIDER,ClinTox]:
    all_cpds = all_cpds.union(set(df.columns))


all_cpds = sorted(list(all_cpds))
with open(dirs.RESULTS / "method_1_compounds.txt","w") as ostream:
    ostream.writelines([x+"\n" for x in all_cpds])




"""
Method 2: 

Take compounds with known MOA and find k closest molecules by Tanimoto
"""


k = 5
moaData = moaData.dropna(subset=["SMILES"])
moaMissing = molData[~(molData['HDD.Compound.ID'].isin(moaData))]


# moaMols = 

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4,
                                                       fpSize=1024,
                                                       useBondTypes=True,
                                                       includeChirality=True)


print("Making MOA Mols")
idx_2_moa_mol,moa_mols = {},  []
i = 0
for idx, row in tqdm.tqdm(moaData.iterrows(),total=moaData.shape[0]):
    mol = row['SMILES']
    _mol = Chem.MolFromSmiles(mol)
    if _mol is not None:
        idx_2_moa_mol[i]=row['HDD.Compound.ID']
        i+=1
        moa_mols.append(mfpgen.GetFingerprint(_mol))

        
idx_2_mol,mols = {},  []
i = 0
for idx, row in tqdm.tqdm(moaMissing.iterrows(),total=moaMissing.shape[0],leave=False):
    mol = row['SMILES']
    _mol = Chem.MolFromSmiles(mol)
    if _mol is not None:
        idx_2_mol[i]=row['HDD.Compound.ID']
        i+=1
        mols.append(mfpgen.GetFingerprint(_mol))        


pairwise_distance = []
k = 10
sim_mols = []
for i in tqdm.tqdm(range(len(moa_mols))):
    m1 = moa_mols[i]
    my_heap = []
    for j in tqdm.tqdm(range(len(mols)),leave=False):
        m2 = mols[j]
        sim = DataStructs.TanimotoSimilarity(m1,m2)
        if len(my_heap)<k:
            my_heap.append((idx_2_mol[j],sim))
        elif len(my_heap)==k:
            my_heap = sorted(my_heap,key = lambda item: item[1])
            if sim>my_heap[0][1]:
                my_heap[0] = (idx_2_mol[j],sim)        
    sim_mols.extend([x[0] for x in my_heap])


all_cpds = list(set(sim_mols))
print(len(all_cpds))
with open(dirs.RESULTS / "method_2_compounds.txt","w") as ostream:
    ostream.writelines([x+"\n" for x in all_cpds])