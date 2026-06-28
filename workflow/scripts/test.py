import pandas as pd
from damply import dirs

# df = pd.read_csv(dirs.RAWDATA / 'GEOM'/ 'colData.tsv', sep="\t")
# print(df.shape)
# print(len(pd.unique(df["GEOM.Source.SMILES"])))
# print(df['Pubchem.CID'].value_counts())
# print(len(pd.unique(df['Pubchem.CID'])))
# df = df.dropna(subset=['Pubchem.CID'])
# print(df.shape)
# print(df['Pubchem.CID'].value_counts())
#CDK4, CDK6

df = pd.read_csv(dirs.RAWDATA / 'HDD'/ 'colData.csv')
moas = pd.DataFrame(df['Mechanism.of.Action'].value_counts()).reset_index()
moas.to_csv(dirs.RESULTS / "moas.csv")
