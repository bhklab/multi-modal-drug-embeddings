from textwrap import wrap
from damply import dirs
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os
from damply import dirs
import umap
from sklearn.preprocessing import StandardScaler
# from moa_labels import CATEGORY_RULES
from utils import KEEP_MOAS, CATEGORY_RULES, MOA_RENAME
import networkx as nx 
sclr = StandardScaler()
fitter = umap.UMAP(random_state=42)
from collections import defaultdict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
# def categorize_mechanism(text):
#     lower = text.lower()

#     for category, rules in CATEGORY_RULES.items():
#         for rule in rules:
#             if rule.lower() in lower:
#                 return category

#     return "Miscellaneous"

sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
classifier_results = defaultdict(list)

all_experiments = os.listdir(dirs.RESULTS)

NUM_FOLDS = 10

seed = 1234

rng = np.random.default_rng(seed)

for exp in all_experiments:
    if not os.path.isdir(dirs.RESULTS / exp):
        continue
    
    
    emb_df = pd.read_csv(dirs.RESULTS / exp / "Embeddings.csv",index_col=0)
    # emb_df['MOA']=[categorize_mechanism(x) for x in emb_df['Mechanism.of.Action']]
    # emb_df = emb_df[~(emb_df['MOA'].isin(['Receptor Targeted','Miscellaneous',
    #                                      'Enzymes','Ion Channels','Ion Channels & Transporters']))]

    
    # keep_moas = pd.DataFrame(emb_df['Mechanism.of.Action'].value_counts()).reset_index()
    # keep_moas = keep_moas['Mechanism.of.Action'].values[:5]
    # print(keep_moas)
    # sys.exit()

    
    embeddings = emb_df[[c for c in emb_df.columns if c not in ['HDD.Compound.ID','Mechanism.of.Action','MOA']]].values
    U = fitter.fit_transform(embeddings)
    # U = sclr.fit_transform(U)


    plot_df = pd.DataFrame({'UMAP-1':U[:,0], 
                            'UMAP-2':U[:,1],
                            'MOA':emb_df['Mechanism.of.Action'].values})
    # 
    plot_df = plot_df[plot_df['MOA'].isin(KEEP_MOAS)]
    plot_df['MOA'] = [MOA_RENAME[x] if x in MOA_RENAME.keys() else x for x in plot_df['MOA']]
    plot_df['MOA'] = [ '\n'.join(wrap(l, 20)) for l in plot_df['MOA']]
    
    ax = sns.scatterplot(plot_df,x='UMAP-1',y='UMAP-2',hue='MOA',legend=True)
    
  
    
    sns.move_legend(ax,"upper left", bbox_to_anchor=(1, 1))
    
    
    
    plt.title(f"{exp} UMAP")
    plt.tight_layout()
    plt.savefig(dirs.RESULTS / exp /  f"{exp}_umap.png")
    plt.close()
    
  
    

