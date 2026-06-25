from damply import dirs
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from damply import dirs
import umap
from sklearn.preprocessing import StandardScaler
from moa_labels import CATEGORY_RULES
import networkx as nx 

sclr = StandardScaler()
fitter = umap.UMAP(random_state=42)


def categorize_mechanism(text):
    lower = text.lower()

    for category, rules in CATEGORY_RULES.items():
        for rule in rules:
            if rule.lower() in lower:
                return category

    return "Miscellaneous"



for exp in ['Structure','Response']:
    emb_df = pd.read_csv(dirs.RESULTS / exp / "Embeddings.csv",index_col=0)
    emb_df['MOA']=[categorize_mechanism(x) for x in emb_df['Mechanism.of.Action']]
    emb_df = emb_df[~(emb_df['MOA'].isin(['Receptor Targeted','Miscellaneous',
                                         'Enzymes','Ion Channels','Ion Channels & Transporters']))]


    embeddings = emb_df[[c for c in emb_df.columns if c not in ['HDD.Compound.ID','Mechanism.of.Action','MOA']]].values
    U = fitter.fit_transform(embeddings)
# U = sclr.fit_transform(U)


    plot_df = pd.DataFrame({'UMAP-1':U[:,0], 
                            'UMAP-2':U[:,1],
                            'MOA':emb_df['MOA'].values})
# plot_df = plot_df[~(plot_df['MOA'].isin(['Receptor Targeted','Miscellaneous',
    #                                          'Enzymes','Ion Channels','Ion Channels & Transporters']))]
    ax = sns.scatterplot(plot_df,x='UMAP-1',y='UMAP-2',hue='MOA',legend=True)
    sns.move_legend(ax,"upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    # plt.scatter(U[:,0],U[:,1], c = requested_compounds['Mechanism.of.Action'].values)
    plt.savefig(dirs.RESULTS / exp /  f"{exp}_umap.png")
    plt.close()

    adjacency = pd.read_csv(dirs.RESULTS / exp / "Adjacency.csv",index_col=0)
    
    G = nx.from_numpy_array(adjacency.values)
    nx.draw(G)
    plt.savefig(dirs.RESULTS / exp / "network.png")
    plt.close()
    

