from textwrap import wrap
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

    
    keep_moas = pd.DataFrame(emb_df['Mechanism.of.Action'].value_counts()).reset_index()
    keep_moas = keep_moas['Mechanism.of.Action'].values[:5]
    # print(keep_moas)
    # sys.exit()
    embeddings = emb_df[[c for c in emb_df.columns if c not in ['HDD.Compound.ID','Mechanism.of.Action','MOA']]].values
    U = fitter.fit_transform(embeddings)
    # U = sclr.fit_transform(U)


    plot_df = pd.DataFrame({'UMAP-1':U[:,0], 
                            'UMAP-2':U[:,1],
                            'MOA':emb_df['Mechanism.of.Action'].values})
# plot_df = plot_df[~(plot_df['MOA'].isin(['Receptor Targeted','Miscellaneous',
    #                                          'Enzymes','Ion Channels','Ion Channels & Transporters']))]
    plot_df = plot_df[plot_df['MOA'].isin(keep_moas)]
    ax = sns.scatterplot(plot_df,x='UMAP-1',y='UMAP-2',hue='MOA',legend=True)
    handles, labels = ax.get_legend_handles_labels()
    labels = [ '\n'.join(wrap(l, 20)) for l in labels]
    plt.legend(labels = labels)
    # print(labels)
    # ax.legend()
    
    sns.move_legend(ax,"upper left", bbox_to_anchor=(1, 1))
    
    
    # plt.scatter(U[:,0],U[:,1], c = requested_compounds['Mechanism.of.Action'].values)
    plt.title(f"{exp} UMAP")
    plt.tight_layout()
    plt.savefig(dirs.RESULTS / exp /  f"{exp}_umap.png")
    plt.close()
    
    adjacency = pd.read_csv(dirs.RESULTS / exp / "Adjacency.csv",index_col=0)
    
    # G = nx.from_numpy_array(adjacency.values)
    # nx.draw(G)
    # plt.savefig(dirs.RESULTS / exp / "network.png")
    # plt.close()
    

