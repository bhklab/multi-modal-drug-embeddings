from damply import dirs
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from damply import dirs
import umap
from sklearn.preprocessing import StandardScaler
from MOA_clustering import CATEGORY_RULES

sclr = StandardScaler()
fitter = umap.UMAP()


def categorize_mechanism(text):
    lower = text.lower()

    for category, rules in CATEGORY_RULES.items():
        for rule in rules:
            if rule.lower() in lower:
                return category

    return "Miscellaneous"


df = pd.read_csv(dirs.RESULTS / "integration_res.csv",index_col=0)
df['MOA']=[categorize_mechanism(x) for x in df['Mechanism.of.Action']]
df = df[~(df['MOA'].isin(['Receptor Targeted','Miscellaneous',
                                         'Enzymes','Ion Channels','Ion Channels & Transporters']))]


embeddings = df[[c for c in df.columns if c not in ['HDD.Compound.ID','Mechanism.of.Action','MOA']]].values
U = fitter.fit_transform(embeddings)
U = sclr.fit_transform(U)


plot_df = pd.DataFrame({'UMAP-1':U[:,0], 
                        'UMAP-2':U[:,1],
                        'MOA':df['MOA'].values})
# plot_df = plot_df[~(plot_df['MOA'].isin(['Receptor Targeted','Miscellaneous',
#                                          'Enzymes','Ion Channels','Ion Channels & Transporters']))]
ax = sns.scatterplot(plot_df,x='UMAP-1',y='UMAP-2',hue='MOA',legend=True)
sns.move_legend(ax,"upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
# plt.scatter(U[:,0],U[:,1], c = requested_compounds['Mechanism.of.Action'].values)
plt.savefig(dirs.RESULTS / "dnf_structure.png")
plt.close()