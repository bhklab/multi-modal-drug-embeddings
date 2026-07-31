from collections import defaultdict
from integrao.dataset import GraphDataset
from integrao.main import dist2
from integrao.integrater import integrao_integrater, integrao_predictor


import pandas as pd
from mvfusion.mv_integrator import MultiViewIntegrator
from scipy import stats
from itertools import combinations
from snf import compute
from snf import datasets
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import tqdm

neighbor_size = 20
embedding_dims = 64
fusing_iteration = 20
normalization_factor = 1.0
alighment_epochs = 100
beta = 1.0
mu = 0.3



cnv = pd.read_csv("./brca_tcga_pan_can_atlas_2018/data_cna.txt",sep="\t")
methyl = pd.read_csv("./brca_tcga_pan_can_atlas_2018/data_methylation_hm27_hm450_merged.txt",sep="\t")
protein =pd.read_csv("./brca_tcga_pan_can_atlas_2018/data_rppa_zscores.txt",sep="\t")


cnv = cnv[cnv.columns[2:]].dropna()
methyl = methyl[methyl.columns[5:]].dropna()
protein = protein[protein.columns[1:]].dropna()


common_samples = set(protein.columns).intersection(set(methyl.columns))
common_samples = common_samples.intersection(set(cnv.columns))
common_samples = sorted(list(common_samples))
all_samples = set(protein.columns).union(set(methyl.columns))
all_samples = all_samples.union(set(cnv.columns))
all_samples = sorted(list(all_samples))


view_to_metric = {'rna':'sqeuclidean', 'mut':'hamming','prot':'sqeuclidean','cnv':'sqeuclidean', 'methyl':'sqeuclidean'}

protein = protein.transpose()
cnv = cnv.transpose()
cnv = cnv.astype(float)
methyl = methyl.transpose()

# protein = protein.loc[common_samples]
# cnv= cnv.loc[common_samples]
# methyl = methyl.loc[common_samples]






for ae in [1,10,100,200,300,400,500, 10000]:
    print("\n\n----------------\n\n")
    print(ae)
    integrater = integrao_integrater(
        [cnv, methyl, protein],
        "lung",
        modalities_name_list=["cnv", "methyl", "protein"],   # used for naming the incomplete modalities during new sample inference
        neighbor_size=neighbor_size,
        embedding_dims=embedding_dims,
        fusing_iteration=fusing_iteration,
        normalization_factor=normalization_factor,
        alighment_epochs=ae,
        mu=mu
    )

    integrater.network_diffusion()

    affs = integrater.affinity_matrices

    lifted_affs = []
    print("lifting")
    for aff in tqdm.tqdm(affs,total=len(affs)):
        lifted_mat = pd.DataFrame(
            np.nan*np.zeros((len(all_samples),len(all_samples))),
            index = all_samples,
            columns = all_samples)
        for i in aff.index:
            for j in aff.index:
                lifted_mat.loc[i,j]=aff.loc[i,j]
                # lifted_mat.loc[j,i]=aff.loc[i,j]
        lifted_affs.append(lifted_mat)
        
        


    embeds_final, S_final, model = integrater.unsupervised_alignment()
    # print(embeds_final.head())
    # print(embeds_final.shape)
    all_samples = embeds_final.index
    
    # print(affs[0])
    
    fus = integrater.fused_networks
    print("lifting")
    lifted_fus =[]
    for f in tqdm.tqdm(fus,total=len(fus)):
        lifted_mat = pd.DataFrame(
            np.nan*np.zeros((len(all_samples),len(all_samples))),
            index = all_samples,
            columns = all_samples)
        for i in f.index:
            for j in f.columns:
                lifted_mat.loc[i,j]=f.loc[i,j]
        lifted_fus.append(lifted_mat)


    vns = ["cnv", "methyl", "protein"]

    res = defaultdict(list)
    print("about to do spearmans")
    for i,j in combinations(range(len(lifted_affs)),2):
        corr, _ = stats.spearmanrho(lifted_affs[i].values.flatten(),lifted_affs[j].values.flatten(),nan_policy='omit')
    
        res['Comparison'].append(f"{vns[i]} vs. {vns[j]}")
        res['Spearman Corr'].append(corr)
        res['Matrix Type'].append("Affinity")




    for i,j in combinations(range(len(lifted_fus)),2):
        corr, _ = stats.spearmanrho(lifted_fus[i].values.flatten(),lifted_fus[j].values.flatten(),nan_policy='omit')
        res['Comparison'].append(f"{vns[i]} vs. {vns[j]}")
        res['Spearman Corr'].append(corr)
        res['Matrix Type'].append("Fused Affinity")


    

    print("about to do spearmans")
    for i in range(len(lifted_affs)):
        corr, _ = stats.spearmanrho(lifted_affs[i].values.flatten(),S_final.flatten(),nan_policy='omit')
    
        res['Comparison'].append(f"{vns[i]} vs. Fused")
        res['Spearman Corr'].append(corr)
        res['Matrix Type'].append("Affinity")

    for i in range(len(lifted_fus)):
        corr, _ = stats.spearmanrho(lifted_fus[i].values.flatten(),S_final.flatten(),nan_policy='omit')
    
        res['Comparison'].append(f"{vns[i]} vs. Fused")
        res['Spearman Corr'].append(corr)
        res['Matrix Type'].append("Fused Affinity")
    sns.barplot(res,x='Comparison',y='Spearman Corr',hue = 'Matrix Type')
    plt.xticks(rotation=30)
    plt.title(f"BRCA Comparison of Network Edge Correlations w/ {ae} epochs")
    plt.tight_layout()
    # plt.show()
    # sys.exit()
    plt.savefig(f"breast_comparison_{ae}_all_samples.png")
    plt.close()