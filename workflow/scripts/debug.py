import pandas as pd
from mvfusion.mv_integrator import MultiViewIntegrator
from scipy import stats
from itertools import combinations
from snf import compute
from snf import datasets
import sys
simdata = datasets.load_simdata()

# print(simdata)
print(len(simdata.data))
# for j in simdata.data:
#     # print
#     print(j)
# sys.exit()
cnv = pd.read_csv("./lusc_tcga_pan_can_atlas_2018/data_cna.txt",sep="\t")
methyl = pd.read_csv("./lusc_tcga_pan_can_atlas_2018/data_methylation_hm27_hm450_merged.txt",sep="\t")
protein =pd.read_csv("./lusc_tcga_pan_can_atlas_2018/data_rppa_zscores.txt",sep="\t")




# print(methyl.iloc[:5,:4])
cnv = cnv[cnv.columns[2:]].dropna()
methyl = methyl[methyl.columns[5:]].dropna()
protein = protein[protein.columns[1:]].dropna()

common_samples = set(protein.columns).intersection(set(methyl.columns))
common_samples = common_samples.intersection(set(cnv.columns))

common_samples = sorted(list(common_samples))


view_to_metric = {'rna':'sqeuclidean', 'mut':'hamming','prot':'sqeuclidean','cnv':'sqeuclidean', 'methyl':'sqeuclidean'}

protein = protein.transpose()
cnv = cnv.transpose()
cnv = cnv.astype(float)
methyl = methyl.transpose()

protein = protein.loc[common_samples].values
cnv= cnv.loc[common_samples].values
methyl = methyl.loc[common_samples].values
# methyl = methyl.iloc[:,:500]
# cnv = cnv.iloc[:,:500]

view_names = ['prot','cnv','methyl']
view_to_data= {'prot':protein, 'cnv':cnv,'methyl':methyl}
# view_to_data= {'prot':protein, 'methyl':methyl}

# print(protein.shape)
# print(cnv.shape)
# print(methyl.shape)
affs = compute.make_affinity([view_to_data[x] for x in view_names], metric='euclidean')
for i,j in combinations(range(len(affs)),2):
    corr, _ = stats.spearmanr(affs[i].flatten(),affs[j].flatten())
    print(f"The Corr between {i} and {j} is {corr}")


fused = compute.snf(affs)
for i in range(len(affs)):
    corr, _ = stats.spearmanr(affs[i].flatten(),fused.flatten())
    print(f"The Corr between {i} and fused is {corr}")

# print(affinities)
# print(fused)
sys.exit()
integrator = MultiViewIntegrator(
            views = [view_to_data[view] for view in view_names],
            view_names=view_names,
                metrics = [view_to_metric[view] for view in view_names],
                neighborhood_size= 5,
                mu= 0.4,
                alignment_epochs=200,
                emb_dim = 100,
                seed = 30)

z = integrator.get_fused_matrices()

for i,j in combinations(z.keys(),2):
    # print(z[i])
    # print(z[i].values.flatten())
    # print(z[i].shape)
    # print(z[j].shape)
    # sys.exit()
    corr, _ = stats.spearmanr(z[i].values.flatten(),z[j].values.flatten())
    print(f"The Corr between {i} and {j} is {corr}")
    # print(j)

z = integrator.get_affinity_matrices()


print("---------")
for i,j in combinations(z.keys(),2):
    # print(z[i])
    # print(z[i].values.flatten())
    # print(z[i].shape)
    # print(z[j].shape)
    # sys.exit()
    corr, _ = stats.spearmanr(z[i].values.flatten(),z[j].values.flatten())
    print(f"The Corr between {i} and {j} is {corr}")
    # print(j)
# embeds_final, S_final, model = integrator.neural_integration()
# print(embeds_final)
# print(S_final)