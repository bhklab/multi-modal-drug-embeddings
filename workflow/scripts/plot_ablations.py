import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from damply import dirs




ablation = pd.read_csv(dirs.RESULTS / "ablation.csv",index_col=0)
sns.barplot(ablation, x="AUC", y="Modalities", hue="Model")
plt.tight_layout()
plt.savefig(dirs.RESULTS / "modality_ablation.png")
plt.close()
neighbors = pd.read_csv(dirs.RESULTS / "auc.csv",index_col=0)
neighbors = neighbors[neighbors['Model'].isin(['SNF','IntegrAO'])]
sns.lineplot(data=neighbors, x="K", y="AUC",hue="Model")
plt.xlabel("Num. Neighbours")
plt.savefig(dirs.RESULTS / "nbr_ablation.png")
plt.close()