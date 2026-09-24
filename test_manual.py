import sys
sys.path.insert(0, "src")

from si.io.csv_file import read_csv
from si.feature_selection.variance_threshold import VarianceThreshold
from si.feature_selection.select_k_best import SelectKBest

# ajusta o caminho conforme onde tens o iris.csv em "datasets"
ds = read_csv("datasets/iris/iris.csv", sep=",", features=True, label=True)
print("shape original:", ds.shape)

vt = VarianceThreshold(threshold=0.5)
ds_vt = vt.fit_transform(ds)
print("Depois de VarianceThreshold:", ds_vt.shape, ds_vt.features)

skb = SelectKBest(k=2)
ds_skb = skb.fit_transform(ds)
print("Depois de SelectKBest (k=2):", ds_skb.shape, ds_skb.features)