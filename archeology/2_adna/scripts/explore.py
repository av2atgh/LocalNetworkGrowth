import sys, pandas as pd, numpy as np
D = sys.argv[1]
meta = pd.read_csv(f"{D}/ancibd_zenodo8417049/supp_tables/sheet1_Table1_PublishedIndivdiualsEura.tsv", sep="\t", skiprows=1)
meta.columns = ["iid","master","loc","lat","lon","age","age_range","region","study","clst","nsnp","cov","sex","contact","dtype","gp"]
print(meta.shape); print(meta.region.value_counts().head(30).to_string())
seg = pd.read_csv(f"{D}/ancibd_zenodo8417049/ibd220.ibd.v54.1.pub.tsv", sep="\t")
print(seg.shape, seg.lengthM.describe().to_string())
s = seg[seg.lengthM >= 0.12]
pairs = s.groupby(["iid1","iid2"]).agg(n=("lengthM","size"), tot=("lengthM","sum")).reset_index()
print("pairs with >=1 seg >=12cM:", len(pairs))
ids = set(meta.iid); print("ids in meta:", pairs.iid1.isin(ids).mean(), pairs.iid2.isin(ids).mean())
m = meta.set_index("iid")
pairs = pairs[pairs.iid1.isin(ids) & pairs.iid2.isin(ids)]
L1, L2 = m.loc[pairs.iid1,"loc"].values, m.loc[pairs.iid2,"loc"].values
same = L1 == L2; print("same locality frac:", same.mean())
lat1,lon1,lat2,lon2 = [np.radians(m.loc[pairs[c],k].astype(float).values) for c,k in [("iid1","lat"),("iid1","lon"),("iid2","lat"),("iid2","lon")]]
d = 6371*2*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))
dt = np.abs(m.loc[pairs.iid1,"age"].values - m.loc[pairs.iid2,"age"].values)
print("dist quantiles km:", np.nanpercentile(d,[50,75,90,95,99]))
print("dist quantiles km (diff loc):", np.nanpercentile(d[~same],[25,50,75,90,99]), (~same).sum())
print("dt quantiles yr:", np.nanpercentile(dt,[50,75,90,99]))
c1,c2 = m.loc[pairs.iid1,"clst"].values, m.loc[pairs.iid2,"clst"].values
print("same clst frac:", (c1==c2).mean(), " among diff loc:", (c1==c2)[~same].mean())
deg = pd.concat([pairs.iid1,pairs.iid2]).value_counts()
print("inds with >=1 link:", len(deg), "mean deg", deg.mean())
