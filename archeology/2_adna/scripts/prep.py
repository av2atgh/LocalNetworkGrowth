"""Build per-region individual tables and IBD edge lists from ancIBD Zenodo 8417049.
usage: python -I prep.py <data_dir> <out_dir>
Culture labels: keyword map over AADR v54.1 Group IDs (the 'Cluster Label' of ancIBD Supp. Table 1).
Genetic qualifiers (_o*, lowEEF/highEEF, kin tags, possible.contam) are stripped; site names are not
used as culture. Individuals without a recognised culture keyword get a generic country_period label
and is_culture=False."""
import sys, re, numpy as np, pandas as pd
D, O = sys.argv[1], sys.argv[2]
meta = pd.read_csv(f"{D}/ancibd_zenodo8417049/supp_tables/sheet1_Table1_PublishedIndivdiualsEura.tsv", sep="\t", skiprows=1)
meta.columns = ["iid","master","loc","lat","lon","age","age_range","region","study","clst","nsnp","cov","sex","contact","dtype","gp"]
meta = meta.dropna(subset=["iid","lat","lon","age"]).copy()

CULT = [  # (regex, culture) first match wins; order matters
 (r"Starcevo|Koros", "Starcevo-Koros"), (r"LBK", "LBK"), (r"ALPc|AVK|Szatmar|Tiszadob|Bukk", "ALP"),
 (r"Sopot", "Sopot"), (r"Lengyel", "Lengyel"), (r"BKG", "BKG"), (r"Tiszapolgar|Bodrogker", "Bodrogkeresztur"),
 (r"Tisza", "Tisza"), (r"Lasinja", "Lasinja"), (r"Baalberge", "Baalberge"), (r"TRB", "TRB"),
 (r"Jordanow", "Jordanow"), (r"Rivnac", "Rivnac"), (r"Protoboleraz|Baden", "Baden"),
 (r"GlobularAmphora", "GlobularAmphora"), (r"Yamnaya", "Yamnaya"), (r"CordedWare", "CordedWare"),
 (r"ChopiceVesele", "Chopice-Vesele"), (r"BellBeaker", "BellBeaker"), (r"Unetice|Starounetice", "Unetice"),
 (r"Maros", "Maros"), (r"Mako", "Mako"), (r"Singen", "Singen"), (r"Veterov", "Veterov"), (r"Vatya", "Vatya"),
 (r"Fuzesabony", "Fuzesabony"), (r"Tumulus", "Tumulus"), (r"Knoviz", "Knoviz"), (r"Kyjatice", "Kyjatice"),
 (r"Halva", "Halva"), (r"Mezocsat", "Mezocsat"), (r"Vekerzug", "Vekerzug"), (r"Scythian", "Scythian"),
 (r"LaTene", "LaTene"), (r"Hallstatt", "Hallstatt"), (r"SremGroup|Syrmian", "Srem"),
 (r"Megalithic", "Megalithic"),
]
def clean(c):
    c = re.sub(r"\.(SG|DG|AG|TW)\b", "", c)
    c = re.split(r"_(?:father|mother|son|daughter|brother|sister|halfbro|1d|possible|dauhter)", c)[0]
    c = re.sub(r"_(o\w*|low\w*EEF|high\w*EEF|medium\w*EEF|noUDG)$", "", c)
    return c
def culture(c):
    for rx, name in CULT:
        if re.search(rx, c): return name, True
    return None, False
meta["label"] = meta.clst.astype(str).map(clean)
cc = meta.label.map(culture)
meta["culture"] = [x[0] for x in cc]; meta["is_culture"] = [x[1] for x in cc]
def period(a):  # crude Central-European period from date (BP)
    return ("N" if a > 5500 else "C" if a > 4500 else "EBA" if a > 3600 else "MBA" if a > 3300 else "LBA" if a > 2800 else "IA")
meta["period"] = meta.age.map(period)
meta.loc[~meta.is_culture, "culture"] = meta.region.str.replace(" ", "") + "_" + meta.period + "_generic"
meta["tbin"] = (meta.age // 500 * 500).astype(int)  # lower edge of 500-yr bin in BP

seg = pd.read_csv(f"{D}/ancibd_zenodo8417049/ibd220.ibd.v54.1.pub.tsv", sep="\t")
seg = seg[seg.lengthM >= 0.12]
pairs = seg.groupby(["iid1","iid2"]).agg(n12=("lengthM","size"), sum12=("lengthM","sum"), max12=("lengthM","max")).reset_index()
pairs[["sum12","max12"]] *= 100  # cM

REG = {"CE": (45,55,8,24,2000), "BI": (49,61,-11,2.5,2000), "IB": (35,44,-10,4,2000)}
for name,(la0,la1,lo0,lo1,amin) in REG.items():
    s = meta[(meta.lat>=la0)&(meta.lat<=la1)&(meta.lon>=lo0)&(meta.lon<=lo1)&(meta.age>amin)]
    ids = set(s.iid)
    e = pairs[pairs.iid1.isin(ids) & pairs.iid2.isin(ids)]
    s.to_csv(f"{O}/{name}_individuals.tsv", sep="\t", index=False)
    e.to_csv(f"{O}/{name}_ibd12_pairs.tsv", sep="\t", index=False)
    print(name, len(s), "inds;", s["loc"].nunique(), "localities;", len(e), "pairs >=12cM;",
          "named-culture frac %.2f" % s.is_culture.mean(), "; cultures:", s[s.is_culture].culture.nunique())
    if name == "CE": print(s[s.is_culture].culture.value_counts().to_string())
