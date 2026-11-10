import sys, pandas as pd
for p in sys.argv[1:]:
    x = pd.read_excel(p, sheet_name=None)
    for k, df in x.items():
        print('==', p, '| sheet', k, df.shape)
        print(df.head(4).to_string()[:1500])
        print(list(df.columns)[:60])
