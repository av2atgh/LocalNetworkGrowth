"""Convert every sheet of an xlsx to TSV: xlsx2tsv.py in.xlsx outdir"""
import sys, os, csv, openpyxl
wb = openpyxl.load_workbook(sys.argv[1], read_only=True, data_only=True)
os.makedirs(sys.argv[2], exist_ok=True)
for i, ws in enumerate(wb.worksheets):
    fn = os.path.join(sys.argv[2], f"sheet{i+1}_" + "".join(c if c.isalnum() else "_" for c in ws.title)[:40] + ".tsv")
    with open(fn, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        n = 0
        for row in ws.iter_rows(values_only=True):
            w.writerow(["" if v is None else v for v in row]); n += 1
    print(fn, n)
