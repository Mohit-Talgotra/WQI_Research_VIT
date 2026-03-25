import pandas as pd

xls = pd.ExcelFile("x", engine="openpyxl")
print("Sheets:", xls.sheet_names)

for i, sheet in enumerate(xls.sheet_names):
    df = pd.read_excel(xls, sheet_name=i)
    print(f"\nSheet {i} — '{sheet}' — shape: {df.shape}")
    print(df.head(5))