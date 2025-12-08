import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

standards = {
    "pH (NA)": {"Sn": 8.5, "Videal": 7},
    "TDS (mg/l)": {"Sn": 500, "Videal": 0},
    "Total Hardness (As CaCO3) (mg/l)": {"Sn": 200, "Videal": 0},
    "Chloride (as Cl) (mg/l)": {"Sn": 250, "Videal": 0},
    "Fluoride (as F) (mg/l)": {"Sn": 1.0, "Videal": 0},
    "Total Alkalinity (as Calcium Carbonate) (mg/l)": {"Sn": 200, "Videal": 0},
    "Sulphate (as SO4) (mg/l)": {"Sn": 200, "Videal": 0},
    "Nitrate (as NO3) (mg/l)": {"Sn": 45, "Videal": 0}
}

df = pd.read_csv(os.environ["CLEANED_DATA_FILE_PATH"])

K = 1 / sum([1 / standards[p]["Sn"] for p in standards])

weights = {p: K / standards[p]["Sn"] for p in standards}

def calc_qn(param, Vn):
    Sn = standards[param]["Sn"]
    Videal = standards[param]["Videal"]
    return ((Vn - Videal) / (Sn - Videal)) * 100

def calc_wqi(row):
    wqi_sum = 0
    for param in standards:
        Vn = row[param]
        qn = calc_qn(param, Vn)
        wn = weights[param]
        wqi_sum += qn * wn
    return wqi_sum

df["WQI"] = df.apply(calc_wqi, axis=1)

df.to_csv(os.environ["WQI_CALCULATED_DATA_FILE_PATH"], index=False)
print("WQI calculated EXACTLY like Excel and saved.")