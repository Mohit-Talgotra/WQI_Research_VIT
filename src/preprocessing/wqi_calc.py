import pandas as pd

standards = {
    "Turbidity": {"Si": 5, "Vi": 0},
    "pH": {"Si": 8.5, "Vi": 7},
    "TDS": {"Si": 500, "Vi": 0},
    "Total Alkalinity": {"Si": 200, "Vi": 0},
    "Chloride": {"Si": 250, "Vi": 0},
    "Fluoride": {"Si": 1.5, "Vi": 1},
    "Nitrate": {"Si": 45, "Vi": 0},
    "Sulphate": {"Si": 200, "Vi": 0},
    "Total Hardness": {"Si": 200, "Vi": 0},
    "Iron": {"Si": 0.3, "Vi": 0},
    "Free Residual Chlorine": {"Si": 1.0, "Vi": 0.5}
}

df = pd.read_csv("water_data.csv")

K = 1 / sum([1 / standards[p]["Si"] for p in standards])
weights = {p: K / standards[p]["Si"] for p in standards}

def calc_qi(param, value):
    Si = standards[param]["Si"]
    Vi = standards[param]["Vi"]
    try:
        qi = ((value - Vi) / (Si - Vi)) * 100
        if qi < 0: qi = 0
    except ZeroDivisionError:
        qi = 0
    return min(qi, 100)

def calc_wqi(row):
    num = 0
    den = 0
    for param in standards:
        qi = calc_qi(param, row[param])
        wi = weights[param]
        num += qi * wi
        den += wi
    return num / den if den != 0 else 0

df["WQI"] = df.apply(calc_wqi, axis=1)

df.to_csv("water_data_with_wqi.csv", index=False)
print("WQI column added and saved to 'water_data_with_wqi.csv'")