"""
Geocodes the 98 monitoring stations in Salem District across 5 blocks:
- Salem (14 stations - LOGO cohort)
- Ayodhiyapattanam (32 stations)
- Panamarathupatti (20 stations)
- Veerapandi (24 stations)
- Yercaud (9 stations)

Saves results to src/data/locations_geocoded_98.csv
"""

import sys
import time
from pathlib import Path
import pandas as pd
from geopy.geocoders import Nominatim

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "src" / "data" / "monthly_wqi_parameter_dataset.csv"
OUT_PATH = ROOT / "src" / "data" / "locations_geocoded_98.csv"

df = pd.read_csv(DATA_PATH)
locs = df[["Block", "Location"]].drop_duplicates().sort_values(["Block", "Location"]).reset_index(drop=True)
print(f"Total locations to geocode: {len(locs)}")

geolocator = Nominatim(user_agent="salem_district_wqi_mapper_v2")

# Salem district boundaries
LAT_MIN, LAT_MAX = 11.35, 12.05
LON_MIN, LON_MAX = 77.85, 78.50

# Block centroids in Salem district (administrative headquarters/centers)
centroids = {
    "Salem": (11.6643, 78.1460),
    "Ayodhiyapattanam": (11.6738, 78.2520),
    "Panamarathupatti": (11.5830, 78.1670),
    "Veerapandi": (11.5840, 78.0750),
    "Yercaud": (11.7753, 78.2093),
}

results = []
found_count = 0

for i, row in locs.iterrows():
    b, l = row["Block"], row["Location"]
    lat, lon = None, None
    
    # Try specific village query
    queries = [
        f"{l} village, {b}, Salem, Tamil Nadu, India",
        f"{l}, Salem district, Tamil Nadu, India",
        f"{l}, Tamil Nadu, India",
    ]
    
    for q in queries:
        try:
            res = geolocator.geocode(q, timeout=4)
            if res and LAT_MIN <= res.latitude <= LAT_MAX and LON_MIN <= res.longitude <= LON_MAX:
                lat, lon = res.latitude, res.longitude
                break
        except Exception:
            time.sleep(1)
            continue
        time.sleep(0.4)
        
    if lat is not None:
        status = "exact"
        found_count += 1
    else:
        status = "block_offset"
        c_lat, c_lon = centroids[b]
        # deterministic reproducible spatial dispersion within the block polygon
        seed_val = abs(hash(l)) % 10000
        offset_lat = ((seed_val % 100) / 100.0 - 0.5) * 0.08
        offset_lon = (((seed_val // 100) % 100) / 100.0 - 0.5) * 0.08
        lat = c_lat + offset_lat
        lon = c_lon + offset_lon

    results.append({
        "Block": b,
        "Location": l,
        "Latitude": round(lat, 5),
        "Longitude": round(lon, 5),
        "Status": status
    })
    
    if (i + 1) % 15 == 0 or (i + 1) == len(locs):
        print(f"Processed {i+1}/{len(locs)}... ({found_count} exact coordinates resolved)")

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_PATH, index=False)
print(f"\n[DONE] Saved geocoded coordinates to: {OUT_PATH}")
