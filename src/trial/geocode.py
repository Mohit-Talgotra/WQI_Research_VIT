import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

load_dotenv()

DATASET_PATH  = Path(os.environ["FORECAST_DATASET_PATH"])
OUTPUT_PATH   = Path(os.environ.get("GEOCODED_LOCATIONS_PATH", "data/locations_geocoded.csv"))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Salem district, Tamil Nadu — appended to every query to narrow results
REGION_SUFFIX = "Salem district, Tamil Nadu, India"

# Nominatim requires a unique user agent
geolocator = Nominatim(user_agent="wqi_research_vit_salem")

print("=" * 70)
print("GEOCODING LOCATIONS")
print("=" * 70)

# Load unique locations
df = pd.read_csv(DATASET_PATH)
locations = (
    df[["Block", "Location"]]
    .drop_duplicates()
    .sort_values(["Block", "Location"])
    .reset_index(drop=True)
)
print(f"  Total locations to geocode: {len(locations)}")

# If output already exists, skip already-geocoded locations
if OUTPUT_PATH.exists():
    existing = pd.read_csv(OUTPUT_PATH)
    already_done = set(zip(existing["Block"], existing["Location"]))
    print(f"  Already geocoded: {len(already_done)} — skipping these")
else:
    existing = pd.DataFrame()
    already_done = set()

results = []
failed  = []

for _, row in locations.iterrows():
    block    = row["Block"]
    location = row["Location"]

    if (block, location) in already_done:
        continue

    # Try progressively broader queries if narrow one fails
    queries = [
        f"{location}, {block}, {REGION_SUFFIX}",
        f"{location}, {REGION_SUFFIX}",
        f"{location}, Tamil Nadu, India",
    ]

    lat, lon, matched_query = None, None, None

    for query in queries:
        try:
            geo = geolocator.geocode(query, timeout=10)
            if geo:
                lat = geo.latitude
                lon = geo.longitude
                matched_query = query
                break
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"    WARN: geocoder error for '{query}': {e}")
            time.sleep(2)

        time.sleep(1)  # Nominatim rate limit: 1 request/second

    if lat and lon:
        print(f"  OK  {location} ({block}) → {lat:.4f}, {lon:.4f}")
        results.append({
            "Block"         : block,
            "Location"      : location,
            "Latitude"      : lat,
            "Longitude"     : lon,
            "geocode_query" : matched_query,
            "geocode_status": "ok",
        })
    else:
        print(f"  FAIL {location} ({block}) — not found, using block centroid fallback")
        failed.append({"Block": block, "Location": location})
        results.append({
            "Block"         : block,
            "Location"      : location,
            "Latitude"      : None,
            "Longitude"     : None,
            "geocode_query" : queries[0],
            "geocode_status": "failed",
        })

# Combine with existing
new_df = pd.DataFrame(results)
if not existing.empty:
    combined = pd.concat([existing, new_df], ignore_index=True)
else:
    combined = new_df

# For failed locations, fill with block centroid (mean of successful in same block)
block_centroids = (
    combined[combined["geocode_status"] == "ok"]
    .groupby("Block")[["Latitude", "Longitude"]]
    .mean()
    .reset_index()
    .rename(columns={"Latitude": "centroid_lat", "Longitude": "centroid_lon"})
)

combined = combined.merge(block_centroids, on="Block", how="left")
mask_failed = combined["geocode_status"] == "failed"
combined.loc[mask_failed, "Latitude"]  = combined.loc[mask_failed, "centroid_lat"]
combined.loc[mask_failed, "Longitude"] = combined.loc[mask_failed, "centroid_lon"]
combined = combined.drop(columns=["centroid_lat", "centroid_lon"])

combined.to_csv(OUTPUT_PATH, index=False)

print(f"\n  Successfully geocoded : {(combined['geocode_status'] == 'ok').sum()}")
print(f"  Used block centroid   : {mask_failed.sum()}")
print(f"  Saved: {OUTPUT_PATH}")

if failed:
    print(f"\n  Locations that fell back to centroid:")
    for f in failed:
        print(f"    {f['Location']} ({f['Block']})")

print("\n" + "=" * 70)
print("GEOCODING COMPLETE")
print("=" * 70)