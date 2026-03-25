import os
import re
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEOCODED_PATH = Path(os.environ["GEOCODED_LOCATIONS_PATH"])
USERNAME      = os.environ["OXYLABS_USERNAME"]
PASSWORD      = os.environ["OXYLABS_PASSWORD"]
OXYLABS_URL   = "https://realtime.oxylabs.io/v1/queries"

# Salem district bounding box for sanity checking
LAT_MIN, LAT_MAX = 10.0, 13.5
LON_MIN, LON_MAX = 76.0, 80.5

# Coordinate extraction
def extract_coords(html: str):
    """
    Try every known pattern Google embeds in search result pages.
    Returns first (lat, lon) pair that passes the Tamil Nadu sanity check.
    """
    patterns = [
        r"/@(-?\d+\.\d+),(-?\d+\.\d+)",
        r'"latitude"\s*:\s*"?(-?\d+\.\d+)"?[,\s]+"longitude"\s*:\s*"?(-?\d+\.\d+)"?',
        r'"lat"\s*:\s*(-?\d+\.\d+)[,\s]+"lng"\s*:\s*(-?\d+\.\d+)',
        r"maps\.google\.[a-z]+/\?[^\"']*ll=(-?\d+\.\d+),(-?\d+\.\d+)",
        r"maps\.google\.[a-z]+/maps\?[^\"']*q=(-?\d+\.\d+),(-?\d+\.\d+)",
        r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)",
        r'data-lat="(-?\d+\.\d+)"[^>]*data-lng="(-?\d+\.\d+)"',
        r'data-latlng="(-?\d+\.\d+),(-?\d+\.\d+)"',
        r"(-1?\d{1,2}\.\d{5,}),\s*(-?\d{1,2}\.\d{5,})",
    ]

    for pat in patterns:
        for m in re.finditer(pat, html):
            try:
                lat, lon = float(m.group(1)), float(m.group(2))
                if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                    return lat, lon
            except (ValueError, IndexError):
                continue

    return None, None


def geocode_via_oxylabs(location: str, block: str):
    queries = [
        f"{location} gram panchayat {block} Salem Tamil Nadu",
        f"{location} village Salem district Tamil Nadu map",
        f"{location} Salem Tamil Nadu India location coordinates",
    ]

    for query in queries:
        payload = {
            "source"      : "google_search",
            "query"       : query,
            "geo_location": "Tamil Nadu,India",
            "locale"      : "en-IN",
            "parse"       : False,
        }

        try:
            resp = requests.post(
                OXYLABS_URL,
                auth=(USERNAME, PASSWORD),
                json=payload,
                timeout=30,
            )

            if resp.status_code != 200:
                continue

            results = resp.json().get("results", [])
            if not results:
                continue

            html = results[0].get("content", "")
            lat, lon = extract_coords(html)

            if lat and lon:
                return lat, lon, query

        except Exception as e:
            print(f"    Error: {e}")

        time.sleep(1)

    return None, None, "failed"


# Load and filter failed locations
geo    = pd.read_csv(GEOCODED_PATH)
failed = geo[geo["geocode_status"] == "failed"].copy()
failed = failed.head(1)

print("=" * 70)
print("RE-GEOCODING FAILED LOCATIONS VIA OXYLABS (google_search)")
print("=" * 70)
print(f"  Locations to retry: {len(failed)}\n")

if len(failed) == 0:
    print("  Nothing to do.")
    exit(0)

# Geocode each failed location
new_rows = []

for i, (_, row) in enumerate(failed.iterrows()):
    location = row["Location"]
    block    = row["Block"]

    print(f"  [{i+1:02d}/{len(failed)}] {location} ({block}) ... ", end="", flush=True)

    lat, lon, matched_query = geocode_via_oxylabs(location, block)

    if lat and lon:
        print(f"✓  {lat:.5f}, {lon:.5f}")
        new_rows.append({
            "Block"         : block,
            "Location"      : location,
            "Latitude"      : lat,
            "Longitude"     : lon,
            "geocode_query" : matched_query,
            "geocode_status": "ok_oxylabs",
        })
    else:
        print("✗  still failed")
        new_rows.append({
            "Block"         : block,
            "Location"      : location,
            "Latitude"      : None,
            "Longitude"     : None,
            "geocode_query" : "failed",
            "geocode_status": "failed",
        })

    time.sleep(0.5)

# Merge back into main geocoded file
new_df      = pd.DataFrame(new_rows)
geo_updated = geo[geo["geocode_status"] != "failed"].copy()
geo_updated = pd.concat([geo_updated, new_df], ignore_index=True)

# Centroid fallback for any still-failed
block_centroids = (
    geo_updated[geo_updated["geocode_status"].isin(["ok", "ok_oxylabs"])]
    .groupby("Block")[["Latitude", "Longitude"]]
    .mean()
    .reset_index()
    .rename(columns={"Latitude": "centroid_lat", "Longitude": "centroid_lon"})
)

geo_updated  = geo_updated.merge(block_centroids, on="Block", how="left")
still_failed = geo_updated["geocode_status"] == "failed"
geo_updated.loc[still_failed, "Latitude"]       = geo_updated.loc[still_failed, "centroid_lat"]
geo_updated.loc[still_failed, "Longitude"]      = geo_updated.loc[still_failed, "centroid_lon"]
geo_updated.loc[still_failed, "geocode_status"] = "centroid_fallback"
geo_updated = geo_updated.drop(columns=["centroid_lat", "centroid_lon"])

geo_updated.to_csv(GEOCODED_PATH, index=False)

# Summary
print("\n" + "=" * 70)
print("FINAL STATUS:")
print(geo_updated["geocode_status"].value_counts().to_string())

still_bad = geo_updated[geo_updated["geocode_status"] == "centroid_fallback"]
if len(still_bad):
    print(f"\n  {len(still_bad)} using block centroid fallback:")
    for _, r in still_bad.iterrows():
        print(f"    {r['Location']} ({r['Block']})")
    print(f"\n  To fix manually, edit: {GEOCODED_PATH}")

print(f"\nSaved: {GEOCODED_PATH}")
print("=" * 70)