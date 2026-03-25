import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.environ["OXYLABS_USERNAME"]
PASSWORD = os.environ["OXYLABS_PASSWORD"]

# Test 1: Google Maps search URL
print("=" * 70)
print("TEST 1: Google Maps search URL")
print("=" * 70)

maps_url = "https://www.google.com/maps/search/Valaiyakaranur+Ayodhiyapattanam+Salem+Tamil+Nadu+India"

payload = {
    "source"      : "google",
    "url"         : maps_url,
    "render"      : "html",
    "geo_location": "India",
}

response = requests.post(
    "https://realtime.oxylabs.io/v1/queries",
    auth=(USERNAME, PASSWORD),
    json=payload,
    timeout=30,
)

print(f"Status code : {response.status_code}")
print(f"Response keys: {list(response.json().keys())}")

data = response.json()
results = data.get("results", [])
print(f"Results count: {len(results)}")

if results:
    r = results[0]
    print(f"\nResult keys: {list(r.keys())}")
    print(f"Final URL  : {r.get('url', 'N/A')}")
    print(f"Status code: {r.get('status_code', 'N/A')}")

    content = r.get("content", "")
    print(f"\nContent length: {len(content)} chars")
    print(f"\nFirst 2000 chars of content:\n{content[:2000]}")
    print(f"\n...\n")
    print(f"Last 1000 chars of content:\n{content[-1000:]}")

    # Save full response for inspection
    with open("debug_oxylabs_response.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\nFull response saved to: debug_oxylabs_response.json")

# Test 2: Google search (not Maps)
print("\n" + "=" * 70)
print("TEST 2: Regular Google search")
print("=" * 70)

payload2 = {
    "source" : "google_search",
    "query"  : "Valaiyakaranur Ayodhiyapattanam Salem Tamil Nadu coordinates",
    "geo_location": "India",
    "locale" : "en-IN",
}

response2 = requests.post(
    "https://realtime.oxylabs.io/v1/queries",
    auth=(USERNAME, PASSWORD),
    json=payload2,
    timeout=30,
)

print(f"Status code: {response2.status_code}")
data2 = response2.json()

if response2.status_code == 200:
    results2 = data2.get("results", [])
    if results2:
        content2 = results2[0].get("content", "")
        print(f"Content length: {len(content2)} chars")

        # Try to find any coordinate-like patterns in the response
        import re
        coord_patterns = [
            r"(-?\d{1,3}\.\d{4,})[,\s]+(-?\d{1,3}\.\d{4,})",
            r"latitude[\":\s]+(-?\d+\.\d+)",
            r"longitude[\":\s]+(-?\d+\.\d+)",
            r"/@(-?\d+\.\d+),(-?\d+\.\d+)",
        ]
        for pat in coord_patterns:
            matches = re.findall(pat, content2[:5000])
            if matches:
                print(f"  Pattern '{pat[:40]}...' found: {matches[:3]}")

        print(f"\nFirst 3000 chars:\n{content2[:3000]}")

        with open("debug_oxylabs_search_response.json", "w") as f:
            json.dump(data2, f, indent=2)
        print("\nFull search response saved to: debug_oxylabs_search_response.json")
else:
    print(f"Error: {data2}")

print("\n" + "=" * 70)
print("Check debug_oxylabs_response.json and debug_oxylabs_search_response.json")
print("Share the output above so we can see what Oxylabs returns")
print("=" * 70)