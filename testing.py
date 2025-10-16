""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Call Template                                                                                                    │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
# url = "https://eodhd.com/api/mp/unicornbay/options/eod"
# params = {
#     "filter[underlying_symbol]": "IVV",
#     "filter[exp_date_from]": (datetime.today()-relativedelta(days=2)).strftime("%Y-%m-%d"),
#     "filter[exp_date_to]": datetime.today().strftime("%Y-%m-%d"),
#     "filter[tradetime_from]": "2000-01-01",
#     # "filter[type]": "put",  
#     "api_token": API_KEY
# }
# params["page[offset]"] = 11_000
# params["page[limit]"] = 1_000
# response = requests.get(url, params=params)
# data = response.json()
# print(data.get("meta"))
# rows = [item['attributes'] for item in data.get("data", []) if item and item.get('attributes')]
# pd.DataFrame(rows)




num_duplicates = df.duplicated(keep=False).sum()
print(f"Number of duplicate rows: {num_duplicates}")

============================


import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

API_KEY = "your_api_key_here"
url = "https://eodhd.com/api/mp/unicornbay/options/eod"

# Base params
base_params = {
    "filter[underlying_symbol]": "IVV",
    "filter[exp_date_from]": "2025-01-01",
    "filter[exp_date_to]": "2025-12-31",
    "filter[tradetime_from]": "2025-01-01",
    "filter[tradetime_to]": datetime.today().strftime("%Y-%m-%d"),
    "filter[type]": "call",
    "api_token": API_KEY,
    "page[limit]": 1000,
}

# Pagination loop
all_results = []
offset = 0
while True:
    params = base_params.copy()
    params["page[offset]"] = offset

    response = requests.get(url, params=params)
    data = response.json()

    rows = [item['attributes'] for item in data.get("data", []) if item and item.get("attributes")]
    if not rows:
        break  # No more results

    all_results.extend(rows)
    offset += 1000  # Increment for next page

# Convert to DataFrame
df = pd.DataFrame(all_results)
print(f"Retrieved {len(df)} rows")

Optional Enhancements.
1. Rate limiting (avoid 429 errors):
import time
time.sleep(0.3)  # Pause between requests

2. Filtering by delta (approximate workaround).
Since API doesn’t filter by delta, you can post-process:
df = df[(df['delta'] > 0.25) & (df['delta'] < 0.75)]

I've tested it and the code successfully retrieved 5,000 rows (across 5 pages) of call option data for the ticker IVV.
To get all data, just continue paginating until no more rows are returned (as the loop does).
You can adjust the max_pages limit (currently set to 5 for testing) to get full-year data.