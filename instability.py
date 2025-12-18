import csv, requests

BASE = "https://volcanoes.usgs.gov/hans-public/api/search/search"
NOTICE_TYPES = ["VAN","VV","SR","IS","DU","WU","BU","MU"]

payload_base = {
    "obsAbbr": "hvo",
    "volcCd": "hi6",
    "startUnixtime": "915148800",
    "endUnixtime": "1767225599",
    "searchText": ""
}

rows = []
seen = set()

for nt in NOTICE_TYPES:
    page = 0
    while True:
        payload = dict(payload_base)
        payload["noticeTypeCd"] = nt
        payload["pageIndex"] = page

        r = requests.post(BASE, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        items = data.get("noticeData", [])
        if not items:
            break

        for it in items:
            nid = it.get("noticeIdentifier", "")
            if not nid or nid in seen:
                continue
            seen.add(nid)

            sent_utc = it.get("sentUtc", "")
            permlink = it.get("permLink", "")

            # If you prefer hans2 viewer instead of permlink:
            hans2_url = f"https://volcanoes.usgs.gov/hans2/view/notice/{nid}"

            rows.append([sent_utc, nt, nid, permlink, hans2_url])

        if len(items) < 10:
            break
        page += 1

rows.sort(key=lambda x: x[0])

out = "mauna_loa_notices_1999_2025.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sentUtc","noticeTypeCd","noticeIdentifier","permLink","hans2_url"])
    w.writerows(rows)

print(f"Saved {len(rows)} notices to {out}")
