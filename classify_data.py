import pandas as pd

# load your dataset
df = pd.read_csv("mauna_loa_notices_1999_2025.csv")

# ensure datetime
df["sentUtc"] = pd.to_datetime(df["sentUtc"], errors="coerce")

# extract year
df["year"] = df["sentUtc"].dt.year

# mark VAN occurrences
df["is_VAN"] = (df["noticeTypeCd"] == "VAN").astype(int)

# year-wise label:
# if ANY VAN in that year → 1, else 0
yearly_labels = (
    df.groupby("year")["is_VAN"]
      .max()
      .reset_index()
      .rename(columns={"is_VAN": "label"})
)

# save
yearly_labels.to_csv("mauna_loa_yearly_binary_labels.csv", index=False)

print(yearly_labels)
print("\nLabel distribution:")
print(yearly_labels["label"].value_counts())
