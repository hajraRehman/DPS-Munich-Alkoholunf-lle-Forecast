import pandas as pd

# Load dataset
df = pd.read_csv('data/monatszahlen.csv')

# Filter for target
mask = (df["MONATSZAHL"] == "Alkoholunfälle") & (df["AUSPRAEGUNG"] == "insgesamt")
df_alk = df[mask].copy()


# Drop invalid MONAT (e.g., 'Summe')
df_alk = df_alk[df_alk["MONAT"].astype(str).str.len() == 6].copy()

# Date processing
df_alk["JAHR"] = df_alk["JAHR"].astype(int)
df_alk["MONAT_NUM"] = df_alk["MONAT"].astype(str).str[-2:].astype(int)
df_alk["DATE"] = pd.to_datetime(df_alk["JAHR"].astype(str) + "-" + df_alk["MONAT_NUM"].astype(str) + "-01")
df_alk = df_alk.sort_values("DATE").reset_index(drop=True)

# Split: Train up to 2020, future after
train = df_alk[df_alk["JAHR"] <= 2020].copy()[["DATE", "WERT"]].set_index("DATE")
future = df_alk[df_alk["JAHR"] > 2020].copy()[["DATE", "WERT"]].set_index("DATE")

# Save for reuse (optional, but professional for caching)
train.to_csv("train_data.csv")
future.to_csv("future_data.csv")

print("Train shape:", train.shape)
print("Future shape:", future.shape)