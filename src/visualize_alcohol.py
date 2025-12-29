import pandas as pd
import matplotlib.pyplot as plt

# Load the original data
df = pd.read_csv('data/monatszahlen.csv')

# Filter to Alkoholunfälle insgesamt
mask = (df["MONATSZAHL"] == "Alkoholunfälle") & (df["AUSPRAEGUNG"] == "insgesamt")
df_alk = df[mask].copy()

# Keep only valid months and all years (including post-2020 for full context)
df_alk = df_alk[df_alk["MONAT"].astype(str).str.len() == 6].copy()

# Create date column
df_alk["JAHR"] = df_alk["JAHR"].astype(int)
df_alk["MONAT_NUM"] = df_alk["MONAT"].astype(str).str[-2:].astype(int)
df_alk["DATE"] = pd.to_datetime(
    df_alk["JAHR"].astype(str) + "-" + df_alk["MONAT_NUM"].astype(str).str.zfill(2) + "-01"
)

# Sort by date
df_alk = df_alk.sort_values("DATE").reset_index(drop=True)

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df_alk["DATE"], df_alk["WERT"], marker='o', linestyle='-', color='darkred', markersize=4)

# Highlight the training period (up to 2020)
plt.axvline(pd.to_datetime("2021-01-01"), color='gray', linestyle='--', linewidth=1.5, label="Train/Forecast Split")

plt.title("Historical Alcohol-Related Traffic Accidents (Alkoholunfälle insgesamt) in Munich\n(2000–2025)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Accidents per Month", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save as high-quality PNG
plt.savefig('alcohol_historical_viz.png', dpi=300, bbox_inches='tight')
plt.show()

print("Alcohol-specific visualization saved as alcohol_historical_viz.png")
print(f"Data points: {len(df_alk)} months")
print(f"Latest value (Dec 2025): {df_alk['WERT'].iloc[-1] if not df_alk.empty else 'N/A'}")