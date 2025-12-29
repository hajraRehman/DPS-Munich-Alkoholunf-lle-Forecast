import pandas as pd
import matplotlib.pyplot as plt

# Load the original data
df = pd.read_csv('data/monatszahlen.csv')

# Filter to data up to 2020 and valid months
df = df[df['JAHR'] <= 2020]
df = df[df['MONAT'].astype(str).str.len() == 6].copy()

# Create proper date
df['JAHR'] = df['JAHR'].astype(int)
df['MONAT_NUM'] = df['MONAT'].astype(str).str[-2:].astype(int)
df['DATE'] = pd.to_datetime(df['JAHR'].astype(str) + '-' + df['MONAT_NUM'].astype(str).str.zfill(2) + '-01')

# Group by category (MONATSZAHL) and date, sum WERT
category_df = df.groupby(['MONATSZAHL', 'DATE'])['WERT'].sum().reset_index()

# Plot each category
plt.figure(figsize=(14, 8))
for category in category_df['MONATSZAHL'].unique():
    data = category_df[category_df['MONATSZAHL'] == category]
    plt.plot(data['DATE'], data['WERT'], label=category)

plt.title('Historical Number of Traffic Accidents by Category in Munich (2000–2020)')
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('historical_viz.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as historical_viz.png")