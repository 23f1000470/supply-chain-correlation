# chart.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(2025)

categories = [
    "Electronics",
    "Home & Kitchen",
    "Clothing",
    "Sports & Outdoors",
    "Beauty & Personal Care",
    "Toys & Baby"
]

means = [78, 72, 80, 75, 77, 70]
sds   = [6, 7, 5, 6, 5, 8]
n = 160

rows = []
for cat, mu, sd in zip(categories, means, sds):
    samples = np.random.normal(mu, sd, n)
    samples = np.clip(samples, 0, 100)
    for x in samples:
        rows.append({"Category": cat, "Satisfaction": round(float(x), 1)})

df = pd.DataFrame(rows)

order = df.groupby("Category")["Satisfaction"].mean().sort_values(ascending=False).index

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.95)

plt.figure(figsize=(8, 8))   # 8 inches * 64 dpi = EXACT 512 px
ax = sns.barplot(
    x="Category",
    y="Satisfaction",
    data=df,
    order=order,
    ci="sd",
    capsize=0.1,
    palette=sns.color_palette("rocket_r", len(categories))
)

means_series = df.groupby("Category")["Satisfaction"].mean().reindex(order)

for i, m in enumerate(means_series):
    ax.text(i, m + 1.2, f"{m:.1f}%", ha="center", va="bottom", fontsize=10, weight="semibold")

ax.set_title("Average Customer Satisfaction by Product Category", fontsize=16, weight="bold", pad=14)
ax.set_ylabel("Average Satisfaction (%)")
ax.set_xlabel("")
ax.set_ylim(0, 100)
plt.xticks(rotation=20, ha="right")

sns.despine()
plt.tight_layout()

# THE CRITICAL FIX — EXACT OUTPUT SIZE
plt.savefig("chart.png", dpi=64)   # must be EXACTLY this
plt.close()

print("chart.png saved at exactly 512x512 pixels.")
