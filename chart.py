# chart.py
# Generates a professional Seaborn barplot of customer satisfaction by product category.
# Author: generated for 23f1000470@ds.study.iitm.ac.in
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reproducible random seed
RNG = 2025
np.random.seed(RNG)

# ---- Synthetic data generation (realistic) ----
# Product categories and their typical mean satisfaction (scale 0-100)
categories = [
    "Electronics",
    "Home & Kitchen",
    "Clothing",
    "Sports & Outdoors",
    "Beauty & Personal Care",
    "Toys & Baby"
]
means = [78, 72, 80, 75, 77, 70]     # realistic mean satisfaction % by category
sds =   [6, 7, 5, 6, 5, 8]           # typical std deviations
n_per_cat = 160                      # number of customers sampled per category

rows = []
for cat, mu, sd in zip(categories, means, sds):
    samples = np.random.normal(loc=mu, scale=sd, size=n_per_cat)
    # Clamp to 0-100 range and round to 1 decimal for realism
    samples = np.clip(samples, 0, 100)
    for s in samples:
        rows.append({"Category": cat, "Satisfaction": round(float(s), 1)})

df = pd.DataFrame(rows)

# ---- Prepare summary order: sort categories by mean satisfaction descending ----
order = df.groupby("Category")["Satisfaction"].mean().sort_values(ascending=False).index.tolist()

# ---- Styling ----
sns.set_style("whitegrid")                  # professional background grid
sns.set_context("talk", font_scale=0.95)    # presentation-ready text sizes
palette = sns.color_palette("rocket_r", n_colors=len(categories))  # attractive palette

# ---- Create the barplot ----
plt.figure(figsize=(8, 8))  # 8x8 inches
ax = sns.barplot(
    x="Category",
    y="Satisfaction",
    data=df,
    order=order,
    estimator=np.mean,
    ci="sd",               # show standard deviation as error bar
    capsize=0.08,
    palette=palette
)

# Add exact mean labels above bars
means_series = df.groupby("Category")["Satisfaction"].mean().reindex(order)
for i, (cat, mean_val) in enumerate(means_series.items()):
    ax.text(i, mean_val + 1.4, f"{mean_val:.1f}%", ha="center", va="bottom", fontsize=10, weight="semibold")

# Titles and labels (publication-ready)
ax.set_title("Average Customer Satisfaction by Product Category", fontsize=16, weight="bold", pad=14)
ax.set_ylabel("Average Satisfaction (%)", fontsize=12)
ax.set_xlabel("")
ax.set_ylim(0, 100)

# Improve layout
plt.xticks(rotation=20, ha="right")
sns.despine(trim=True)
plt.tight_layout()

# ---- Save the figure as exactly 512x512 px ----
# figsize 8x8 inches * dpi 64 = 512 x 512 pixels
outfile = "chart.png"
plt.savefig(outfile, dpi=64, bbox_inches="tight")
plt.close()

print(f"Saved chart to {outfile} (512x512 px).")
