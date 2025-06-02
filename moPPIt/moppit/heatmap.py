import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data preparation
data = {
    "Position": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 105, 106, 107, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149],
    "Relative interface score (%)": [9.26, 10.19, 14.84, 8.27, 8.27, 8.28, 8.28, 8.24, 5.90, None, None, 11.65, 16.46, 35.29, 39.85, 47.36, 47.37, 47.37, 47.47, 47.40, 32.52, 11.03, 17.38, 11.94, 12.87, 12.76, 12.76, 12.76, 12.98, 12.91, 1.27, 0.92, 11.74, 11.74, 11.74, None, None, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
}

df = pd.DataFrame(data)
df.set_index("Position", inplace=True)

# Plotting the heatmap
plt.figure(figsize=(15, 6))
sns.heatmap(df.T, annot=True, cmap="coolwarm", cbar_kws={'label': 'Score'})
plt.title('Heatmap of Binding Scores by Position')
plt.xlabel('Position')
plt.ylabel('Score Type')
plt.show()
