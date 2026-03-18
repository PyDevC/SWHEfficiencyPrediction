import matplotlib.pyplot as plt

# Data preparation
metrics = {
    "Metric": ["R2 Score", "MAE", "MAPE", "Explained Variance", "RMSE"],
    "Value": [0.634743, 0.075470, 3.029521, 0.638928, 0.117832]
}

# Create figure
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')  # Hide the axes

# Format the data: Rounding values for better readability
table_data = [[m, f"{v:.4f}"] for m, v in zip(metrics["Metric"], metrics["Value"])]

# Create the table
table = ax.table(
    cellText=table_data,
    colLabels=["Model Metric", "Score"],
    cellLoc='center',
    loc='center',
    colColours=["#40466e", "#40466e"]  # Dark header background
)

# Styling the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.5)  # Scale (width, height) to add padding

# Iterate through cells to apply custom styling
for (row, col), cell in table.get_celld().items():
    if row == 0:  # Header row
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
    else:
        # Alternating row colors for better readability
        if row % 2 == 0:
            cell.set_facecolor('#f2f2f2')
        else:
            cell.set_facecolor('white')
    
    # Remove borders for a cleaner look
    cell.set_linewidth(0.5)
    cell.set_edgecolor('#dddddd')

plt.title('Final Performance Analysis', fontsize=16, pad=20, weight='bold')
plt.show()
