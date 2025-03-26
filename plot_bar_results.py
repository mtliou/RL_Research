import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = 'cleaned_M4_benchmarks.csv'  # Replace with your file path
df = pd.read_csv(csv_file_path)

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")

# Define custom colors
method_colors = {
    "Method 1": "cyan",
    "Method 2": "cyan",
    "Method 3": "cyan",
    "Method 4": "cyan",
    "MLP benchmark": "lightpink",
    "RNN benchmark": "lightpink"
}
default_color = "magenta"  # Default color for other methods

# Function to add numerical values inside the bars in bold
def add_values_on_bars(ax):
    for bar in ax.patches:
        value = bar.get_height()
        if not pd.isna(value):  # Check if the value is not NaN
            ax.annotate(
                f"{value:.2f}",  # Format the value with two decimal places
                (bar.get_x() + bar.get_width() / 2, value / 2),  # Position inside the bar
                ha='center', va='center', fontsize=10, fontweight='bold', color='black'
            )

# Plot MASE and sMAPE Mean
for measure in ["MASE_Mean", "sMAPE_Mean"]:
    for category in df["Category"].unique():
        subset = df[df["Category"] == category]
        plt.figure(figsize=(12, 8))

        # Assign colors based on the method
        colors = [
            method_colors[method] if method in method_colors else default_color
            for method in subset["Method"]
        ]

        bars = plt.bar(subset["Method"], subset[measure], color=colors)
        plt.title(f"{measure} for {category} Category", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=10, fontweight='bold')
        plt.ylabel(measure)
        plt.xlabel("Method")
        add_values_on_bars(plt.gca())  # Call the function to add values
        plt.tight_layout()
        # Save the figure
        filename = f"{measure}_{category}.png".replace(" ", "_")  # Replace spaces with underscores
        plt.tight_layout()
        plt.savefig(filename, dpi=300)  # Save the figure with high resolution (300 DPI)
        plt.show()
