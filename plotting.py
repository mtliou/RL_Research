import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
data = pd.read_csv("cleaned_performance_data.csv")

# Group the data by temporal series (Hourly, Daily, etc.)
grouped_data = data["Series_Method"].str.extract(r'(\w+)\s')[0].unique()
grouped_data = {group: data[data["Series_Method"].str.contains(group)] for group in grouped_data}

# Function to plot MASE and sMAPE with bar charts and numerical values inside the bars
def plot_bar_metrics(data, title, filename):
    methods = data["Series_Method"].values
    x = np.arange(len(methods))  # X positions for the groups

    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    mase_bars = ax.bar(x - width/2, data["MASE_Mean"], width, yerr=data["MASE_Std"], label='MASE', capsize=5, color='cyan')
    smape_bars = ax.bar(x + width/2, data["sMAPE_Mean"], width, yerr=data["sMAPE_Std"], label='sMAPE', capsize=5, color='magenta')

    # Adding labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_title(f'{title} - MASE and sMAPE with Std Dev (Bar Chart)')
    ax.set_ylabel('Performance Metrics')
    ax.set_xlabel('Methods')
    ax.legend()

    # Adding numerical values inside the bars
    for bar in mase_bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, yval / 2,  # Middle of the bar
            f'{yval:.2f}', ha='center', va='center', color='black', fontsize=12
        )
    for bar in smape_bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, yval / 2,  # Middle of the bar
            f'{yval:.2f}', ha='center', va='center', color='black', fontsize=12
        )

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Example usage for grouped data
for group, data in grouped_data.items():
    plot_bar_metrics(data, f"{group} Data", f"{group}_Data.png")
