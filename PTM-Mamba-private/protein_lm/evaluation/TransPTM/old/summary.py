import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Plot settings
plot_settings = {
    'ytick.labelsize': 16,
    'xtick.labelsize': 16,
    'font.size': 22,
    'figure.figsize': (10, 5),
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'lines.linewidth': 2,
    'lines.markersize': 3,
    'legend.fontsize': 11,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral'
}
plt.rcParams.update(plot_settings)
sns.set_context("paper", rc={"font.size":16, "axes.titlesize":16, "axes.labelsize":16})

def load_results(csv_path):
    return pd.read_csv(csv_path)

def map_model_names(model_name):
    if model_name == 'ptm_mamba':
        return 'PTM-Mamba'
    elif 'esm' in model_name:
        parts = model_name.split('_')
        model_size = parts[1]
        return f'ESM-2-{model_size.upper()}'
    else:
        return model_name

def summarize_results(df):
    df['Model'] = df['Model'].apply(map_model_names)
    df = df[df['F1'] > 0]
    summarized_df = df.groupby(['Model', 'Seq_Len']).mean().reset_index()
    return summarized_df

def plot_avg_metrics(df):
    avg_df = df.groupby(['Model']).mean().reset_index().round(3)
    avg_df.set_index('Model', inplace=True)

    # Plotting
    ax = avg_df.T.plot(kind='bar', figsize=(10, 6))
    ax.set_xlabel('Metrics')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

    plt.show()

def plot_metrics_by_length(df):
    metrics = df.columns[2:]
    os.makedirs('figures', exist_ok=True)
    models = df['Model'].unique()
    colors = sns.color_palette('husl', len(models))

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for model, color in zip(models, colors):
            sns.lineplot(x='Seq_Len', y=metric, data=df[df['Model'] == model], label=model, marker='o', color=color)
        
        plt.xlabel('Length', fontsize=16)
        plt.ylabel(metric, fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(f'figures/{metric}.png')
        plt.show()

def main(csv_path):
    df = load_results(csv_path)
    summarized_df = summarize_results(df)
    plot_avg_metrics(summarized_df)
    plot_metrics_by_length(summarized_df)

if __name__ == "__main__":
    # Set the path to your CSV file here
    csv_path = 'test_results.csv'
    main(csv_path)
