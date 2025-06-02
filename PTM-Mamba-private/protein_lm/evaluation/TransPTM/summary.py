import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import protein_lm.evaluation.TransPTM.config as config
warnings.filterwarnings("ignore")

global color_map
esm_map = {
    "esm2_t48_15B_UR50D": "ESM-2-15B",                           
    "esm2_t36_3B_UR50D": "ESM-2-3B",                   
    "esm2_t33_650M_UR50D": "ESM-2-650M",             
    "esm2_t30_150M_UR50D": "ESM-2-150M",
    "esm2_t12_35M_UR50D": "ESM-2-35M",          
    "esm2_t6_8M_UR50D": "ESM-2-8M"
}
color_map = {
    'ESM-2-15B': '#8B0000',
    'ESM-2-3B': '#EC5800',
    'ESM-2-650M': '#fecc00',
    'ESM-2-150M': '#8F00FF',
    'ESM-2-35M': '#9370DB',
    'ESM-2-8M': '#D8BFD8',
    'PTM-Mamba': '#008000',
    'PTM-Transformer': "#AC71F3",
    'PTM-Mamba-SaProt': "#7220D6",      
}

# Set the seaborn style and font scale for publication
def set_style():
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
        return esm_map[model_name]
    elif model_name=='ptm_transformer':
        return 'PTM-Transformer'
    elif model_name=='mamba_saprot':
        return 'PTM-Mamba-SaProt'
    else:
        return model_name

def summarize_results(df,across='replicate'):
    """
    'across' can be 'replicate' or 'seq_len' to summarize by either
    """
    df['model'] = df['model'].apply(map_model_names)
    #df = df[df['F1'] > 0]
    if across=='replicate':
        summarized_df = df.groupby(['model', 'seq_len']).mean().reset_index()
        summarized_df = summarized_df.drop(columns=['replicate'])
    if across=='seq_len':
        summarized_df = df.groupby(['model', 'replicate']).mean().reset_index()
        summarized_df = summarized_df.drop(columns=['seq_len'])
    # average first by replicate, then by seq len
    if across is None:
        summarized_df = df.groupby(['model', 'seq_len']).mean().reset_index()
        summarized_df = summarized_df.drop(columns=['replicate'])
        summarized_df = summarized_df.groupby(['model']).mean().reset_index()
        summarized_df = summarized_df.drop(columns=['seq_len'])
    return summarized_df

def plot_avg_metrics(df, plot_trials=False):
    """
    if plot_trials = True, we'll plot across "trials" (replicates or seq length, depending on the df passed in)
    """
    metrics_to_plot = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUPRC', 'AUROC']
    # get the title now
    title='Average Test Metrics'
    if plot_trials:
        if 'replicate' in list(df.columns):
            title = title + ' and Averages Per Replicate'
        elif 'seq_len' in list(df.columns):
            title = title + ' and Averages Per Sequence Length'
            
    # Rename the columns as specified
    df = df.rename(columns={'model': 'Model',
        'loss': 'Loss', 'acc': 'Accuracy',
        'precision': 'Precision', 
        'recall': 'Recall', 
        'f1': 'F1', 
        'mcc': 'MCC', 
        'auprc': 'AUPRC', 
        'auroc': 'AUROC'
    })

    # Initialize a dictionary to store the average metrics for each method
    model_names = list(df['Model'].unique())
    average_metrics = {model: {} for model in model_names}

    for model in model_names:
        sub_df = df.loc[df['Model']==model].reset_index(drop=True)

        for metric in metrics_to_plot:  # Only consider the specified metrics
            # Store the average metrics and metrics/run for this method
            average_metrics[model][metric] = {
                'Avg': sub_df[metric].mean(),
                'All': sub_df[metric].values.tolist()
            }

    # Convert the dictionary to a DataFrame for easier plotting
    metrics_df = pd.DataFrame(average_metrics).reset_index().melt(id_vars='index', var_name='Model', value_name='Value')
    metrics_df[['Avg', 'All']] = metrics_df['Value'].apply(pd.Series)
    metrics_df = metrics_df.drop('Value', axis=1)
    metrics_df = metrics_df.explode('All').reset_index(drop=True)
    metrics_df.rename(columns={'index': 'Metric'}, inplace=True)
    metrics_df = metrics_df[metrics_df['Metric'].isin(metrics_to_plot)]  # Filter the DataFrame to include only the specified metrics

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot bars for the average metrics
    colors = [color_map.get(model, '#000000') for model in metrics_df.index]  # Use black as default if model not found
    #avg_df[metrics].T.plot(kind='bar', ax=ax, color=colors, legend=False)
    bar_plot = sns.barplot(x='Metric', y='Avg', hue='Model', data=metrics_df, errorbar=None, palette=color_map, dodge=True) # bars
    if plot_trials: # plot the individual dots if wanted
        strip_plot = sns.stripplot(x='Metric', y='All', hue='Model', data=metrics_df, dodge=True, jitter=False, palette=color_map, size=4, marker='o', edgecolor='black', linewidth=0.5) # results for each run

    # Formatting the plot
    ax.set_title(title,fontsize=16)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('')
    # set a larger size for the x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    # Customize the legend to exclude the dots
    handles, labels = bar_plot.get_legend_handles_labels()
    plt.legend(handles[0:len(model_names)], labels[0:len(model_names)], bbox_to_anchor=(0.6, 1), loc='upper left',fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_metrics_by_length(df, plot_trials=False):
    """
    if plot_trials = True, we'll plot a dot for each "trials" (replicate)
    """
    set_style()
    
    # Rename columns for better labels
    df = df.rename(columns={
        'model': 'Model',
        'loss': 'Loss', 'acc': 'Accuracy',
        'precision': 'Precision', 
        'recall': 'Recall', 
        'f1': 'F1', 
        'mcc': 'MCC', 
        'auprc': 'AUPRC', 
        'auroc': 'AUROC'
    })

    # Metrics to plot
    metrics_to_plot = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUPRC', 'AUROC']
    individual_figs = dict()

    # Create a 2x4 plot grid
    fig, axs = plt.subplots(2, 4, figsize=(24, 15))
    axs = axs.flatten()  # Flatten the grid

    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i]
        # Individual plot
        fig_metric, ax_metric = plt.subplots(figsize=(10, 6))

        # Plot individual points for each model - only if indicated
        if plot_trials:
            for model in df['Model'].unique():
                sub_df = df[df['Model'] == model].reset_index(drop=True)
                scatter_full = ax.scatter(sub_df['seq_len'], sub_df[metric], alpha=0.5, color=color_map[model], edgecolor='black', s=10)
                scatter_individual = ax_metric.scatter(sub_df['seq_len'], sub_df[metric], alpha=0.5, color=color_map[model], edgecolor='black', s=10)
        # Plot the average lines
        avg_df = df.groupby(['Model', 'seq_len'])[metric].mean().reset_index()
        for model in df['Model'].unique():
            model_avg_df = avg_df[avg_df['Model'] == model].reset_index(drop=True)
            avg_full = ax.plot(model_avg_df['seq_len'], model_avg_df[metric], marker='o', color=color_map[model], label=f'{model} Avg')
            avg_individual = ax_metric.plot(model_avg_df['seq_len'], model_avg_df[metric], marker='o', color=color_map[model], label=f'{model} Avg')

        # Adjust labels and title
        if plot_trials:
            title=metric+' Across Replicates'
        else:
            title=f'Average {metric}'
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Length', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # same thing for ax individual
        ax_metric.set_title(title, fontsize=16)
        ax_metric.set_xlabel('Length', fontsize=14)
        ax_metric.set_ylabel(metric, fontsize=14)
        ax_metric.tick_params(axis='x', labelsize=12)
        ax_metric.tick_params(axis='y', labelsize=12)

        # Set legend for individual
        ax_metric.legend(loc='best', fontsize=14, frameon=True)  # Adjust `loc` as needed (e.g., 'upper right', 'lower left')

        fig_metric.tight_layout()

        # Save individual figure in dictionary
        if plot_trials:
            individual_figs[f'{metric}_with_replicates.png'] = fig_metric
        else:
            individual_figs[f'{metric}.png'] = fig_metric

        # Add a single legend for the entire figure
        ax.legend(loc='best', fontsize=14, frameon=True)  # Adjust `loc` as needed (e.g., 'upper right', 'lower left')

    fig.tight_layout()
    
    return individual_figs, fig
        
def make_all_tables_and_plots(csv_dir):
    # only difference between this method and main is it doesn't utilize config file
    csv_path = f'{csv_dir}/test_results.csv'
    
    df = load_results(csv_path)
    
    # make results directories if they aren't there
    os.makedirs(f'{csv_dir}/figures', exist_ok=True)
    os.makedirs(f'{csv_dir}/figures/individual_points_plotted', exist_ok=True)
    os.makedirs(f'{csv_dir}/figures/averages_only', exist_ok=True)
    
    # Make and save the summary dataframes: one where results are averaged across replicates, one where results are averaged across seq len, one where results are totally averaged
    summarized_df = summarize_results(df, across=None)
    summarized_df_across_replicates = summarize_results(df, across='replicate')
    summarized_df_across_lengths = summarize_results(df, across='seq_len')
    
    summarized_df.to_csv(f'{csv_dir}/summarized_test_results.csv',index=False)
    summarized_df_across_replicates.to_csv(f'{csv_dir}/summarized_test_results_across_reps.csv',index=False)
    summarized_df_across_lengths.to_csv(f'{csv_dir}/summarized_test_results_across_seqlens.csv',index=False)
    
    # Make and save the average metrics plots
    avg_metric_fig = plot_avg_metrics(summarized_df, plot_trials=False)
    # Avg metric fig with lengths is the one that was summarized across replicates. We're left with average points per length.
    avg_metric_fig_lengths = plot_avg_metrics(summarized_df_across_replicates, plot_trials=True)
    # Avg metric fig with lengths is the one that was summarized across sequence lengths. We're left with average points per replicate.
    avg_metric_fig_replicates = plot_avg_metrics(summarized_df_across_lengths, plot_trials=True)
    
    avg_metric_fig.savefig(f'{csv_dir}/figures/averages_only/avg_metrics.png')
    avg_metric_fig_replicates.savefig(f'{csv_dir}/figures/individual_points_plotted/avg_metrics_with_replicate_avgs.png')
    avg_metric_fig_lengths.savefig(f'{csv_dir}/figures/individual_points_plotted/avg_metrics_with_seqlen_avgs.png')
    
    # Make and save the metrics by length plots - first, without individual results / length / metric
    individual_figs, combined_metrics_fig = plot_metrics_by_length(summarized_df_across_replicates, plot_trials=False)
    # Now, make and save the metrics by length plots WITH individaul results / length / metric. Need the original file, not the summarized, for this.
    df['model'] = df['model'].apply(map_model_names)
    individual_figs_with_reps, combined_metrics_fig_with_reps = plot_metrics_by_length(df, plot_trials=True)
    combined_metrics_fig.savefig((f'{csv_dir}/figures/averages_only/metrics_grid.png'))
    combined_metrics_fig_with_reps.savefig((f'{csv_dir}/figures/individual_points_plotted/metrics_grid_across_replicates.png'))
    # Save
    for plotname, fig in individual_figs.items():
        fig.savefig((f'{csv_dir}/figures/averages_only/{plotname}'))
    for plotname, fig in individual_figs_with_reps.items():
        fig.savefig((f'{csv_dir}/figures/individual_points_plotted/{plotname}'))

def main():
    # Set the path to your CSV file here
    csv_dir = config.RESULTS_DIR
    make_all_tables_and_plots(csv_dir)
    
if __name__ == "__main__":
    main()
