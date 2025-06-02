import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import protein_lm.evaluation.PhosphositePTM.config as config

# Define the metrics to plot and color map for each method
global metrics_to_plot, color_map 
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUPRC', 'AUROC']
color_map = {
    'OneHot-WT': '#207fc2',
    'ESM-2-650M': '#fecc00',
    'ESM-2-3B': '#EC5800',
    'PTM-Mamba': '#008000',
    'PTM-Transformer': "#AC71F3",
    'PTM-Mamba-SaProt': "#7220D6",
}

# Function to extract numeric value from a tensor string
def extract_numeric(value):
    if isinstance(value, str) and value.startswith('tensor('):
        return float(value[7:-1])
    return value

# Plot formatting
def plot_barchart(plot_save_dir='', onehot_path='', esm2_650m_path='', esm2_3b_path='', mamba_path='',ptm_transformer_path='',mamba_saprot_path=''):
    """
    Args: 
        method_results: dictionary of method names and the paths to their result files
    """
    # Set the seaborn style and font scale for publication
    plot_settings = {'ytick.labelsize': 16,
                    'xtick.labelsize': 16,
                    'font.size': 22,
                    'figure.figsize': (8, 5),
                    'axes.titlesize': 22,
                    'axes.labelsize': 18,
                    'lines.linewidth': 2,
                    'lines.markersize': 3,
                    'legend.fontsize': 11,
                    'mathtext.fontset': 'stix',
                    'font.family': 'STIXGeneral'}
    plt.style.use(plot_settings)
    sns.set_context("paper", rc={"font.size":16,"axes.titlesize":16,"axes.labelsize":16})

    # Define method_results based on what is passed into the model
    method_results = {'OneHot-WT': onehot_path,
                    'ESM-2-650M': esm2_650m_path,
                    'ESM-2-3B': esm2_3b_path,
                    'PTM-Mamba': mamba_path,
                    'PTM-Transformer': ptm_transformer_path,
                    'PTM-Mamba-SaProt': mamba_saprot_path
            }
    method_results = {k:v for k,v in method_results.items() if v not in [None, '']}

    method_names = list(method_results.keys())

    # Initialize a dictionary to store the average metrics for each method
    average_metrics = {method: {} for method in method_names}

    # Read each result file and calculate the average metrics
    for method, path in method_results.items():
        df = pd.read_csv(path,index_col=0)
        # Rename the columns to match the metrics
        df.rename(columns={'accuracy': 'Accuracy', 
                           'precision': 'Precision', 
                           'recall': 'Recall', 
                           'f1': 'F1', 
                           'mcc': 'MCC', 
                           'auprc': 'AUPRC', 
                           'auroc': 'AUROC'}, inplace=True)

        for metric in metrics_to_plot:  # Only consider the specified metrics
            df[metric] = df[metric].apply(extract_numeric)  # Convert tensor strings to numeric values
            # Take the average by run (there are 5 runs, or replicates)
            df = df.groupby('run').agg(
                {'Accuracy': 'mean', 'Precision': 'mean', 'Recall': 'mean', 'F1': 'mean', 'MCC': 'mean', 'AUPRC': 'mean', 'AUROC': 'mean'}
            )
            # Store the average metrics and metrics/run for this method
            average_metrics[method][metric] = {
                'Avg': df[metric].mean(),
                'All': df[metric].values.tolist()
            }

    # Convert the dictionary to a DataFrame for easier plotting
    metrics_df = pd.DataFrame(average_metrics).reset_index().melt(id_vars='index', var_name='Method', value_name='Value')
    metrics_df[['Avg', 'All']] = metrics_df['Value'].apply(pd.Series)
    metrics_df = metrics_df.drop('Value', axis=1)
    metrics_df = metrics_df.explode('All').reset_index(drop=True)
    metrics_df.rename(columns={'index': 'Metric'}, inplace=True)
    metrics_df = metrics_df[metrics_df['Metric'].isin(metrics_to_plot)]  # Filter the DataFrame to include only the specified metrics
    
    # Save metrics_df
    metrics_df.to_csv(f'{plot_save_dir}/barchart_metrics.csv',index=False)

    # Create the plot
    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(x='Metric', y='Avg', hue='Method', data=metrics_df, errorbar=None, palette=color_map, dodge=True) # bars
    strip_plot = sns.stripplot(x='Metric', y='All', hue='Method', data=metrics_df, dodge=True, jitter=False, palette=color_map, size=4, marker='o', edgecolor='black', linewidth=0.5) # results for each run
    plt.xlabel('Metric')
    plt.ylabel('')
    # set a larger size for the x-axis labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Customize the legend to exclude the dots
    handles, labels = bar_plot.get_legend_handles_labels()
    plt.legend(handles[0:len(method_names)], labels[0:len(method_names)], bbox_to_anchor=(0.6, 1), loc='upper left',fontsize=16)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{plot_save_dir}/barchart.png')

def plot_linegraph(plot_save_dir='', onehot_path='', esm2_650m_path='', esm2_3b_path='', mamba_path='',ptm_transformer_path='',mamba_saprot_path=''):    
    # Define method_results based on what is passed into the model
    method_results = {'OneHot-WT': onehot_path,
                    'ESM-2-650M': esm2_650m_path,
                    'ESM-2-3B': esm2_3b_path,
                    'PTM-Mamba': mamba_path,
                    'PTM-Transformer': ptm_transformer_path,
                    'PTM-Mamba-SaProt': mamba_saprot_path
            }
    method_results = {k:v for k,v in method_results.items() if v not in [None, '']}

    method_names = list(method_results.keys())
    
    # Read each result file and plot the metrics
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        average_metrics = pd.DataFrame()
        for method, path in method_results.items():
            df = pd.read_csv(path)
            # Rename the columns to match the metrics
            df.rename(columns={'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'mcc': 'MCC', 'auprc': 'AUPRC', 'auroc': 'AUROC'}, inplace=True)
            df[metric] = df[metric].apply(extract_numeric)  # Convert tensor strings to numeric values
            #plt.plot(df['Length'], df[metric], label=method, color=color_map[method], marker='o')
            plt.scatter(df['Length'], df[metric], color=color_map[method], alpha=0.6,label='_nolegend_')

            # Calculate and plot the average line
            average_metric = df.groupby('Length')[metric].mean()
            plt.plot(average_metric.index, average_metric.values, label=f'{method} average', color=color_map[method], marker='o', linestyle='-')
            
            # Prepare averages csv for saving
            average_metric = average_metric.reset_index().rename(columns={metric:f'Avg {metric}'})
            average_metric['Model'] = [method]*len(average_metric)
            average_metrics = pd.concat([average_metrics,average_metric[['Model','Length',f'Avg {metric}']]])

        plt.xlabel('Length')
        plt.ylabel(metric)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Make a directory to store the plot if one doesn't already exist
        linegraphs_save_dir = f"{plot_save_dir}/linegraphs"
        if not(os.path.exists(linegraphs_save_dir)):
            os.mkdir(linegraphs_save_dir)
            print("made dir: ", linegraphs_save_dir)
        # Save
        plt.savefig(f"{linegraphs_save_dir}/{metric}_linegraph.png")
        average_metrics.to_csv(f"{linegraphs_save_dir}/{metric}_averages.csv",index=False)
    
if __name__ == "__main__":
    kwargs = {
    'plot_save_dir': config.PLOT_SAVE_DIR if config.PLOT_SAVE_DIR not in [None,''] else os.getcwd(),
    'onehot_path': config.ONEHOT_METRICS_PATH,
    'esm2_650m_path': config.ESM650M_METRICS_PATH,
    'esm2_3b_path': config.ESM3B_METRICS_PATH,
    'mamba_path': config.MAMBA_METRICS_PATH,
    'ptm_transformer_path': config.PTM_TRANSFORMER_METRICS_PATH,
    'mamba_saprot_path': config.MAMBA_SAPROT_METRICS_PATH,
    }

    # Filter out arguments where the value is None or ''
    filtered_kwargs = {key: value for key, value in kwargs.items() if value not in [None, '']}

    # Call the functions with filtered arguments
    plot_barchart(**filtered_kwargs)
    plot_linegraph(**filtered_kwargs)    