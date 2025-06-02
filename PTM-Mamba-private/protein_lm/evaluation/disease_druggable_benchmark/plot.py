import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import protein_lm.evaluation.disease_druggable_benchmark.config as config

global color_map
color_map = {
    'OneHot': '#207fc2',
    'OneHot-PTM': "#00A5E3",
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
    
def check_label_consistency(df):
    """
    Check whether the "label" column has the same labels for all runs. It should, because the test dataloader was not shuffled.
    Raises an error if inconsistent.  
    """
    # Group the DataFrame by 'run'
    grouped = df.groupby('run')
    
    # Get the labels of the first group to compare against
    reference_labels = None
    
    for name, group in grouped:
        # Get the labels of the current group
        current_labels = group['label'].values
        
        if reference_labels is None:
            # Set the first group's labels as the reference
            reference_labels = current_labels
        else:
            # Check if current group's labels match the reference
            if not np.array_equal(reference_labels, current_labels):
                raise ValueError(f"Labels are not consistent across runs. Run {name} has different labels.")
            
# Plot Precision-Recall curve
def make_pr_curve(results_dir='', task_name='',onehot_path='',onehotptm_path='',esm2_15b_path='',esm2_3b_path='',esm2_650m_path='',esm2_150m_path='',esm2_35m_path='',esm2_8m_path='',mamba_path='',ptm_transformer_path='',mamba_saprot_path=''):
    method_results = {'OneHot': onehot_path,
                    'OneHot-PTM': onehotptm_path,
                    'ESM-2-15B': esm2_15b_path,
                    'ESM-2-3B': esm2_3b_path,
                    'ESM-2-650M': esm2_650m_path,
                    'ESM-2-150M': esm2_150m_path,
                    'ESM-2-35M': esm2_35m_path,
                    'ESM-2-8M': esm2_8m_path,
                    'PTM-Mamba': mamba_path,
                    'PTM-Transformer': ptm_transformer_path,
                    'PTM-Mamba-SaProt': mamba_saprot_path       
                }
    method_results = {k:v for k,v in method_results.items() if v not in [None, '']}
    
    set_style()
    plt.figure()
    # Read each result file and plot the metrics
    for method, path in method_results.items():
        df = pd.read_csv(path) # columns = prob_0,prob_1,label,run
        check_label_consistency(df)
        
        # Label each sample so we can take the average for each test sample across runs
        df['sample'] = list(np.arange(1, 1+len(df.loc[df['run']==1]))) * len(df['run'].unique())
        df = df.groupby('sample').agg({'prob_0': 'mean', 'prob_1': 'mean', 'label': 'mean'})
        
        # Extract probabilities and labels
        prob_1 = df['prob_1'].values
        labels = df['label'].values

        # Compute Precision-Recall curve and average precision
        precision, recall, thresholds = precision_recall_curve(labels, prob_1)
        average_precision = average_precision_score(labels, prob_1)
        
        plt.plot(recall, precision, color=color_map[method], lw=2, label=f'{method} ({average_precision:0.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
    
    plt.show()
    plt.savefig(f'{results_dir}/{task_name}_PR_curve.png')
     
# Plot AUROC curve
def make_auroc_curve(results_dir='', task_name='',onehot_path='',onehotptm_path='',esm2_15b_path='',esm2_3b_path='',esm2_650m_path='',esm2_150m_path='',esm2_35m_path='',esm2_8m_path='',mamba_path='',ptm_transformer_path='',mamba_saprot_path=''):
    method_results = {'OneHot': onehot_path,
                    'OneHot-PTM': onehotptm_path,
                    'ESM-2-15B': esm2_15b_path,
                    'ESM-2-3B': esm2_3b_path,
                    'ESM-2-650M': esm2_650m_path,
                    'ESM-2-150M': esm2_150m_path,
                    'ESM-2-35M': esm2_35m_path,
                    'ESM-2-8M': esm2_8m_path,
                    'PTM-Mamba': mamba_path,
                    'PTM-Transformer': ptm_transformer_path,
                    'PTM-Mamba-SaProt': mamba_saprot_path      
                }
    method_results = {k:v for k,v in method_results.items() if v not in [None, '']}
    
    set_style()
    plt.figure()
    # Read each result file and plot the metrics
    for method, path in method_results.items():
        df = pd.read_csv(path) # columns = prob_0,prob_1,label,run
        check_label_consistency(df)
        
        # Label each sample so we can take the average for each test sample across runs
        df['sample'] = list(np.arange(1, 1+len(df.loc[df['run']==1]))) * len(df['run'].unique())
        df = df.groupby('sample').agg({'prob_0': 'mean', 'prob_1': 'mean', 'label': 'mean'})
        
        # Extract probabilities and labels
        prob_1 = df['prob_1'].values
        labels = df['label'].values

        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(labels, prob_1)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color_map[method], lw=2, label=f'{method} ({roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='darkgrey', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
    plt.show()
    plt.savefig(f'{results_dir}/{task_name}_AUROC_curve.png')

def main():
    druggable_prefix = 'druggability_test_probs_and_labels'
    # Define kwargs from config file
    druggable_kwargs = {
        'results_dir': config.DRUGGABILITY_RESULTS_DIR if config.DRUGGABILITY_RESULTS_DIR not in [None,''] else os.getcwd(),
        'task_name': 'druggability',
    }
    # Define potential paths
    druggable_paths={
        'onehot_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_onehot.csv',
        'onehotptm_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_onehotptm.csv',
        'esm2_15b_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_esm2_t48_15B_UR50D.csv',
        'esm2_3b_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_esm2_t36_3B_UR50D.csv',
        'esm2_650m_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_esm2_t33_650M_UR50D.csv',
        'esm2_150m_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_esm2_t30_150M_UR50D.csv',
        'esm2_35m_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_esm2_t12_35M_UR50D.csv',
        'esm2_8m_path':  druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_esm2_t6_8M_UR50D.csv',
        'mamba_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_ptm_mamba.csv',
        'ptm_transformer_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_ptm_transformer.csv',
        'mamba_saprot_path': druggable_kwargs['results_dir']+ f'/test_probabilities/{druggable_prefix}_mamba_saprot.csv',
        }
    # Remove any paths that don't exist
    druggable_paths = {key: value for key, value in druggable_paths.items() if os.path.exists(value)}
    druggable_kwargs.update(druggable_paths)
    
    # Repeat for disease
    disease_prefix = 'disease_test_probs_and_labels'
    # Define kwargs from config file
    disease_kwargs = {
        'results_dir': config.DISEASE_RESULTS_DIR if config.DISEASE_RESULTS_DIR not in [None,''] else os.getcwd(),
        'task_name': 'disease',
    }
    # Define potential paths
    disease_paths={
        'onehot_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_onehot.csv',
        'onehotptm_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_onehotptm.csv',
        'esm2_15b_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_esm2_t48_15B_UR50D.csv',
        'esm2_3b_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_esm2_t36_3B_UR50D.csv',
        'esm2_650m_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_esm2_t33_650M_UR50D.csv',
        'esm2_150m_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_esm2_t30_150M_UR50D.csv',
        'esm2_35m_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_esm2_t12_35M_UR50D.csv',
        'esm2_8m_path':  disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_esm2_t6_8M_UR50D.csv',
        'mamba_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_ptm_mamba.csv',
        'ptm_transformer_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_ptm_transformer.csv',
        'mamba_saprot_path': disease_kwargs['results_dir']+ f'/test_probabilities/{disease_prefix}_mamba_saprot.csv',
        }
    # Remove any paths that don't exist
    disease_paths = {key: value for key, value in disease_paths.items() if os.path.exists(value)}
    disease_kwargs.update(disease_paths)
    
    make_pr_curve(**druggable_kwargs)
    make_pr_curve(**disease_kwargs)
    make_auroc_curve(**druggable_kwargs)
    make_auroc_curve(**disease_kwargs)
    
if __name__ == "__main__":
    main()
