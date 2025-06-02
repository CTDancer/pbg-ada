# Summarizes the test statistics
import pandas as pd
import numpy as np
import os
import protein_lm.evaluation.disease_druggable_benchmark.config as config
            
def make_table(results_dir='', table_name='',onehot_path='',onehotptm_path='',esm2_15b_path='',esm2_3b_path='',esm2_650m_path='',esm2_150m_path='',esm2_35m_path='',esm2_8m_path='',mamba_path='',ptm_transformer_path='',mamba_saprot_path=''):
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

    full_table = pd.DataFrame()
    # Read each result file and plot the metrics
    for method, path in method_results.items():
        df = pd.read_csv(path)
        df = df.rename(columns={'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'mcc': 'MCC', 'auprc': 'AUPRC', 'auroc': 'AUROC'})
        
        # group by run and get means across all lengths
        df = df.groupby('run').agg(
                {'Accuracy': 'mean', 'Precision': 'mean', 'Recall': 'mean', 'F1': 'mean', 'MCC': 'mean', 'AUPRC': 'mean', 'AUROC': 'mean'}
            ).reset_index()

        # get the mean of each metric, and the standard deviation between runs
        stds = pd.DataFrame(df.std()).transpose().drop(columns='run').rename(
            columns={
                    'Accuracy': 'Accuracy SD',
                    'Precision': 'Precision SD',
                    'Recall': 'Recall SD',
                    'F1': 'F1 SD',
                    'MCC': 'MCC SD',
                    'AUPRC': 'AUPRC SD',
                    'AUROC': 'AUROC SD'
                }
        )
        means = pd.DataFrame(df.mean()).transpose().drop(columns='run')
        # put these together 
        means = means.join(stds)
        means['Model'] = [method]
        means = means[['Model',
            'Accuracy', 'Accuracy SD',
            'Precision', 'Precision SD',
            'Recall', 'Recall SD',
            'F1', 'F1 SD',
            'MCC', 'MCC SD',
            'AUPRC', 'AUPRC SD',
            'AUROC', 'AUROC SD'
        ]]
        # Round to 4 decimal places
        for x in list(means.columns)[1::]:
            means[x] = means[x].round(4)
        
        full_table = pd.concat([full_table, means])
    
    # save the full table
    if not os.path.exists(f'{results_dir}/summary'): os.mkdir(f'{results_dir}/summary')
    full_table.to_csv(f'{results_dir}/summary/{table_name}',index=False)
    
def main():
    druggable_prefix = 'druggability_test_results'
    # Define kwargs from config file
    druggable_kwargs = {
        'results_dir': config.DRUGGABILITY_RESULTS_DIR if config.DRUGGABILITY_RESULTS_DIR not in [None,''] else os.getcwd(),
        'table_name': 'druggable_summary.csv',
    }
    # Define potential paths
    druggable_paths={
        'onehot_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_onehot.csv',
        'onehotptm_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_onehotptm.csv',
        'esm2_15b_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_esm2_t48_15B_UR50D.csv',
        'esm2_3b_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_esm2_t36_3B_UR50D.csv',
        'esm2_650m_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_esm2_t33_650M_UR50D.csv',
        'esm2_150m_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_esm2_t30_150M_UR50D.csv',
        'esm2_35m_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_esm2_t12_35M_UR50D.csv',
        'esm2_8m_path':  druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_esm2_t6_8M_UR50D.csv',
        'mamba_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_ptm_mamba.csv',
        'ptm_transformer_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_ptm_transformer.csv',
        'mamba_saprot_path': druggable_kwargs['results_dir']+ f'/test_metrics/{druggable_prefix}_mamba_saprot.csv',
       
        }
    # Remove any paths that don't exist
    druggable_paths = {key: value for key, value in druggable_paths.items() if os.path.exists(value)}
    druggable_kwargs.update(druggable_paths)
    
    # Repeat for disease
    disease_prefix = 'disease_test_results'
    # Define kwargs from config file
    disease_kwargs = {
        'results_dir': config.DISEASE_RESULTS_DIR if config.DISEASE_RESULTS_DIR not in [None,''] else os.getcwd(),
        'table_name': 'disease_summary.csv',
    }
    # Define potential paths
    disease_paths={
        'onehot_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_onehot.csv',
        'onehotptm_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_onehotptm.csv',
        'esm2_15b_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_esm2_t48_15B_UR50D.csv',
        'esm2_3b_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_esm2_t36_3B_UR50D.csv',
        'esm2_650m_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_esm2_t33_650M_UR50D.csv',
        'esm2_150m_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_esm2_t30_150M_UR50D.csv',
        'esm2_35m_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_esm2_t12_35M_UR50D.csv',
        'esm2_8m_path':  disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_esm2_t6_8M_UR50D.csv',
        'mamba_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_ptm_mamba.csv',
        'ptm_transformer_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_ptm_transformer.csv',
        'mamba_saprot_path': disease_kwargs['results_dir']+ f'/test_metrics/{disease_prefix}_mamba_saprot.csv',
        }
    # Remove any paths that don't exist
    disease_paths = {key: value for key, value in disease_paths.items() if os.path.exists(value)}
    disease_kwargs.update(disease_paths)
    
    make_table(**druggable_kwargs)
    make_table(**disease_kwargs)

if __name__ == "__main__":
    main()