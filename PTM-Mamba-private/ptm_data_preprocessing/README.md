# Construction of PTM sequences from Uniprot-Swissprot


## Code Structure


- `extract_ptm_labels.py` 
    - Download and extract all the Uniprot-Swissprot labels.
- `label_preprocess.py`
    - Preprocess the labels to remove the non-PTM labels.

- `analysis.ipynb` 
    - Analysis of the PTM label distribution.


## Future Work
- Expand the data sources to include the Uniprot-trembl database, which contains a larger number of unreviewed annotations.
- Include all the labels in the database, not just the PTM labels.


