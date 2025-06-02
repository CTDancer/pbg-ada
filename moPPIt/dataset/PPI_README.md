# 1. Run comtamination.py to get initial results and error sequences.
`python -u contamination.py -i 1,2,3,4`

# 2. Run extract_full_sequence.py to get full sequences for error sequences. extract_full_sequence.py can only run for one id at a time.
`python -u extract_full_sequence.py -id 1`

# 3. After getting full sequences for error sequences, run final_contamination.py to get the final results.
`python -u final_contamination.py -i 1,2,3,4`