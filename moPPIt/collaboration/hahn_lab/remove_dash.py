def remove_dashes_and_flatten(text):
    # Remove all '-' characters and join the lines into one
    cleaned_text = text.replace('-', '').replace('\n', '')
    return cleaned_text

# Example multiline string
input_string = """
MAAIRKKLVIVGDGACGKTCLLIVFSKDQFPEVYVPTVFENYVADIEVDGKQVELALWDTAGQEDYDRLRPLSYPDTDVI
LMCFSIDSPDSLENIPEKWTPEVKH--FCPNVPIILVGNKKDLRNDEHTRRELAKMKQEPVKPEEGRDMANRIGAFGYME
CSAKTKDGVREVFEMATRAALQARRG-KKKS---------------GCLVL
"""

# Process the string
result = remove_dashes_and_flatten(input_string)

# Print the result
print(result)
