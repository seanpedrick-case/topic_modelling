import re
import string
import polars as pl
import gradio as gr

# Adding custom words to the stopwords
custom_words = []
my_stop_words = custom_words

# #### Some of my cleaning functions
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|(?:www\.)[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}'
html_pattern_regex = r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|\xa0|&nbsp;'
html_start_pattern_end_dots_regex = r'<(.*?)\.\.'
non_ascii_pattern = r'[^\x00-\x7F]+'
email_pattern_regex = r'\S*@\S*\s?'
num_pattern_regex = r'[0-9]+'
nums_two_more_regex = r'\b\d+[\.|\,]\d+\b|\b[0-9]{2,}\b|\b[0-9]+\s[0-9]+\b' # Should match two digit numbers or more, and also if there are full stops or commas in between
postcode_pattern_regex = r'(\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2})|((GIR ?0A{2})\b$)|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$)|(\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\b$)'
multiple_spaces_regex = r'\s{2,}'

def initial_clean(texts, custom_regex, progress=gr.Progress()):
    # Convert to polars Series
    texts = pl.Series(texts).str.strip_chars()
    
    # Define a list of patterns and their replacements
    patterns = [
        (url_pattern, ' '),
        (html_pattern_regex, ' '),
        (html_start_pattern_end_dots_regex, ' '),
        (non_ascii_pattern, ' '),
        (email_pattern_regex, ' '),
        (nums_two_more_regex, ' '),
        (postcode_pattern_regex, ' '),
        (multiple_spaces_regex, ' ')
    ]
    
    # Apply each regex replacement
    for pattern, replacement in patterns:
        texts = texts.str.replace_all(pattern, replacement)
    
    # Convert the series back to a list
    texts = texts.to_list()
    
    return texts

def regex_clean(texts, custom_regex, progress=gr.Progress()):
    texts = pl.Series(texts).str.strip_chars()

    # Allow for custom regex patterns to be removed
    if len(custom_regex) > 0:
        for pattern in custom_regex:
            raw_string_pattern = r'{}'.format(pattern)
            print("Removing regex pattern: ", raw_string_pattern)
            texts = texts.str.replace_all(raw_string_pattern, ' ')

    texts = texts.str.replace_all(multiple_spaces_regex, ' ')

    texts = texts.to_list()
    
    return texts

def remove_hyphens(text_text):
    return re.sub(r'(\w+)-(\w+)-?(\w)?', r'\1 \2 \3', text_text)


def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens

def convert_to_lowercase(tokens):
    return [token.lower() for token in tokens if token.isalpha()]

def remove_short_tokens(tokens):
    return [token for token in tokens if len(token) > 3]


def remove_dups_text(data_samples_ready, data_samples_clean, data_samples):
   # Identify duplicates in the data: https://stackoverflow.com/questions/44191465/efficiently-identify-duplicates-in-large-list-500-000
    # Only identifies the second duplicate

    seen = set()
    dups = []

    for i, doi in enumerate(data_samples_ready):
        if doi not in seen:
            seen.add(doi)
        else:
            dups.append(i) 
    #data_samples_ready[dupes[0:]]
    
    # To see a specific duplicated value you know the position of
    #matching = [s for s in data_samples_ready if data_samples_ready[83] in s]
    #matching
    
    # Remove duplicates only (keep first instance)
    #data_samples_ready = list( dict.fromkeys(data_samples_ready) ) # This way would keep one version of the duplicates
    
    ### Remove all duplicates including original instance
    
    # Identify ALL duplicates including initial values
    # https://stackoverflow.com/questions/11236006/identify-duplicate-values-in-a-list-in-python

    from collections import defaultdict
    D = defaultdict(list)
    for i,item in enumerate(data_samples_ready):
        D[item].append(i)
    D = {k:v for k,v in D.items() if len(v)>1}
    
    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    L = list(D.values())
    flat_list_dups = [item for sublist in L for item in sublist]

    # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time
    for index in sorted(flat_list_dups, reverse=True):
        del data_samples_ready[index]
        del data_samples_clean[index]
        del data_samples[index]
    
    # Remove blanks
    data_samples_ready = [i for i in data_samples_ready if i]
    data_samples_clean = [i for i in data_samples_clean if i]
    data_samples = [i for i in data_samples if i]
    
    return data_samples_ready, data_samples_clean, flat_list_dups, data_samples

