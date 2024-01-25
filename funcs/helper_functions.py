import os
import zipfile
import re
import pandas as pd
import gradio as gr
import gzip
import pickle
import numpy as np


def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    elif filename.endswith('.pkl.gz'):
        return 'pkl.gz'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
        
    print("Loading in file")

    if file_type == 'csv':
        file = pd.read_csv(filename, low_memory=False).reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'xlsx':
        file = pd.read_excel(filename).reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'parquet':
        file = pd.read_parquet(filename).reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'pkl.gz':
        with gzip.open(filename, 'rb') as file:
            file = pickle.load(file)
            #file = pd.read_pickle(filename)

    print("File load complete")

    return file

def put_columns_in_df(in_file, in_bm25_column):
    '''
    When file is loaded, update the column dropdown choices and change 'clean data' dropdown option to 'no'.
    '''

    file_list = [string.name for string in in_file]

    data_file_names = [string.lower() for string in file_list if "npz" not in string.lower()]
    data_file_name = data_file_names[0]


    new_choices = []
    concat_choices = []
    
    
    df = read_file(data_file_name)

    new_choices = list(df.columns)


    concat_choices.extend(new_choices)     
    
    #The np.array([]) at the end is for clearing the embedding state when a new file is loaded
    return gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), df, np.array([])

def get_file_path_end(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    #print(filename_without_extension)
    
    return filename_without_extension

def get_file_path_end_with_ext(file_path):
    match = re.search(r'(.*[\/\\])?(.+)$', file_path)
        
    filename_end = match.group(2) if match else ''

    return filename_end

def dummy_function(in_colnames):
    """
    A dummy function that exists just so that dropdown updates work correctly.
    """
    return None

# Zip the above to export file


def zip_folder(folder_path, output_zip_file):
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create a complete file path
                file_path = os.path.join(root, file)
                # Add file to the zip file
                # The arcname argument sets the archive name, i.e., the name within the zip file
                zipf.write(file_path, arcname=os.path.relpath(file_path, folder_path))

def delete_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over all files in the folder and remove each
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            else:
                print(f"Skipping {file_path} as it is a directory")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")