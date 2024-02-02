import sys
import os
import zipfile
import re
import pandas as pd
import gradio as gr
import gzip
import pickle
import numpy as np
from bertopic import BERTopic
from datetime import datetime

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# Log terminal output: https://github.com/gradio-app/gradio/issues/2362
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return False    

sys.stdout = Logger("output.log")

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()


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
    elif filename.endswith('.pkl'):
        return 'pkl'
    elif filename.endswith('.npz'):
        return 'npz'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
        
    print("Loading in file")

    if file_type == 'csv':
        file = pd.read_csv(filename, low_memory=False)#.reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'xlsx':
        file = pd.read_excel(filename)#.reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'parquet':
        file = pd.read_parquet(filename)#.reset_index().drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'pkl.gz':
        with gzip.open(filename, 'rb') as file:
            file = pickle.load(file)
            #file = pd.read_pickle(filename)
    elif file_type == 'pkl':
        file = BERTopic.load(filename)
    elif file_type == 'npz':
        file = np.load(filename)['arr_0']

        # If embedding files have 'super_compress' in the title, they have been multiplied by 100 before save
        if "compress" in filename:
            file /= 100

    print("File load complete")

    return file

def initial_file_load(in_file):
    '''
    When file is loaded, update the column dropdown choices and write to relevant data states.
    '''
    new_choices = []
    concat_choices = []
    custom_labels = pd.DataFrame()
    topic_model = None
    embeddings = np.array([])

    file_list = [string.name for string in in_file]

    data_file_names = [string.lower() for string in file_list if "npz" not in string.lower() and "pkl" not in string.lower() and "topic_list.csv" not in string.lower()]
    if data_file_names:
        data_file_name = data_file_names[0]
        df = read_file(data_file_name)
        data_file_name_no_ext = get_file_path_end(data_file_name)

        new_choices = list(df.columns)
        concat_choices.extend(new_choices)
        output_text = "Data file loaded."
    else:
        error = "No data file provided."
        print(error)
        output_text = error

    model_file_names = [string.lower() for string in file_list if "pkl" in string.lower()]
    if model_file_names:
        model_file_name = model_file_names[0]
        topic_model = read_file(model_file_name)
        output_text = "Bertopic model loaded."

    embedding_file_names = [string.lower() for string in file_list if "npz" in string.lower()]
    if embedding_file_names:
        embedding_file_name = embedding_file_names[0]
        embeddings = read_file(embedding_file_name)
        output_text = "Embeddings loaded."

    label_file_names = [string.lower() for string in file_list if "topic_list" in string.lower()]
    if label_file_names:
        label_file_name = label_file_names[0]
        custom_labels = read_file(label_file_name)
        output_text = "Labels loaded." 
   
        
    #The np.array([]) at the end is for clearing the embedding state when a new file is loaded
    return gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), df, output_text, topic_model, embeddings, data_file_name_no_ext, custom_labels

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


def save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model, progress=gr.Progress()):
        
        progress(0.7, desc= "Checking data")

        topic_dets = topic_model.get_topic_info()

        if topic_dets.shape[0] == 1:
            topic_det_output_name = "topic_details_" + data_file_name_no_ext + "_" + today_rev + ".csv"
            topic_dets.to_csv(topic_det_output_name)
            output_list.append(topic_det_output_name)

            return output_list, "No topics found, original file returned"

        
        progress(0.8, desc= "Saving output")
        
        topic_det_output_name = "topic_details_" + data_file_name_no_ext + "_" + today_rev + ".csv"
        topic_dets.to_csv(topic_det_output_name)
        output_list.append(topic_det_output_name)

        doc_det_output_name = "doc_details_" + data_file_name_no_ext + "_" + today_rev + ".csv"
        doc_dets = topic_model.get_document_info(docs)[["Document",	"Topic", "Name", "Probability", "Representative_document"]]
        doc_dets.to_csv(doc_det_output_name)
        output_list.append(doc_det_output_name)

        if "CustomName" in topic_dets.columns:
            topics_text_out_str = str(topic_dets["CustomName"])
        else:
            topics_text_out_str = str(topic_dets["Name"])
        output_text = "Topics: " + topics_text_out_str
    
        # Save topic model to file
        if save_topic_model == "Yes":
            print("Saving BERTopic model in .pkl format.")

            folder_path = "output_model/"

            if not os.path.exists(folder_path):
                # Create the folder
                os.makedirs(folder_path)

            topic_model_save_name_pkl = folder_path + data_file_name_no_ext + "_topics_" + today_rev + ".pkl"# + ".safetensors"
            topic_model_save_name_zip = topic_model_save_name_pkl + ".zip"

            # Clear folder before replacing files
            #delete_files_in_folder(topic_model_save_name_pkl)

            topic_model.save(topic_model_save_name_pkl, serialization='pickle', save_embedding_model=False, save_ctfidf=False)

            # Zip file example
            
            #zip_folder(topic_model_save_name_pkl, topic_model_save_name_zip)
            output_list.append(topic_model_save_name_pkl)

        return output_list, output_text
