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

from typing import List, Tuple

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

def get_or_create_env_var(var_name:str, default_value:str) -> str:
    # Get the environment variable if it exists
    value = os.environ.get(var_name)
    
    # If it doesn't exist, set it to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value
    
    return value

# Retrieving or setting output folder
env_var_name = 'GRADIO_OUTPUT_FOLDER'
default_value = 'output/'

output_folder = get_or_create_env_var(env_var_name, default_value)
print(f'The value of {env_var_name} is {output_folder}')

def ensure_output_folder_exists():
    """Checks if the 'output/' folder exists, creates it if not."""

    folder_name = "output/"

    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        print(f"Created the 'output/' folder.")
    else:
        print(f"The 'output/' folder already exists.")

def get_connection_params(request: gr.Request):
    '''
    Get connection parameter values from request object.
    '''
    if request:

        # print("Request headers dictionary:", request.headers)
        # print("All host elements", request.client)           
        # print("IP address:", request.client.host)
        # print("Query parameters:", dict(request.query_params))
        print("Session hash:", request.session_hash)

        if 'x-cognito-id' in request.headers:
            out_session_hash = request.headers['x-cognito-id']
            base_folder = "user-files/"
            #print("Cognito ID found:", out_session_hash)

        else:
            out_session_hash = request.session_hash
            base_folder = "temp-files/"
            #print("Cognito ID not found. Using session hash as save folder.")

        output_folder = base_folder + out_session_hash + "/"
        #print("S3 output folder is: " + "s3://" + bucket_name + "/" + output_folder)

        return out_session_hash
    else:
        print("No session parameters found.")
        return ""

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

    data_file_names = [string for string in file_list if "npz" not in string.lower() and "pkl" not in string.lower() and "topic_list.csv" not in string.lower()]
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

    model_file_names = [string for string in file_list if "pkl" in string.lower()]
    if model_file_names:
        model_file_name = model_file_names[0]
        topic_model = read_file(model_file_name)
        output_text = "Bertopic model loaded."

    embedding_file_names = [string for string in file_list if "npz" in string.lower()]
    if embedding_file_names:
        embedding_file_name = embedding_file_names[0]
        embeddings = read_file(embedding_file_name)
        output_text = "Embeddings loaded."

    label_file_names = [string for string in file_list if "topic_list" in string.lower()]
    if label_file_names:
        label_file_name = label_file_names[0]
        custom_labels = read_file(label_file_name)
        output_text = "Labels loaded." 
   
        
    #The np.array([]) at the end is for clearing the embedding state when a new file is loaded
    return gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), df, output_text, topic_model, embeddings, data_file_name_no_ext, custom_labels, df

def custom_regex_load(in_file):
    '''
    When file is loaded, update the column dropdown choices and write to relevant data states.
    '''

    custom_regex = pd.DataFrame()

    file_list = [string.name for string in in_file]

    regex_file_names = [string for string in file_list if "csv" in string.lower()]
    if regex_file_names:
        regex_file_name = regex_file_names[0]
        custom_regex = pd.read_csv(regex_file_name, low_memory=False, header=None)
        #regex_file_name_no_ext = get_file_path_end(regex_file_name)

        output_text = "Data file loaded."
        print(output_text)
    else:
        error = "No regex file provided."
        print(error)
        output_text = error
        return error, custom_regex
       
    return output_text, custom_regex

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

def save_topic_outputs(topic_model: BERTopic, data_file_name_no_ext: str, output_list: List[str], docs: List[str], save_topic_model: bool, prepared_docs: pd.DataFrame, split_sentence_drop: str, output_folder: str = output_folder, progress: gr.Progress = gr.Progress()) -> Tuple[List[str], str]:
    """
    Save the outputs of a topic model to specified files.

    Args:
        topic_model (BERTopic): The topic model object.
        data_file_name_no_ext (str): The base name of the data file without extension.
        output_list (List[str]): List to store the output file names.
        docs (List[str]): List of documents.
        save_topic_model (bool): Flag to save the topic model.
        prepared_docs (pd.DataFrame): DataFrame containing prepared documents.
        split_sentence_drop (str): Option to split sentences ("Yes" or "No").
        output_folder (str, optional): Folder to save the output files. Defaults to output_folder.
        progress (gr.Progress, optional): Progress tracker. Defaults to gr.Progress().

    Returns:
        Tuple[List[str], str]: A tuple containing the list of output file names and a status message.
    """
        
    progress(0.7, desc= "Checking data")

    topic_dets = topic_model.get_topic_info()

    if topic_dets.shape[0] == 1:
        topic_det_output_name = output_folder + "topic_details_" + data_file_name_no_ext + "_" + today_rev + ".csv"
        topic_dets.to_csv(topic_det_output_name)
        output_list.append(topic_det_output_name)

        return output_list, "No topics found, original file returned"

    progress(0.8, desc= "Saving output")
    
    topic_det_output_name = output_folder + "topic_details_" + data_file_name_no_ext + "_" + today_rev + ".csv"
    topic_dets.to_csv(topic_det_output_name)
    output_list.append(topic_det_output_name)

    doc_det_output_name = output_folder + "doc_details_" + data_file_name_no_ext + "_" + today_rev + ".csv"

    ## Check that the following columns exist in the dataframe, keep only the ones that exist
    columns_to_check = ["Document",	"Topic", "Name", "Probability", "Representative_document"]

    columns_found = [column for column in columns_to_check if column in topic_model.get_document_info(docs).columns]
    doc_dets = topic_model.get_document_info(docs)[columns_found]

    # If you have created a 'sentence split' dataset from the cleaning options, map these sentences back to the original document.
    try:
        if split_sentence_drop == "Yes":
            doc_dets = doc_dets.merge(prepared_docs[['document_index']], how = "left", left_index=True, right_index=True)
            doc_dets = doc_dets.rename(columns={"document_index": "parent_document_index"}, errors='ignore')

            # 1. Group by Parent Document Index:
            grouped = doc_dets.groupby('parent_document_index')

            # 2. Aggregate Topics and Probabilities:
            def aggregate_topics(group):
                original_text = ' '.join(group['Document'])
                topics = group['Topic'].tolist()

                if 'Name' in group.columns:
                    topic_names = group['Name'].tolist()
                else:
                    topic_names = None

                if 'Probability' in group.columns:
                    probabilities = group['Probability'].tolist()
                else:
                    probabilities = None  # Or any other default value you prefer

                return pd.Series({'Document':original_text, 'Topic numbers': topics, 'Topic names': topic_names, 'Probabilities': probabilities})

            #result_df = grouped.apply(aggregate_topics).reset_index()
            doc_det_agg = grouped.apply(lambda x: aggregate_topics(x)).reset_index()

            # Join back original text
            #doc_det_agg = doc_det_agg.merge(original_data[[in_colnames_list_first]], how = "left", left_index=True, right_index=True)

            doc_det_agg_output_name = output_folder + "doc_details_agg_" + data_file_name_no_ext + "_" + today_rev + ".csv"
            doc_det_agg.to_csv(doc_det_agg_output_name)
            output_list.append(doc_det_agg_output_name)

    except Exception as e:
        print("Creating aggregate document details failed, error:", e)

    # Save document details to file
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

        #folder_path = output_folder #"output_model/"

        #if not os.path.exists(folder_path):
            # Create the folder
        #    os.makedirs(folder_path)

        topic_model_save_name_pkl = output_folder + data_file_name_no_ext + "_topics_" + today_rev + ".pkl"# + ".safetensors"
        topic_model_save_name_zip = topic_model_save_name_pkl + ".zip"

        # Clear folder before replacing files
        #delete_files_in_folder(topic_model_save_name_pkl)

        topic_model.save(topic_model_save_name_pkl, serialization='pickle', save_embedding_model=False, save_ctfidf=False)

        # Zip file example
        
        #zip_folder(topic_model_save_name_pkl, topic_model_save_name_zip)
        output_list.append(topic_model_save_name_pkl)

    return output_list, output_text
