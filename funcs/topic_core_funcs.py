# Dendrograms will not work with the latest version of scipy (1.12.0), so installing the version prior to be safe
#os.system("pip install scipy==1.11.4")

import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
import time
from bertopic import BERTopic
import spaces

from typing import List, Type, Union
PandasDataFrame = Type[pd.DataFrame]

from funcs.clean_funcs import initial_clean, regex_clean
from funcs.anonymiser import expand_sentences_spacy
from funcs.helper_functions import read_file, zip_folder, delete_files_in_folder, save_topic_outputs, output_folder, get_or_create_env_var, custom_regex_load
from funcs.embeddings import make_or_load_embeddings, torch_device
from funcs.bertopic_vis_documents import visualize_documents_custom, visualize_hierarchical_documents_custom, hierarchical_topics_custom, visualize_hierarchy_custom
from funcs.representation_model import create_representation_model, llm_config, chosen_start_tag, random_seed, RUNNING_ON_AWS
from sklearn.feature_extraction.text import CountVectorizer
import funcs.anonymiser as anon
from umap import UMAP

# Default options can be changed in number selection on options page
umap_n_neighbours = 15
umap_min_dist = 0.0
umap_metric = 'cosine'

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# Load embeddings
if RUNNING_ON_AWS=="0":
    embeddings_name = "mixedbread-ai/mxbai-embed-xsmall-v1" #"mixedbread-ai/mxbai-embed-large-v1"
else:
    embeddings_name = "mixedbread-ai/mxbai-embed-xsmall-v1"

# LLM model used for representing topics
hf_model_name = "bartowski/Llama-3.2-3B-Instruct-GGUF" #"bartowski/Phi-3.1-mini-128k-instruct-GGUF"
hf_model_file = "Llama-3.2-3B-Instruct-Q5_K_M.gguf" #"Phi-3.1-mini-128k-instruct-Q4_K_M.gguf"

# When topic modelling column is chosen, change the default visualisation column to the same
def change_default_vis_col(in_colnames:List[str]):
    '''
    When topic modelling column is chosen, change the default visualisation column to the same
    '''
    if in_colnames:
        return gr.Dropdown(value=in_colnames[0])
    else:
        return gr.Dropdown()

def pre_clean(data: pd.DataFrame, in_colnames: list, data_file_name_no_ext: str, custom_regex: pd.DataFrame, clean_text: str, drop_duplicate_text: str, anonymise_drop: str, sentence_split_drop: str, min_sentence_length: int, embeddings_state: dict, output_folder: str = output_folder, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> tuple:
    """
    Pre-processes the input data by cleaning text, removing duplicates, anonymizing data, and splitting sentences based on the provided options.

    Args:
        data (pd.DataFrame): The input data to be cleaned.
        in_colnames (list): List of column names to be used for cleaning and finding topics.
        data_file_name_no_ext (str): The base name of the data file without extension.
        custom_regex (pd.DataFrame): Custom regex patterns for initial cleaning.
        clean_text (str): Option to clean text ("Yes" or "No").
        drop_duplicate_text (str): Option to drop duplicate text ("Yes" or "No").
        anonymise_drop (str): Option to anonymize data ("Yes" or "No").
        sentence_split_drop (str): Option to split text into sentences ("Yes" or "No").
        min_sentence_length (int): Minimum length of sentences after split (integer value of character length)
        embeddings_state (dict): State of the embeddings.
        output_folder (str, optional): Output folder. Defaults to output_folder.
        progress (gr.Progress, optional): Progress tracker for the cleaning process.

    Returns:
        tuple: A tuple containing the error message (if any), cleaned data, updated file name, and embeddings state.
    """
    
    output_text = ""
    output_list = []

    progress(0, desc = "Cleaning data")

    # If custom_regex is a string, assume this is a string path, and load in the data from the path
    if isinstance(custom_regex, str):
       custom_regex_text, custom_regex =  custom_regex_load(custom_regex)

    if not in_colnames:
        error_message = "Please enter one column name to use for cleaning and finding topics."
        print(error_message)
        return error_message, None, data_file_name_no_ext, None, None, embeddings_state

    all_tic = time.perf_counter()

    output_list = []
    #file_list = [string.name for string in in_files]

    for in_colnames_list_first in in_colnames:

        print("Cleaning column:", in_colnames_list_first)

        #in_colnames_list_first = in_colnames[0]

        # Reset original index to a new column so you can link it to data outputted from cleaning
        if not "original_index" in data.columns:
            data = data.reset_index(names="original_index")

        if clean_text == "Yes":
            clean_tic = time.perf_counter()
            print("Starting data clean.")

            data[in_colnames_list_first] = initial_clean(data[in_colnames_list_first], [])

            if '_clean' not in data_file_name_no_ext:
                data_file_name_no_ext = data_file_name_no_ext + "_clean"

            clean_toc = time.perf_counter()
            clean_time_out = f"Cleaning the text took {clean_toc - clean_tic:0.1f} seconds."
            print(clean_time_out)

        # Clean custom regex if exists
        if not custom_regex.empty:
            data[in_colnames_list_first] = regex_clean(data[in_colnames_list_first], custom_regex.iloc[:, 0].to_list())

            if '_clean' not in data_file_name_no_ext:
                data_file_name_no_ext = data_file_name_no_ext + "_clean"
            

        if drop_duplicate_text == "Yes":
            progress(0.3, desc= "Drop duplicates - remove short texts")

            data_file_name_no_ext = data_file_name_no_ext + "_dedup"

            #print("Removing duplicates and short entries from data")
            #print("Data shape before: ", data.shape)
            data[in_colnames_list_first] = data[in_colnames_list_first].str.strip()
            data = data[data[in_colnames_list_first].str.len() >= 50]
            data = data.drop_duplicates(subset = in_colnames_list_first).dropna(subset= in_colnames_list_first).reset_index()
            
            #print("Data shape after duplicate/null removal: ", data.shape)

        if anonymise_drop == "Yes":
            progress(0.4, desc= "Anonymising data")

            if '_anon' not in data_file_name_no_ext:
                data_file_name_no_ext = data_file_name_no_ext + "_anon"

            anon_tic = time.perf_counter()
            
            data_anon_col, anonymisation_success = anon.anonymise_script(data, in_colnames_list_first, anon_strat="redact")

            data[in_colnames_list_first] = data_anon_col

            print(anonymisation_success)

            anon_toc = time.perf_counter()
            time_out = f"Anonymising text took {anon_toc - anon_tic:0.1f} seconds"

            print(time_out)

        if sentence_split_drop == "Yes":
            progress(0.6, desc= "Splitting text into sentences")

            if '_split' not in data_file_name_no_ext:
                data_file_name_no_ext = data_file_name_no_ext + "_split"

            anon_tic = time.perf_counter()
            
            data = expand_sentences_spacy(data, in_colnames_list_first)
            data = data[data[in_colnames_list_first].str.len() > min_sentence_length] # Keep only rows with at more than 5 characters
            data[in_colnames_list_first] = data[in_colnames_list_first].str.strip()
            data.reset_index(inplace=True, drop=True)

            anon_toc = time.perf_counter()
            time_out = f"Splitting text took {anon_toc - anon_tic:0.1f} seconds"

            print(time_out)

            data[in_colnames_list_first] = data[in_colnames_list_first].str.strip()

    out_data_name = output_folder + data_file_name_no_ext + "_" + today_rev +  ".csv"
    data.to_csv(out_data_name)
    output_list.append(out_data_name)

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds."
    print(time_out)

    output_text = "Data clean completed."
    
    # Overwrite existing embeddings as they will likely have changed
    return output_text, output_list, data, data_file_name_no_ext, np.array([])

def optimise_zero_shot():
    """
    Return options that optimise the topic model to keep only zero-shot topics as the main topics
    """
    return gr.Dropdown(value="Yes"), gr.Slider(value=2), gr.Slider(value=2), gr.Slider(value=0.01), gr.Slider(value=0.95), gr.Slider(value=0.55)

def extract_topics(
    data: pd.DataFrame, 
    in_files: list, 
    min_docs_slider: int, 
    in_colnames: list, 
    max_topics_slider: int, 
    candidate_topics: list, 
    data_file_name_no_ext: str, 
    custom_labels_df: pd.DataFrame, 
    return_intermediate_files: str, 
    embeddings_super_compress: str, 
    high_quality_mode: str, 
    save_topic_model: str, 
    embeddings_out: np.ndarray, 
    embeddings_type_state: str, 
    zero_shot_similarity: float,
    calc_probs: str, 
    vectoriser_state: CountVectorizer, 
    min_word_occurence_slider: float, 
    max_word_occurence_slider: float, 
    split_sentence_drop: str,
    random_seed: int = random_seed,
    return_only_embeddings_drop: str = "No",
    output_folder: str = output_folder, 
    umap_n_neighbours:int = umap_n_neighbours,
    umap_min_dist:float = umap_min_dist,
    umap_metric:str = umap_metric,
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> tuple:
    """
    Extract topics from the given data using various parameters and settings.

    Args:
        data (pd.DataFrame): The input data.
        in_files (list): List of input files.
        min_docs_slider (int): Minimum number of similar documents needed to make a topic.
        in_colnames (list): List of column names to use for cleaning and finding topics.
        max_topics_slider (int): Maximum number of topics.
        candidate_topics (list): List of candidate topics.
        data_file_name_no_ext (str): Data file name without extension.
        custom_labels_df (pd.DataFrame): DataFrame containing custom labels.
        return_intermediate_files (str): Whether to return intermediate files.
        embeddings_super_compress (str): Whether to round embeddings to three decimal places.
        high_quality_mode (str): Whether to use high quality (transformers based) embeddings.
        save_topic_model (str): Whether to save the topic model.
        embeddings_out (np.ndarray): Output embeddings.
        embeddings_type_state (str): State of the embeddings type.
        zero_shot_similarity (float): Zero-shot similarity threshold.
        random_seed (int): Random seed for reproducibility.
        return_only_embeddings_drop (str): If you only want to output embeddings.
        calc_probs (str): Whether to calculate all topic probabilities.
        vectoriser_state (CountVectorizer): Vectorizer state.
        min_word_occurence_slider (float): Minimum word occurrence slider value.
        max_word_occurence_slider (float): Maximum word occurrence slider value.
        split_sentence_drop (str): Whether to split open text into sentences.
        original_data_state (pd.DataFrame): Original data state.
        output_folder (str, optional): Output folder. Defaults to output_folder.
        umap_n_neighbours (int): Nearest neighbours value for UMAP.
        umap_min_dist (float): Minimum distance for UMAP.
        umap_metric (str): Metric for UMAP.
        progress (gr.Progress, optional): Progress tracker. Defaults to gr.Progress(track_tqdm=True).

    Returns:
        tuple: A tuple containing output text, output list, data, data file name without extension, and an empty numpy array.
    """
    all_tic = time.perf_counter()

    progress(0, desc= "Loading data")

    vectoriser_state = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_word_occurence_slider, max_df=max_word_occurence_slider)

    output_list = []

    # If in_file is a string file path, otherwise assume it is a Gradio file input component
    if isinstance(in_files, str):
        file_list = [in_files]
    else:
        file_list = [string.name for string in in_files]

    if calc_probs == "No":
        calc_probs = False

    elif calc_probs == "Yes":
        print("Calculating all probabilities.")
        calc_probs = True
        
    if max_topics_slider == 0:
        max_topics_slider = 'auto'

    if not in_colnames:
        error_message = "Please enter one column name to use for cleaning and finding topics."
        print(error_message)
        return error_message, None, data_file_name_no_ext, embeddings_out, embeddings_type_state, data_file_name_no_ext, None, None, vectoriser_state, []

    in_colnames_list_first = in_colnames[0]

    docs = list(data[in_colnames_list_first])

    # Check if embeddings are being loaded in 
    progress(0.2, desc= "Loading/creating embeddings")


    if high_quality_mode == "Yes":
        print("Using high quality embedding model")

        #embedding_model = SentenceTransformer(embeddings_name, truncate_dim=512)       

        # If tfidf embeddings currently exist, wipe these empty
        if embeddings_type_state == "tfidf":
            embeddings_out = np.array([])

        embeddings_type_state = "large"

        # UMAP model uses Bertopic defaults
        #umap_model = UMAP(n_neighbors=umap_n_neighbours, n_components=5, min_dist=umap_min_dist, metric=umap_metric, low_memory=False, random_state=random_seed)

    else:
        print("Choosing low resource TF-IDF model.")

        # embedding_model = make_pipeline(
        #         TfidfVectorizer(),
        #         TruncatedSVD(100, random_state=random_seed)
        #         )
        
        # If large embeddings currently exist, wipe these empty, then rename embeddings type
        if embeddings_type_state == "large":
            embeddings_out = np.array([])

        embeddings_type_state = "tfidf"

        #umap_model = TruncatedSVD(n_components=5, random_state=random_seed)
    # UMAP model uses Bertopic defaults
    umap_model = UMAP(n_neighbors=umap_n_neighbours, n_components=5, min_dist=umap_min_dist, metric=umap_metric, low_memory=True, random_state=random_seed)

    embeddings_out, embedding_model = make_or_load_embeddings(docs, file_list, embeddings_out, embeddings_super_compress, high_quality_mode, embeddings_name)

     # If you want to save your embedding files
    if return_intermediate_files == "Yes":
        print("Saving embeddings to file")
        if high_quality_mode == "No":
            embeddings_file_name = output_folder + data_file_name_no_ext + '_' + 'tfidf_embeddings.npz'
        else:
            if embeddings_super_compress == "No":
                embeddings_file_name = output_folder + data_file_name_no_ext + '_' + 'large_embeddings.npz'
            else:
                embeddings_file_name = output_folder + data_file_name_no_ext + '_' + 'large_embeddings_compress.npz'

        print("output_folder:", output_folder)
        print("data_file_name_no_ext:", data_file_name_no_ext)
        print("embeddings_file_name:", embeddings_file_name)

        np.savez_compressed(embeddings_file_name, embeddings_out)

        output_list.append(embeddings_file_name)

        if return_only_embeddings_drop == "Yes":

            return "Embeddings output returned", output_list, embeddings_out, embeddings_type_state, data_file_name_no_ext, None, docs, vectoriser_state, []

    # This is saved as a Gradio state object
    vectoriser_model = vectoriser_state
 
    progress(0.3, desc= "Embeddings loaded. Creating BERTopic model")

    fail_error_message = "Topic model creation failed. Try reducing minimum documents per topic on the slider above (try 15 or less), then click 'Extract topics' again. If that doesn't work, try running the first two clean steps on your data first (see Clean data above) to ensure there are no NaNs/missing texts in your data."

    if not candidate_topics:
        
        try:
            # print("vectoriser_model:", vectoriser_model)

            topic_model = BERTopic( embedding_model=embedding_model,
                                    vectorizer_model=vectoriser_model,
                                    umap_model=umap_model,
                                    min_topic_size = min_docs_slider,
                                    nr_topics = max_topics_slider,
                                    calculate_probabilities=calc_probs,
                                    verbose = True)

            assigned_topics, probs = topic_model.fit_transform(docs, embeddings_out)

            if calc_probs == True:
                
                topics_probs_out = pd.DataFrame(topic_model.probabilities_)
                topics_probs_out_name = output_folder + "topic_full_probs_" + data_file_name_no_ext + "_" + today_rev + ".csv"
                topics_probs_out.to_csv(topics_probs_out_name)
                output_list.append(topics_probs_out_name)

        except Exception as error:
            print(error)
            print(fail_error_message)

            out_fail_error_message = '\n'.join([fail_error_message, str(error)])

            return out_fail_error_message, output_list, embeddings_out, embeddings_type_state, data_file_name_no_ext, None, docs, vectoriser_model, []
    

    # Do this if you have pre-defined topics
    else:
        #if high_quality_mode == "No":
        #    error_message = "Zero shot topic modelling currently not compatible with low-resource embeddings. Please change this option to 'No' on the options tab and retry."
        #    print(error_message)

        #    return error_message, output_list, embeddings_out, embeddings_type_state, data_file_name_no_ext, None, docs, vectoriser_model, []

        zero_shot_topics = read_file(candidate_topics.name)
        zero_shot_topics_lower = list(zero_shot_topics.iloc[:, 0].str.lower())

        print("Zero shot topics are:", zero_shot_topics_lower)

 
        try:
            topic_model = BERTopic( embedding_model=embedding_model, #embedding_model_pipe, # for Jina
                                    vectorizer_model=vectoriser_model,
                                    umap_model=umap_model,
                                    min_topic_size = min_docs_slider,
                                    nr_topics = max_topics_slider,
                                    zeroshot_topic_list = zero_shot_topics_lower,
                                    zeroshot_min_similarity = zero_shot_similarity, # 0.7
                                    calculate_probabilities=calc_probs,
                                    verbose = True)
            
            assigned_topics, probs = topic_model.fit_transform(docs, embeddings_out)

            if calc_probs == True:

                assigned_topics, probs = topic_model.transform(docs, embeddings_out)
                print("Probs:", probs)
                topic_model.probabilities_ = probs
                topics_probs_out = pd.DataFrame(topic_model.probabilities_)
                topics_probs_out_name = output_folder + "topic_full_probs_" + data_file_name_no_ext + "_" + today_rev + ".csv"
                topics_probs_out.to_csv(topics_probs_out_name)
                output_list.append(topics_probs_out_name)

        except Exception as error:
            print("An exception occurred:", error)
            print(fail_error_message)

            out_fail_error_message = '\n'.join([fail_error_message, str(error)])

            return out_fail_error_message, output_list, embeddings_out, embeddings_type_state, data_file_name_no_ext, None, docs, vectoriser_model, []

        # For some reason, zero topic modelling exports assigned topics as a np.array instead of a list. Converting it back here.
        if isinstance(assigned_topics, np.ndarray):
            assigned_topics = assigned_topics.tolist()

         # Zero shot modelling is a model merge, which wipes the c_tf_idf part of the resulting model completely. To get hierarchical modelling to work, we need to recreate this part of the model with the CountVectorizer options used to create the initial model. Since with zero shot, we are merging two models that have exactly the same set of documents, the vocubulary should be the same, and so recreating the cf_tf_idf component in this way shouldn't be a problem. Discussion here, and below based on Maarten's suggested code: https://github.com/MaartenGr/BERTopic/issues/1700     

        # Get document info
        doc_dets = topic_model.get_document_info(docs)

        documents_per_topic = doc_dets.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})

        # Assign CountVectorizer to merged model
        topic_model.vectorizer_model = vectoriser_model

        # Re-calculate c-TF-IDF
        c_tf_idf, _ = topic_model._c_tf_idf(documents_per_topic)
        topic_model.c_tf_idf_ = c_tf_idf

    # Check we have topics
    if not assigned_topics:
        return "No topics found.", output_list, embeddings_out, embeddings_type_state, data_file_name_no_ext, topic_model, docs, vectoriser_model,[]
    else: 
        print("Topic model created.")

    # Tidy up topic label format a bit to have commas and spaces by default
    if not candidate_topics:
        print("Zero shot topics not found, so not renaming")
        new_topic_labels = topic_model.generate_topic_labels(nr_words=3, separator=", ")
        topic_model.set_topic_labels(new_topic_labels)
    if candidate_topics:
        print("Custom labels:", topic_model.custom_labels_)
        print("Topic labels:", topic_model.topic_labels_)
        topic_model.set_topic_labels(topic_model.topic_labels_)

    # Replace current topic labels if new ones loaded in
    if not custom_labels_df.empty:
        #custom_label_list = list(custom_labels_df.iloc[:,0])
        custom_label_list = [label.replace("\n", "") for label in custom_labels_df.iloc[:,0]]

        topic_model.set_topic_labels(custom_label_list)
        
    print("Custom topics: ", topic_model.custom_labels_)

    # Outputs
    output_list, output_text = save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model, data, split_sentence_drop)

    

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds."
    print(time_out)

    return output_text, output_list, embeddings_out, embeddings_type_state, data_file_name_no_ext, topic_model, docs, vectoriser_model, assigned_topics

def reduce_outliers(topic_model: BERTopic, docs: List[str], embeddings_out: np.ndarray, data_file_name_no_ext: str, assigned_topics: Union[np.ndarray, List[int]], vectoriser_model: CountVectorizer, save_topic_model: str, split_sentence_drop: str, data: PandasDataFrame, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> tuple:
    """
    Reduce outliers in the topic model and update the topic representation.

    Args:
        topic_model (BERTopic): The BERTopic topic model to be used.
        docs (List[str]): List of documents.
        embeddings_out (np.ndarray): Output embeddings.
        data_file_name_no_ext (str): Data file name without extension.
        assigned_topics (Union[np.ndarray, List[int]]): Assigned topics.
        vectoriser_model (CountVectorizer): Vectorizer model.
        save_topic_model (str): Whether to save the topic model.
        split_sentence_drop (str): Dropdown result indicating whether sentences have been split.
        data (PandasDataFrame): The input dataframe
        progress (gr.Progress, optional): Progress tracker. Defaults to gr.Progress(track_tqdm=True).

    Returns:
        tuple: A tuple containing the output text, output list, and the updated topic model.
    """

    progress(0, desc= "Preparing data")

    output_list = []

    all_tic = time.perf_counter()

    if isinstance(assigned_topics, np.ndarray):
        assigned_topics = assigned_topics.tolist()

    # Reduce outliers if required, then update representation
    progress(0.2, desc= "Reducing outliers")
    print("Reducing outliers.")
    # Calculate the c-TF-IDF representation for each outlier document and find the best matching c-TF-IDF topic representation using cosine similarity.
    assigned_topics = topic_model.reduce_outliers(docs, assigned_topics, strategy="embeddings")
    # Then, update the topics to the ones that considered the new data

    progress(0.6, desc= "Updating original model")

    topic_model.update_topics(docs, topics=assigned_topics, vectorizer_model = vectoriser_model)

    # Tidy up topic label format a bit to have commas and spaces by default
    new_topic_labels = topic_model.generate_topic_labels(nr_words=3, separator=", ")
    topic_model.set_topic_labels(new_topic_labels)

    print("Finished reducing outliers.")

    # Outputs   
    progress(0.9, desc= "Saving to file")
    output_list, output_text = save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model, data, split_sentence_drop)

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds"
    print(time_out)
    
    return output_text, output_list, topic_model

def represent_topics(topic_model: BERTopic, docs: List[str], data_file_name_no_ext: str, high_quality_mode: str, save_topic_model: str, representation_type: str, vectoriser_model: CountVectorizer, split_sentence_drop: str, data: PandasDataFrame, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> tuple:
    """
    Represents topics using the specified representation model and updates the topic labels accordingly.

    Args:
        topic_model (BERTopic): The topic model to be used.
        docs (List[str]): List of documents to be processed.
        data_file_name_no_ext (str): The base name of the data file without extension.
        high_quality_mode (str): Whether to use high quality (transformers based) embeddings.
        save_topic_model (str): Whether to save the topic model.
        representation_type (str): The type of representation model to be used.
        vectoriser_model (CountVectorizer): The vectorizer model to be used.
        split_sentence_drop (str): Dropdown result indicating whether sentences have been split.
        data (PandasDataFrame): The input dataframe
        progress (gr.Progress, optional): Progress tracker for the process. Defaults to gr.Progress(track_tqdm=True).

    Returns:
        tuple: A tuple containing the output text, output list, and the updated topic model.
    """

    output_list = []

    all_tic = time.perf_counter()

    # Load in representation model

    progress(0.1, desc= "Loading model and creating new topic representation")

    representation_model = create_representation_model(representation_type, llm_config, hf_model_name, hf_model_file, chosen_start_tag, high_quality_mode)  

    progress(0.3, desc= "Updating existing topics")
    topic_model.update_topics(docs, vectorizer_model=vectoriser_model, representation_model=representation_model)

    topic_dets = topic_model.get_topic_info()

    # Replace original labels with LLM labels
    if representation_type == "LLM":
        llm_labels = [label[0].split("\n")[0] for label in topic_dets["LLM"]]
        topic_model.set_topic_labels(llm_labels)

        label_list_file_name = output_folder + data_file_name_no_ext + '_llm_topic_list_' + today_rev + '.csv'

        llm_labels_df = pd.DataFrame(data={"Label":llm_labels})
        llm_labels_df.to_csv(label_list_file_name, index=None)

        output_list.append(label_list_file_name)
    else:
        new_topic_labels = topic_model.generate_topic_labels(nr_words=3, separator=", ", aspect = representation_type)

        topic_model.set_topic_labels(new_topic_labels)

    # Outputs
    progress(0.8, desc= "Saving outputs")
    output_list, output_text = save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model, data, split_sentence_drop)

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds"
    print(time_out)

    return output_text, output_list, topic_model

def visualise_topics(
    topic_model: BERTopic, 
    data: pd.DataFrame, 
    data_file_name_no_ext: str, 
    high_quality_mode: str,  
    embeddings_out: np.ndarray, 
    in_label: List[str], 
    in_colnames: List[str], 
    legend_label: str, 
    sample_prop: float, 
    visualisation_type_radio: str, 
    random_seed: int = random_seed, 
    umap_n_neighbours: int = umap_n_neighbours, 
    umap_min_dist: float = umap_min_dist, 
    umap_metric: str = umap_metric, 
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> tuple:
    """
    Visualize topics using the provided topic model and data.

    Args:
        topic_model (BERTopic): The topic model to be used for visualization.
        data (pd.DataFrame): The input data containing the documents.
        data_file_name_no_ext (str): The base name of the data file without extension.
        high_quality_mode (str): Whether to use high quality mode for embeddings.
        embeddings_out (np.ndarray): The output embeddings.
        in_label (List[str]): List of labels for the input data.
        in_colnames (List[str]): List of column names in the input data.
        legend_label (str): The label to be used in the legend.
        sample_prop (float): The proportion of data to sample for visualization.
        visualisation_type_radio (str): The type of visualization to be used.
        random_seed (int, optional): Random seed for reproducibility. Defaults to random_seed.
        umap_n_neighbours (int, optional): Number of neighbors for UMAP. Defaults to umap_n_neighbours.
        umap_min_dist (float, optional): Minimum distance for UMAP. Defaults to umap_min_dist.
        umap_metric (str, optional): Metric for UMAP. Defaults to umap_metric.
        progress (gr.Progress, optional): Progress tracker for the process. Defaults to gr.Progress(track_tqdm=True).

    Returns:
        tuple: A tuple containing the output message, output list, reduced embeddings, and topic model.
    """

    progress(0, desc= "Preparing data for visualisation")

    output_list = []
    output_message = []
    vis_tic = time.perf_counter()

    
    if not visualisation_type_radio:
        return "Please choose a visualisation type above.", output_list, None, None

    # Get topic labels
    if in_label:
       in_label_list_first = in_label[0]
    else:
       return "Label column not found. Please enter this above.", output_list, None, None
    
    # Get docs
    if in_colnames:
        in_colnames_list_first = in_colnames[0]
    else:
        return "Label column not found. Please enter this on the data load tab.", output_list, None, None
    
    docs = list(data[in_colnames_list_first].str.lower())

    # Make sure format of input series is good
    data[in_label_list_first] = data[in_label_list_first].fillna('').astype(str)
    label_list = list(data[in_label_list_first])

    topic_dets = topic_model.get_topic_info()

    # Replace original labels with another representation if specified
    if legend_label:
        topic_dets = topic_model.get_topics(full=True)
        if legend_label in topic_dets:
            labels = [topic_dets[legend_label].values()]
            labels = [str(v) for v in labels]
            topic_model.set_topic_labels(labels)

    # Pre-reduce embeddings for visualisation purposes
    if high_quality_mode == "Yes":
        reduced_embeddings = UMAP(n_neighbors=umap_n_neighbours, n_components=2, min_dist=umap_min_dist, metric=umap_metric, random_state=random_seed).fit_transform(embeddings_out)
    else:
        reduced_embeddings = TruncatedSVD(2, random_state=random_seed).fit_transform(embeddings_out)

    progress(0.3, desc= "Creating visualisations")
    # Visualise the topics:
    
    print("Creating visualisations")

    if visualisation_type_radio == "Topic document graph":
        try:
            topics_vis = visualize_documents_custom(topic_model, docs, hover_labels = label_list, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True, sample = sample_prop, width= 1200, height = 750)

            topics_vis_name = output_folder + data_file_name_no_ext + '_' + 'vis_topic_docs_' + today_rev + '.html'
            topics_vis.write_html(topics_vis_name)
            output_list.append(topics_vis_name)
        except Exception as e:
            print(e)
            output_message = str(e)
            return output_message, output_list, None, None

        try:
            topics_vis_2 = topic_model.visualize_heatmap(custom_labels=True, width= 1200, height = 1200)

            topics_vis_2_name = output_folder + data_file_name_no_ext + '_' + 'vis_heatmap_' + today_rev + '.html'
            topics_vis_2.write_html(topics_vis_2_name)
            output_list.append(topics_vis_2_name)
        except Exception as e:
            print(e)
            output_message.append(str(e))

    elif visualisation_type_radio == "Hierarchical view":

        hierarchical_topics = hierarchical_topics_custom(topic_model, docs)

        # Print topic tree - may get encoding errors, so doing try except
        try:
            tree = topic_model.get_topic_tree(hierarchical_topics, tight_layout = True)
            tree_name = output_folder + data_file_name_no_ext + '_' + 'vis_hierarchy_tree_' + today_rev + '.txt'

            with open(tree_name, "w") as file:
                # Write the string to the file
                file.write(tree)

            output_list.append(tree_name)

        except Exception as e:
            new_out_message = "An exception occurred when making topic tree document, skipped:" + str(e)
            output_message.append(str(new_out_message))
            print(new_out_message)


        # Save new hierarchical topic model to file
        try:
            hierarchical_topics_name = output_folder + data_file_name_no_ext + '_' + 'vis_hierarchy_topics_dist_' + today_rev + '.csv'
            hierarchical_topics.to_csv(hierarchical_topics_name, index = None)
            output_list.append(hierarchical_topics_name)

            topics_vis, hierarchy_df, hierarchy_topic_names = visualize_hierarchical_documents_custom(topic_model, docs, label_list, hierarchical_topics, hide_annotations=True, reduced_embeddings=reduced_embeddings, sample = sample_prop, hide_document_hover= False, custom_labels=True, width= 1200, height = 750)
            topics_vis_2 = visualize_hierarchy_custom(topic_model, hierarchical_topics=hierarchical_topics, width= 1200, height = 750)
        except Exception as e:
            new_out_message = "An exception occurred when making hierarchical topic visualisation:" + str(e) + ". Maybe your model doesn't have enough topics to create a hierarchy?"
            output_message.append(str(new_out_message))
            print(new_out_message)
            return new_out_message, output_list, None, None

        # Write hierarchical topics levels to df
        hierarchy_df_name = output_folder + data_file_name_no_ext + '_' + 'hierarchy_topics_df_' + today_rev + '.csv'
        hierarchy_df.to_csv(hierarchy_df_name, index = None)
        output_list.append(hierarchy_df_name)

        # Write hierarchical topics names to df
        hierarchy_topic_names_name = output_folder + data_file_name_no_ext + '_' + 'hierarchy_topics_names_' + today_rev + '.csv'
        hierarchy_topic_names.to_csv(hierarchy_topic_names_name, index = None)
        output_list.append(hierarchy_topic_names_name)


        topics_vis_name = output_folder + data_file_name_no_ext + '_' + 'vis_hierarchy_topic_doc_' + today_rev + '.html'
        topics_vis.write_html(topics_vis_name)
        output_list.append(topics_vis_name)

        topics_vis_2_name = output_folder + data_file_name_no_ext + '_' + 'vis_hierarchy_' + today_rev + '.html'
        topics_vis_2.write_html(topics_vis_2_name)
        output_list.append(topics_vis_2_name)

    all_toc = time.perf_counter()
    output_message.append(f"Creating visualisation took {all_toc - vis_tic:0.1f} seconds")
    print(output_message)

    return '\n'.join(output_message), output_list, topics_vis, topics_vis_2

def save_as_pytorch_model(topic_model: BERTopic, data_file_name_no_ext:str, progress=gr.Progress(track_tqdm=True)):
    """
    Reduce outliers in the topic model and update the topic representation.

    Args:
        topic_model (BERTopic): The BERTopic topic model to be used.
        data_file_name_no_ext (str): Document file name.
    Returns:
        tuple: A tuple containing the output text and output list.
    """
    output_list = []
    output_message = ""

    if not topic_model:
        output_message = "No Pytorch model found."
        return output_message, None

    progress(0, desc= "Saving topic model in Pytorch format")

    topic_model_save_name_folder = output_folder + data_file_name_no_ext + "_topics_" + today_rev# + ".safetensors"
    topic_model_save_name_zip = topic_model_save_name_folder + ".zip"

    # Clear folder before replacing files
    delete_files_in_folder(topic_model_save_name_folder)

    topic_model.save(topic_model_save_name_folder, serialization='pytorch', save_embedding_model=True, save_ctfidf=False)

    # Zip file example    
    zip_folder(topic_model_save_name_folder, topic_model_save_name_zip)
    output_list.append(topic_model_save_name_zip)

    output_message = "Model saved in Pytorch format."

    return output_message, output_list
