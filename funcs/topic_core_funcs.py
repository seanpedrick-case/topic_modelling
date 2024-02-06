# Dendrograms will not work with the latest version of scipy (1.12.0), so installing the version prior to be safe
#os.system("pip install scipy==1.11.4")

import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
import time
from bertopic import BERTopic

from funcs.clean_funcs import initial_clean
from funcs.helper_functions import read_file, zip_folder, delete_files_in_folder, save_topic_outputs
from funcs.embeddings import make_or_load_embeddings

from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import funcs.anonymiser as anon
from umap import UMAP

from torch import cuda, backends, version

# Default seed, can be changed in number selection on options page
random_seed = 42

# Check for torch cuda
# If you want to disable cuda for testing purposes
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Is CUDA enabled? ", cuda.is_available())
print("Is a CUDA device available on this computer?", backends.cudnn.enabled)
if cuda.is_available():
    torch_device = "gpu"
    print("Cuda version installed is: ", version.cuda)
    low_resource_mode = "No"
    #os.system("nvidia-smi")
else: 
    torch_device =  "cpu"
    low_resource_mode = "Yes"

print("Device used is: ", torch_device)

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# Load embeddings
embeddings_name = "BAAI/bge-small-en-v1.5" #"jinaai/jina-embeddings-v2-base-en"

# LLM model used for representing topics
hf_model_name =  'second-state/stablelm-2-zephyr-1.6b-GGUF' #'TheBloke/phi-2-orange-GGUF' #'NousResearch/Nous-Capybara-7B-V1.9-GGUF'
hf_model_file =   'stablelm-2-zephyr-1_6b-Q5_K_M.gguf' # 'phi-2-orange.Q5_K_M.gguf' #'Capybara-7B-V1.9-Q5_K_M.gguf'

def pre_clean(data, in_colnames, data_file_name_no_ext, custom_regex, clean_text, drop_duplicate_text, anonymise_drop, progress=gr.Progress(track_tqdm=True)):
    
    output_text = ""
    output_list = []

    progress(0, desc = "Cleaning data")

    if not in_colnames:
        error_message = "Please enter one column name to use for cleaning and finding topics."
        print(error_message)
        return error_message, None, data_file_name_no_ext, None, None

    all_tic = time.perf_counter()

    output_list = []
    #file_list = [string.name for string in in_files]

    in_colnames_list_first = in_colnames[0]

    if clean_text == "Yes":
        clean_tic = time.perf_counter()
        print("Starting data clean.")

        data_file_name_no_ext = data_file_name_no_ext + "_clean"

        if not custom_regex.empty:
            data[in_colnames_list_first] = initial_clean(data[in_colnames_list_first], custom_regex.iloc[:, 0].to_list())
        else:
            data[in_colnames_list_first] = initial_clean(data[in_colnames_list_first], [])

        clean_toc = time.perf_counter()
        clean_time_out = f"Cleaning the text took {clean_toc - clean_tic:0.1f} seconds."
        print(clean_time_out)

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
        progress(0.6, desc= "Anonymising data")

        data_file_name_no_ext = data_file_name_no_ext + "_anon"

        anon_tic = time.perf_counter()
        
        data_anon_col, anonymisation_success = anon.anonymise_script(data, in_colnames_list_first, anon_strat="redact")

        data[in_colnames_list_first] = data_anon_col

        print(anonymisation_success)

        anon_toc = time.perf_counter()
        time_out = f"Anonymising text took {anon_toc - anon_tic:0.1f} seconds"

    out_data_name = data_file_name_no_ext + "_" + today_rev +  ".csv"
    data.to_csv(out_data_name)
    output_list.append(out_data_name)

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds."
    print(time_out)

    output_text = "Data clean completed."
    
    return output_text, output_list, data, data_file_name_no_ext

def extract_topics(data, in_files, min_docs_slider, in_colnames, max_topics_slider, candidate_topics, data_file_name_no_ext, custom_labels_df, return_intermediate_files, embeddings_super_compress, low_resource_mode, save_topic_model, embeddings_out, zero_shot_similarity, random_seed, calc_probs, vectoriser_state, progress=gr.Progress(track_tqdm=True)):

    all_tic = time.perf_counter()

    progress(0, desc= "Loading data")

    output_list = []
    file_list = [string.name for string in in_files]

    if calc_probs == "No":
        calc_probs = False

    elif calc_probs == "Yes":
        print("Calculating all probabilities.")
        calc_probs = True

    if not in_colnames:
        error_message = "Please enter one column name to use for cleaning and finding topics."
        print(error_message)
        return error_message, None, data_file_name_no_ext, embeddings_out, None, None

    

    in_colnames_list_first = in_colnames[0]

    docs = list(data[in_colnames_list_first])

    # Check if embeddings are being loaded in 
    progress(0.2, desc= "Loading/creating embeddings")

    print("Low resource mode: ", low_resource_mode)

    if low_resource_mode == "No":
        print("Using high resource BGE transformer model")

        embedding_model = SentenceTransformer(embeddings_name)

        # UMAP model uses Bertopic defaults
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', low_memory=False, random_state=random_seed)

    else:
        print("Choosing low resource TF-IDF model.")

        embedding_model_pipe = make_pipeline(
                TfidfVectorizer(),
                TruncatedSVD(100, random_state=random_seed)
                )
        embedding_model = embedding_model_pipe

        umap_model = TruncatedSVD(n_components=5, random_state=random_seed)

    embeddings_out = make_or_load_embeddings(docs, file_list, embeddings_out, embedding_model, embeddings_super_compress, low_resource_mode)

    # This is saved as a Gradio state object
    vectoriser_model = vectoriser_state
 
    progress(0.3, desc= "Embeddings loaded. Creating BERTopic model")

    fail_error_message = "Topic model creation failed. Try reducing minimum documents per topic on the slider above (try 15 or less), then click 'Extract topics' again."

    if not candidate_topics:
        
        try:

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
                topics_probs_out_name = "topic_full_probs_" + data_file_name_no_ext + "_" + today_rev + ".csv"
                topics_probs_out.to_csv(topics_probs_out_name)
                output_list.append(topics_probs_out_name)

        except:
            print(fail_error_message)

            return fail_error_message, output_list, embeddings_out, data_file_name_no_ext, None, docs, vectoriser_model
    

    # Do this if you have pre-defined topics
    else:
        if low_resource_mode == "Yes":
            error_message = "Zero shot topic modelling currently not compatible with low-resource embeddings. Please change this option to 'No' on the options tab and retry."
            print(error_message)

            return error_message, output_list, embeddings_out, data_file_name_no_ext, None, docs, vectoriser_model

        zero_shot_topics = read_file(candidate_topics.name)
        zero_shot_topics_lower = list(zero_shot_topics.iloc[:, 0].str.lower())

 
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
                topics_probs_out = pd.DataFrame(topic_model.probabilities_)
                topics_probs_out_name = "topic_full_probs_" + data_file_name_no_ext + "_" + today_rev + ".csv"
                topics_probs_out.to_csv(topics_probs_out_name)
                output_list.append(topics_probs_out_name)

        except:
            print(fail_error_message)

            return fail_error_message, output_list, embeddings_out, data_file_name_no_ext, None, docs, vectoriser_model

        # For some reason, zero topic modelling exports assigned topics as a np.array instead of a list. Converting it back here.
        if isinstance(assigned_topics, np.ndarray):
            assigned_topics = assigned_topics.tolist()

         # Zero shot modelling is a model merge, which wipes the c_tf_idf part of the resulting model completely. To get hierarchical modelling to work, we need to recreate this part of the model with the CountVectorizer options used to create the initial model. Since with zero shot, we are merging two models that have exactly the same set of documents, the vocubulary should be the same, and so recreating the cf_tf_idf component in this way shouldn't be a problem. Discussion here, and below based on Maarten's suggested code: https://github.com/MaartenGr/BERTopic/issues/1700

        doc_dets = topic_model.get_document_info(docs)

        documents_per_topic = doc_dets.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})

        # Assign CountVectorizer to merged model

        topic_model.vectorizer_model = vectoriser_model

        # Re-calculate c-TF-IDF
        c_tf_idf, _ = topic_model._c_tf_idf(documents_per_topic)
        topic_model.c_tf_idf_ = c_tf_idf

    if not assigned_topics:
    # Handle the empty array case
        return "No topics found.", output_list, embeddings_out, data_file_name_no_ext, topic_model, docs
    
    else: 
        print("Topic model created.")

    # Replace current topic labels if new ones loaded in
    if not custom_labels_df.empty:
        #custom_label_list = list(custom_labels_df.iloc[:,0])
        custom_label_list = [label.replace("\n", "") for label in custom_labels_df.iloc[:,0]]

        topic_model.set_topic_labels(custom_label_list)
        
    print("Custom topics: ", topic_model.custom_labels_)

    # Outputs
    output_list, output_text = save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model)

     # If you want to save your embedding files
    if return_intermediate_files == "Yes":
        print("Saving embeddings to file")
        if low_resource_mode == "Yes":
            embeddings_file_name = data_file_name_no_ext + '_' + 'tfidf_embeddings.npz'
        else:
            if embeddings_super_compress == "No":
                embeddings_file_name = data_file_name_no_ext + '_' + 'bge_embeddings.npz'
            else:
                embeddings_file_name = data_file_name_no_ext + '_' + 'bge_embeddings_compress.npz'

        np.savez_compressed(embeddings_file_name, embeddings_out)

        output_list.append(embeddings_file_name)

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds."
    print(time_out)

    return output_text, output_list, embeddings_out, data_file_name_no_ext, topic_model, docs, vectoriser_model

def reduce_outliers(topic_model, docs, embeddings_out, data_file_name_no_ext, save_topic_model, progress=gr.Progress(track_tqdm=True)):

    progress(0, desc= "Preparing data")

    output_list = []

    all_tic = time.perf_counter()

    assigned_topics, probs = topic_model.fit_transform(docs, embeddings_out)

    if isinstance(assigned_topics, np.ndarray):
        assigned_topics = assigned_topics.tolist()


    # Reduce outliers if required, then update representation
    progress(0.2, desc= "Reducing outliers")
    print("Reducing outliers.")
    # Calculate the c-TF-IDF representation for each outlier document and find the best matching c-TF-IDF topic representation using cosine similarity.
    assigned_topics = topic_model.reduce_outliers(docs, assigned_topics, strategy="embeddings")
    # Then, update the topics to the ones that considered the new data

    progress(0.6, desc= "Updating original model")
    topic_model.update_topics(docs, topics=assigned_topics)

    print("Finished reducing outliers.")

    #progress(0.7, desc= "Replacing topic names with LLMs if necessary")

    #topic_dets = topic_model.get_topic_info()

    # # Replace original labels with LLM labels
    # if "LLM" in topic_model.get_topic_info().columns:
    #     llm_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["LLM"].values()]
    #     topic_model.set_topic_labels(llm_labels)
    # else:
    #     topic_model.set_topic_labels(list(topic_dets["Name"]))

    # Outputs   
    progress(0.9, desc= "Saving to file")
    output_list, output_text = save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model)

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds"
    print(time_out)
    
    return output_text, output_list, topic_model

def represent_topics(topic_model, docs, data_file_name_no_ext, low_resource_mode, save_topic_model, representation_type, vectoriser_model, progress=gr.Progress(track_tqdm=True)):
    from funcs.representation_model import create_representation_model, llm_config, chosen_start_tag

    output_list = []

    all_tic = time.perf_counter()

    progress(0.1, desc= "Loading model and creating new representation")

    representation_model = create_representation_model(representation_type, llm_config, hf_model_name, hf_model_file, chosen_start_tag, low_resource_mode)  

    progress(0.6, desc= "Updating existing topics")
    topic_model.update_topics(docs, vectorizer_model=vectoriser_model, representation_model=representation_model)

    topic_dets = topic_model.get_topic_info()

    # Replace original labels with LLM labels
    if representation_type == "LLM":
        llm_labels = [label[0].split("\n")[0] for label in topic_dets["LLM"]]
        topic_model.set_topic_labels(llm_labels)

        label_list_file_name = data_file_name_no_ext + '_llm_topic_list_' + today_rev + '.csv'

        llm_labels_df = pd.DataFrame(data={"Label":llm_labels})
        llm_labels_df.to_csv(label_list_file_name, index=None)

        output_list.append(label_list_file_name)
    else:
        new_topic_labels = topic_model.generate_topic_labels(nr_words=3, separator=", ", aspect = representation_type)

        topic_model.set_topic_labels(new_topic_labels)#list(topic_dets[representation_type]))
        #topic_model.set_topic_labels(list(topic_dets["Name"]))

    # Outputs
    progress(0.8, desc= "Saving outputs")
    output_list, output_text = save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model)

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds"
    print(time_out)

    return output_text, output_list, topic_model

def visualise_topics(topic_model, data, data_file_name_no_ext, low_resource_mode,  embeddings_out, in_label, in_colnames, legend_label, sample_prop, visualisation_type_radio, random_seed,  progress=gr.Progress(track_tqdm=True)):

    progress(0, desc= "Preparing data for visualisation")

    output_list = []
    vis_tic = time.perf_counter()

    from funcs.bertopic_vis_documents import visualize_documents_custom, visualize_hierarchical_documents_custom, visualize_barchart_custom

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
    if low_resource_mode == "No":
        reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=random_seed).fit_transform(embeddings_out)
    else:
        reduced_embeddings = TruncatedSVD(2, random_state=random_seed).fit_transform(embeddings_out)

    progress(0.5, desc= "Creating visualisation (this can take a while)")
    # Visualise the topics:
    
    print("Creating visualisation")

    # "Topic document graph", "Hierarchical view"

    if visualisation_type_radio == "Topic document graph":
        topics_vis = visualize_documents_custom(topic_model, docs, hover_labels = label_list, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True, sample = sample_prop, width= 1200, height = 750)

        topics_vis_name = data_file_name_no_ext + '_' + 'vis_topic_docs_' + today_rev + '.html'
        topics_vis.write_html(topics_vis_name)
        output_list.append(topics_vis_name)

        topics_vis_2 = topic_model.visualize_heatmap(custom_labels=True, width= 1200, height = 1200)

        topics_vis_2_name = data_file_name_no_ext + '_' + 'vis_heatmap_' + today_rev + '.html'
        topics_vis_2.write_html(topics_vis_2_name)
        output_list.append(topics_vis_2_name)

    elif visualisation_type_radio == "Hierarchical view":

        hierarchical_topics = topic_model.hierarchical_topics(docs)

        # Print topic tree
        tree = topic_model.get_topic_tree(hierarchical_topics, tight_layout = True)
        tree_name = data_file_name_no_ext + '_' + 'vis_hierarchy_tree_' + today_rev + '.txt'

        with open(tree_name, "w") as file:
            # Write the string to the file
            file.write(tree)

        output_list.append(tree_name)

        # Save new hierarchical topic model to file
        hierarchical_topics_name = data_file_name_no_ext + '_' + 'vis_hierarchy_topics_' + today_rev + '.csv'
        hierarchical_topics.to_csv(hierarchical_topics_name)
        output_list.append(hierarchical_topics_name)

        try:
            topics_vis = visualize_hierarchical_documents_custom(topic_model, docs, label_list, hierarchical_topics, reduced_embeddings=reduced_embeddings, sample = sample_prop, hide_document_hover= False, custom_labels=True, width= 1200, height = 750)
            topics_vis_2 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, width= 1200, height = 750)
        except:
            error_message = "Visualisation preparation failed. Perhaps you need more topics to create the full hierarchy (more than 10)?"
            return error_message, output_list, None, None

        topics_vis_name = data_file_name_no_ext + '_' + 'vis_hierarchy_topic_doc_' + today_rev + '.html'
        topics_vis.write_html(topics_vis_name)
        output_list.append(topics_vis_name)

        topics_vis_2_name = data_file_name_no_ext + '_' + 'vis_hierarchy_' + today_rev + '.html'
        topics_vis_2.write_html(topics_vis_2_name)
        output_list.append(topics_vis_2_name)

    all_toc = time.perf_counter()
    time_out = f"Creating visualisation took {all_toc - vis_tic:0.1f} seconds"
    print(time_out)

    return time_out, output_list, topics_vis, topics_vis_2

def save_as_pytorch_model(topic_model, data_file_name_no_ext , progress=gr.Progress(track_tqdm=True)):

    if not topic_model:
        return "No Pytorch model found.", None

    progress(0, desc= "Saving topic model in Pytorch format")

    output_list = []


    topic_model_save_name_folder = "output_model/" + data_file_name_no_ext + "_topics_" + today_rev# + ".safetensors"
    topic_model_save_name_zip = topic_model_save_name_folder + ".zip"

    # Clear folder before replacing files
    delete_files_in_folder(topic_model_save_name_folder)

    topic_model.save(topic_model_save_name_folder, serialization='pytorch', save_embedding_model=True, save_ctfidf=False)

    # Zip file example
    
    zip_folder(topic_model_save_name_folder, topic_model_save_name_zip)
    output_list.append(topic_model_save_name_zip)

    return "Model saved in Pytorch format.", output_list
