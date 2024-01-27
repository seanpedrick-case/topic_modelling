import os
import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
import time

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
from transformers.pipelines import pipeline
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import funcs.anonymiser as anon
from umap import UMAP

from torch import cuda, backends, version

random_seed = 42

# Check for torch cuda
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

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from bertopic import BERTopic
#from sentence_transformers import SentenceTransformer
#from bertopic.backend._hftransformers import HFTransformerBackend

#from cuml.manifold import UMAP

#umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

from funcs.helper_functions import dummy_function, put_columns_in_df, read_file, get_file_path_end, zip_folder, delete_files_in_folder
#from funcs.representation_model import representation_model
from funcs.embeddings import make_or_load_embeddings

# Log terminal output: https://github.com/gradio-app/gradio/issues/2362

import sys

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

# Load embeddings

# Pinning a Jina revision for security purposes: https://www.baseten.co/blog/pinning-ml-model-revisions-for-compatibility-and-security/
# Save Jina model locally as described here: https://huggingface.co/jinaai/jina-embeddings-v2-base-en/discussions/29
embeddings_name = "BAAI/bge-small-en-v1.5" #"jinaai/jina-embeddings-v2-base-en"
# local_embeddings_location = "model/jina/"
#revision_choice = "b811f03af3d4d7ea72a7c25c802b21fc675a5d99"
#revision_choice = "69d43700292701b06c24f43b96560566a4e5ad1f"

# Model used for representing topics
hf_model_name =  'second-state/stablelm-2-zephyr-1.6b-GGUF' #'TheBloke/phi-2-orange-GGUF' #'NousResearch/Nous-Capybara-7B-V1.9-GGUF' # 'second-state/stablelm-2-zephyr-1.6b-GGUF'
hf_model_file =   'stablelm-2-zephyr-1_6b-Q5_K_M.gguf' # 'phi-2-orange.Q5_K_M.gguf' #'Capybara-7B-V1.9-Q5_K_M.gguf' # 'stablelm-2-zephyr-1_6b-Q5_K_M.gguf'

def save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model, progress=gr.Progress()):
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
        doc_dets = topic_model.get_document_info(docs)[["Document",	"Topic", "Name", "Representative_document"]] # "Probability",
        doc_dets.to_csv(doc_det_output_name)
        output_list.append(doc_det_output_name)

        topics_text_out_str = str(topic_dets["Name"])
        output_text = "Topics: " + topics_text_out_str
    
        # Save topic model to file
        if save_topic_model == "Yes":
            topic_model_save_name_pkl = "output_model/" + data_file_name_no_ext + "_topics_" + today_rev + ".pkl"# + ".safetensors"
            topic_model_save_name_zip = topic_model_save_name_pkl + ".zip"

            # Clear folder before replacing files
            #delete_files_in_folder(topic_model_save_name_pkl)

            topic_model.save(topic_model_save_name_pkl, serialization='pickle', save_embedding_model=False, save_ctfidf=False)

            # Zip file example
            
            #zip_folder(topic_model_save_name_pkl, topic_model_save_name_zip)
            output_list.append(topic_model_save_name_pkl)

        return output_list, output_text

def extract_topics(data, in_files, min_docs_slider, in_colnames, max_topics_slider, candidate_topics, in_label, anonymise_drop, return_intermediate_files, embeddings_super_compress, low_resource_mode, save_topic_model, embeddings_out, zero_shot_similarity, progress=gr.Progress()):

    progress(0, desc= "Loading data")

    if not in_colnames or not in_label:
        error_message = "Please enter one column name for the topics and another for the labelling."
        print(error_message)
        return error_message, None, None, embeddings_out

    all_tic = time.perf_counter()

    output_list = []
    file_list = [string.name for string in in_files]

    data_file_names = [string.lower() for string in file_list if "tokenised" not in string and "npz" not in string.lower() and "gz" not in string.lower()]
    data_file_name = data_file_names[0]
    data_file_name_no_ext = get_file_path_end(data_file_name)

    in_colnames_list_first = in_colnames[0]

    if in_label:
        in_label_list_first = in_label[0]
    else:
        in_label_list_first = in_colnames_list_first

    # Make sure format of input series is good
    data[in_colnames_list_first] = data[in_colnames_list_first].fillna('').astype(str)
    data[in_label_list_first] = data[in_label_list_first].fillna('').astype(str)
    label_list = list(data[in_label_list_first])
    
    if anonymise_drop == "Yes":
        progress(0.1, desc= "Anonymising data")
        anon_tic = time.perf_counter()
        
        data_anon_col, anonymisation_success = anon.anonymise_script(data, in_colnames_list_first, anon_strat="replace")
        data[in_colnames_list_first] = data_anon_col[in_colnames_list_first]
        anonymise_data_name = data_file_name_no_ext + "_anonymised_" + today_rev +  ".csv"
        data.to_csv(anonymise_data_name)
        output_list.append(anonymise_data_name)

        anon_toc = time.perf_counter()
        time_out = f"Anonymising text took {anon_toc - anon_tic:0.1f} seconds"

    docs = list(data[in_colnames_list_first].str.lower())
    

    # Check if embeddings are being loaded in 
    progress(0.2, desc= "Loading/creating embeddings")

    print("Low resource mode: ", low_resource_mode)

    if low_resource_mode == "No":
        print("Using high resource BGE transformer model")
        
        

        embedding_model = SentenceTransformer(embeddings_name)
        #try:
        #embedding_model = AutoModel.from_pretrained(embeddings_name, revision = revision_choice, trust_remote_code=True,device_map="auto") # For Jina
        #except:
        #     embedding_model = AutoModel.from_pretrained(embeddings_name)#, revision = revision_choice, trust_remote_code=True, device_map="auto", use_auth_token=os.environ["HF_TOKEN"])
        #tokenizer = AutoTokenizer.from_pretrained(embeddings_name)
        #embedding_model_pipe = pipeline("feature-extraction", model=embedding_model, tokenizer=tokenizer)

        # UMAP model uses Bertopic defaults
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', low_memory=False, random_state=random_seed)

    elif low_resource_mode == "Yes":
        print("Choosing low resource TF-IDF model.")

        embedding_model_pipe = make_pipeline(
                TfidfVectorizer(),
                TruncatedSVD(100) # 100 # To be compatible with zero shot, this needs to be lower than number of suggested topics
                )
        embedding_model = embedding_model_pipe

        umap_model = TruncatedSVD(n_components=5, random_state=random_seed)

 

    embeddings_out, reduced_embeddings = make_or_load_embeddings(docs, file_list, embeddings_out, embedding_model, embeddings_super_compress, low_resource_mode)

    vectoriser_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.1)
 
    progress(0.3, desc= "Embeddings loaded. Creating BERTopic model")

    if not candidate_topics:
        
        topic_model = BERTopic( embedding_model=embedding_model, #embedding_model_pipe, #for Jina
                                vectorizer_model=vectoriser_model,
                                umap_model=umap_model,
                                min_topic_size = min_docs_slider,
                                nr_topics = max_topics_slider,
                                verbose = True)

        topics_text, probs = topic_model.fit_transform(docs, embeddings_out)

        if not topics_text:
        # Handle the empty array case

            return "No topics found.", data_file_name, None, embeddings_out, data_file_name_no_ext, topic_model, docs, label_list
        
        else: 
            print("Topic model created.")


    # Do this if you have pre-defined topics
    else:
        if low_resource_mode == "Yes":
            error_message = "Zero shot topic modelling currently not compatible with low-resource embeddings. Please change this option to 'No' on the options tab and retry."
            print(error_message)

            return error_message, output_list, None, embeddings_out, data_file_name_no_ext, None, docs, label_list

        zero_shot_topics = read_file(candidate_topics.name)
        zero_shot_topics_lower = list(zero_shot_topics.iloc[:, 0].str.lower())

        topic_model = BERTopic( embedding_model=embedding_model, #embedding_model_pipe, # for Jina
                                vectorizer_model=vectoriser_model,
                                umap_model=umap_model,
                                min_topic_size = min_docs_slider,
                                nr_topics = max_topics_slider,
                                zeroshot_topic_list = zero_shot_topics_lower,
                                zeroshot_min_similarity = zero_shot_similarity, # 0.7
                                verbose = True)
        
        topics_text, probs = topic_model.fit_transform(docs, embeddings_out)

       # print(topics_text)

        if topics_text.size == 0:
        # Handle the empty array case

            return "No topics found.", data_file_name, None, embeddings_out, data_file_name_no_ext, topic_model, docs, label_list
        
        else: 
            print("Topic model created.")

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

    return output_text, output_list, None, embeddings_out, data_file_name_no_ext, topic_model, docs, label_list

def reduce_outliers(topic_model, docs, embeddings_out, data_file_name_no_ext, low_resource_mode, create_llm_topic_labels, save_topic_model, progress=gr.Progress()):
    #from funcs.prompts import capybara_prompt, capybara_start, open_hermes_prompt, open_hermes_start, stablelm_prompt, stablelm_start
    from funcs.representation_model import create_representation_model, llm_config, chosen_start_tag

    output_list = []

    all_tic = time.perf_counter()

    vectoriser_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.1)

    topics_text, probs = topic_model.fit_transform(docs, embeddings_out)

    #progress(0.2, desc= "Loading in representation model")
    #print("Create LLM topic labels:", create_llm_topic_labels)
    #representation_model = create_representation_model(create_llm_topic_labels, llm_config, hf_model_name, hf_model_file, chosen_start_tag, low_resource_mode)  

    # Reduce outliers if required, then update representation
    progress(0.2, desc= "Reducing outliers")
    print("Reducing outliers.")
    # Calculate the c-TF-IDF representation for each outlier document and find the best matching c-TF-IDF topic representation using cosine similarity.
    topics_text = topic_model.reduce_outliers(docs, topics_text, strategy="embeddings")
    # Then, update the topics to the ones that considered the new data

    print("Finished reducing outliers.")

    progress(0.5, desc= "Creating topic representations")
    print("Create LLM topic labels:", "No")
    representation_model = create_representation_model("No", llm_config, hf_model_name, hf_model_file, chosen_start_tag, low_resource_mode) 
    topic_model.update_topics(docs, topics=topics_text, vectorizer_model=vectoriser_model, representation_model=representation_model)

    topic_dets = topic_model.get_topic_info()

    # Replace original labels with LLM labels
    if "LLM" in topic_model.get_topic_info().columns:
        llm_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["LLM"].values()]
        topic_model.set_topic_labels(llm_labels)
    else:
        topic_model.set_topic_labels(list(topic_dets["Name"]))

    # Outputs   
    output_list, output_text = save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model)

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds"
    print(time_out)
    
    return output_text, output_list, embeddings_out

def represent_topics(topic_model, docs, embeddings_out, data_file_name_no_ext, low_resource_mode, save_topic_model, progress=gr.Progress()):
    #from funcs.prompts import capybara_prompt, capybara_start, open_hermes_prompt, open_hermes_start, stablelm_prompt, stablelm_start
    from funcs.representation_model import create_representation_model, llm_config, chosen_start_tag

    output_list = []

    all_tic = time.perf_counter()

    vectoriser_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.1)

    topics_text, probs = topic_model.fit_transform(docs, embeddings_out)

    topic_dets = topic_model.get_topic_info()

    progress(0.2, desc= "Creating topic representations")
    print("Create LLM topic labels:", "Yes")
    representation_model = create_representation_model("Yes", llm_config, hf_model_name, hf_model_file, chosen_start_tag, low_resource_mode)  

    topic_model.update_topics(docs, topics=topics_text, vectorizer_model=vectoriser_model, representation_model=representation_model)

    # Replace original labels with LLM labels
    if "LLM" in topic_model.get_topic_info().columns:
        llm_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["LLM"].values()]
        topic_model.set_topic_labels(llm_labels)

        with open('llm_topic_list.csv', 'w') as file:
            for item in llm_labels:
                file.write(f"{item}\n")
        output_list.append('llm_topic_list.csv')
    else:
        topic_model.set_topic_labels(list(topic_dets["Name"]))

    

    # Outputs   
    output_list, output_text = save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model)

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds"
    print(time_out)

    return output_text, output_list, embeddings_out

def visualise_topics(topic_model, docs, data_file_name_no_ext, low_resource_mode,  embeddings_out, label_list, sample_prop, visualisation_type_radio, progress=gr.Progress()):
    output_list = []
    vis_tic = time.perf_counter()

    from funcs.bertopic_vis_documents import visualize_documents_custom

    topic_dets = topic_model.get_topic_info()

    # Replace original labels with LLM labels
    if "LLM" in topic_model.get_topic_info().columns:
        llm_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["LLM"].values()]
        topic_model.set_topic_labels(llm_labels)
    else:
        topic_model.set_topic_labels(list(topic_dets["Name"]))

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
        topics_vis = visualize_documents_custom(topic_model, docs, hover_labels = label_list, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True, sample = sample_prop)

        topics_vis_name = data_file_name_no_ext + '_' + 'visualisation_' + today_rev + '.html'
        topics_vis.write_html(topics_vis_name)
        output_list.append(topics_vis_name)

    elif visualisation_type_radio == "Hierarchical view":
        hierarchical_topics = topic_model.hierarchical_topics(docs)
        topics_vis = topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings, sample = sample_prop)
        topics_vis_2 = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

        topics_vis_name = data_file_name_no_ext + '_' + 'vis_hierarchy_topic_doc_' + today_rev + '.html'
        topics_vis.write_html(topics_vis_name)
        output_list.append(topics_vis_name)

        topics_vis_2_name = data_file_name_no_ext + '_' + 'vis_hierarchy_' + today_rev + '.html'
        topics_vis_2.write_html(topics_vis_2_name)
        output_list.append(topics_vis_2_name)

        # Save new hierarchical topic model to file
        import pandas as pd
        hierarchical_topics_name = data_file_name_no_ext + '_' + 'vis_hierarchy_topics' + today_rev + '.csv'
        hierarchical_topics.to_csv(hierarchical_topics_name)
        output_list.append(hierarchical_topics_name)
        #output_list, output_text = save_topic_outputs(topic_model, data_file_name_no_ext, output_list, docs, save_topic_model)

    

    all_toc = time.perf_counter()
    time_out = f"Creating visualisation took {all_toc - vis_tic:0.1f} seconds"
    print(time_out)

    return time_out, output_list, topics_vis, embeddings_out

def save_as_pytorch_model(topic_model, docs, data_file_name_no_ext , progress=gr.Progress()):
    output_list = []

    topic_model_save_name_folder = "output_model/" + data_file_name_no_ext + "_topics_" + today_rev# + ".safetensors"
    topic_model_save_name_zip = topic_model_save_name_folder + ".zip"

    # Clear folder before replacing files
    delete_files_in_folder(topic_model_save_name_folder)

    topic_model.save(topic_model_save_name_folder, serialization='pytorch', save_embedding_model=True, save_ctfidf=False)

    # Zip file example
    
    zip_folder(topic_model_save_name_folder, topic_model_save_name_zip)
    output_list.append(topic_model_save_name_zip)

# Gradio app

block = gr.Blocks(theme = gr.themes.Base())

with block:

    data_state = gr.State(pd.DataFrame())
    embeddings_state = gr.State(np.array([]))
    topic_model_state = gr.State()
    docs_state = gr.State()
    data_file_name_no_ext_state = gr.State()
    label_list_state = gr.State()
 
    gr.Markdown(
    """
    # Topic modeller
    Generate topics from open text in tabular data. Upload a file (csv, xlsx, or parquet), then specify the open text column that you want to use to generate topics, and another for labels in the visualisation. If you have an embeddings .npz file of the text made using the 'BAAI/bge-small-en-v1.5' model, you can load this in at the same time to skip the first modelling step. If you have a pre-defined list of topics, you can upload this as a csv file under 'I have my own list of topics...'. Further configuration options are available under the 'Options' tab.

    Suggested test dataset: https://huggingface.co/datasets/rag-datasets/mini_wikipedia/tree/main/data (passages.parquet)
    """)    
          
    with gr.Tab("Load files and find topics"):
        with gr.Accordion("Load data file", open = True):
            in_files = gr.File(label="Input text from file", file_count="multiple")
            with gr.Row():
                in_colnames = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column to find topics (first will be chosen if multiple selected).")
                in_label = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column for labelling documents in the output visualisation.")

        with gr.Accordion("I have my own list of topics (zero shot topic modelling).", open = False):
            candidate_topics = gr.File(label="Input topics from file (csv). File should have at least one column with a header and topic keywords in cells below. Topics will be taken from the first column of the file. Currently not compatible with low-resource embeddings.")
            zero_shot_similarity = gr.Slider(minimum = 0.5, maximum = 1, value = 0.65, step = 0.001, label = "Minimum similarity value for document to be assigned to zero-shot topic.")

        with gr.Row():
            min_docs_slider = gr.Slider(minimum = 2, maximum = 1000, value = 15, step = 1, label = "Minimum number of similar documents needed to make a topic.")
            max_topics_slider = gr.Slider(minimum = 2, maximum = 500, value = 10, step = 1, label = "Maximum number of topics")

        with gr.Row():
            topics_btn = gr.Button("Extract topics")
            
        with gr.Row():
            output_single_text = gr.Textbox(label="Output topics")
            output_file = gr.File(label="Output file")

        with gr.Accordion("Post processing options.", open = True):
            with gr.Row():
                reduce_outliers_btn = gr.Button("Reduce outliers")
                represent_llm_btn = gr.Button("Generate topic labels with LLMs")

        #logs = gr.Textbox(label="Processing logs.")
        
      

    with gr.Tab("Visualise"):
        
        sample_slide = gr.Slider(minimum = 0.01, maximum = 1, value = 0.1, step = 0.01, label = "Proportion of data points to show on output visualisation.")
        visualisation_type_radio = gr.Radio(choices=["Topic document graph", "Hierarchical view"])
        plot_btn = gr.Button("Visualise topic model")
        out_plot_file = gr.File(label="Output plots to file", file_count="multiple")
        plot = gr.Plot(label="Visualise your topics here. Go to the 'Options' tab to enable.")
    
    with gr.Tab("Options"):
        with gr.Accordion("Data load and processing options", open = True):
            with gr.Row():
                anonymise_drop = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Anonymise data on file load. Names and other details are replaced with tags e.g. '<person>'.")
                embedding_super_compress = gr.Dropdown(label = "Round embeddings to three dp for smaller files with less accuracy.", value="No", choices=["Yes", "No"])
                #create_llm_topic_labels = gr.Dropdown(label = "Create topic labels based on LLMs.", value="No", choices=["Yes", "No"])
            with gr.Row():
                low_resource_mode_opt = gr.Dropdown(label = "Use low resource embeddings and processing.", value="No", choices=["Yes", "No"])
                return_intermediate_files = gr.Dropdown(label = "Return intermediate processing files from file preparation. Files can be loaded in to save processing time in future.", value="Yes", choices=["Yes", "No"])
                save_topic_model = gr.Dropdown(label = "Save topic model to file.", value="Yes", choices=["Yes", "No"])

    # Update column names dropdown when file uploaded
    in_files.upload(fn=put_columns_in_df, inputs=[in_files], outputs=[in_colnames, in_label, data_state, embeddings_state, output_single_text, topic_model_state])    
    in_colnames.change(dummy_function, in_colnames, None)

    topics_btn.click(fn=extract_topics, inputs=[data_state, in_files, min_docs_slider, in_colnames, max_topics_slider, candidate_topics, in_label, anonymise_drop, return_intermediate_files, embedding_super_compress, low_resource_mode_opt, save_topic_model, embeddings_state, zero_shot_similarity], outputs=[output_single_text, output_file, plot, embeddings_state, data_file_name_no_ext_state, topic_model_state, docs_state, label_list_state], api_name="topics")

    reduce_outliers_btn.click(fn=reduce_outliers, inputs=[topic_model_state, docs_state, embeddings_state, data_file_name_no_ext_state, low_resource_mode_opt], outputs=[output_single_text, output_file, embeddings_state], api_name="reduce_outliers")

    represent_llm_btn.click(fn=represent_topics, inputs=[topic_model_state, docs_state, embeddings_state, data_file_name_no_ext_state, low_resource_mode_opt], outputs=[output_single_text, output_file, embeddings_state], api_name="represent_llm")

    plot_btn.click(fn=visualise_topics, inputs=[topic_model_state, docs_state, data_file_name_no_ext_state, low_resource_mode_opt, embeddings_state, label_list_state, sample_slide, visualisation_type_radio], outputs=[output_single_text, out_plot_file, plot], api_name="plot")

    #block.load(read_logs, None, logs, every=5)

block.queue().launch(debug=True)#, server_name="0.0.0.0", ssl_verify=False, server_port=7860)






