import os

#os.environ["TOKENIZERS_PARALLELISM"] = "true"
#os.environ["HF_HOME"] = "/mnt/c/..."
#os.environ["CUDA_PATH"] = "/mnt/c/..."

print(os.environ["HF_HOME"])

import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel
import funcs.anonymiser as anon

from torch import cuda, backends, version

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

from funcs.helper_functions import dummy_function, put_columns_in_df, read_file, get_file_path_end
from funcs.representation_model import representation_model
from funcs.embeddings import make_or_load_embeddings

# Load embeddings
#embedding_model_name = "BAAI/bge-small-en-v1.5"
#embedding_model = SentenceTransformer(embedding_model_name)

# Pinning a Jina revision for security purposes: https://www.baseten.co/blog/pinning-ml-model-revisions-for-compatibility-and-security/
# Save Jina model locally as described here: https://huggingface.co/jinaai/jina-embeddings-v2-base-en/discussions/29
embeddings_name = "jinaai/jina-embeddings-v2-small-en"
local_embeddings_location = "model/jina/"
revision_choice = "b811f03af3d4d7ea72a7c25c802b21fc675a5d99"

try:
    embedding_model = AutoModel.from_pretrained(local_embeddings_location, revision = revision_choice, trust_remote_code=True,local_files_only=True, device_map="auto")
except:
    embedding_model = AutoModel.from_pretrained(embeddings_name, revision = revision_choice, trust_remote_code=True, device_map="auto")



def extract_topics(in_files, in_file, min_docs_slider, in_colnames, max_topics_slider, candidate_topics, in_label, anonymise_drop, return_intermediate_files, embeddings_super_compress, low_resource_mode_opt):
    
    file_list = [string.name for string in in_file]

    data_file_names = [string.lower() for string in file_list if "tokenised" not in string and "npz" not in string.lower() and "gz" not in string.lower()]
    data_file_name = data_file_names[0]
    data_file_name_no_ext = get_file_path_end(data_file_name)

    in_colnames_list_first = in_colnames[0]

    if in_label:
        in_label_list_first = in_label[0]
    else:
        in_label_list_first = in_colnames_list_first
    
    if anonymise_drop == "Yes":
        in_files_anon_col, anonymisation_success = anon.anonymise_script(in_files, in_colnames_list_first, anon_strat="replace")
        in_files[in_colnames_list_first] = in_files_anon_col[in_colnames_list_first]
        in_files.to_csv("anonymised_data.csv")

    docs = list(in_files[in_colnames_list_first].str.lower())
    label_col = in_files[in_label_list_first]

    # Check if embeddings are being loaded in
    ## Load in pre-embedded file if exists
    file_list = [string.name for string in in_file]

    embeddings_out, reduced_embeddings = make_or_load_embeddings(docs, file_list, data_file_name_no_ext, embedding_model, return_intermediate_files, embeddings_super_compress, low_resource_mode_opt)

    # all_lengths = [len(embedding) for embedding in embeddings_out]
    # if len(set(all_lengths)) > 1:
    #     print("Inconsistent lengths found in embeddings_out:", set(all_lengths))
    # else:
    #     print("All lengths are the same.")

    # print("Embeddings type: ", type(embeddings_out))

    # if isinstance(embeddings_out, np.ndarray):
    #     print("my_object is a NumPy ndarray")
    # else:
    #     print("my_object is not a NumPy ndarray")

    # Clustering set to K-means (not used)
    #cluster_model = KMeans(n_clusters=max_topics_slider)

    # Countvectoriser removes stopwords, combines terms up to 2 together:
    if min_docs_slider < 3:
        min_df_val = min_docs_slider
    else:
        min_df_val = 3

    print(min_df_val)

    vectoriser_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.1)


    if not candidate_topics:
        topic_model = BERTopic( embedding_model=embedding_model, 
                                #hdbscan_model=cluster_model,
                                vectorizer_model=vectoriser_model,
                                min_topic_size= min_docs_slider,
                                nr_topics = max_topics_slider,
                                representation_model=representation_model,
                                verbose = True)

        topics_text, probs = topic_model.fit_transform(docs, embeddings_out)   


    # Do this if you have pre-assigned topics
    else:   
        zero_shot_topics_list = read_file(candidate_topics.name)
        zero_shot_topics_list_lower = [x.lower() for x in zero_shot_topics_list]

        print(zero_shot_topics_list_lower)

        topic_model = BERTopic( embedding_model=embedding_model,
                                #hdbscan_model=cluster_model,
                                vectorizer_model=vectoriser_model,
                                min_topic_size = min_docs_slider,
                                nr_topics = max_topics_slider,
                                zeroshot_topic_list = zero_shot_topics_list_lower,
                                zeroshot_min_similarity = 0.7,
                                representation_model=representation_model,
                                verbose = True)
        
        topics_text, probs = topic_model.fit_transform(docs, embeddings_out)

    if not topics_text:
        return "No topics found, original file returned", data_file_name
        
    else: 
        topics_text_out = topics_text
        topics_scores_out = probs

    topic_det_output_name = "topic_details_" + today_rev + ".csv"

    topic_dets = topic_model.get_topic_info()

    topic_dets.to_csv(topic_det_output_name)
    #print(topic_dets)

    doc_det_output_name = "doc_details_" + today_rev + ".csv"
    doc_dets = topic_model.get_document_info(docs)[["Document",	"Topic", "Probability",	"Name", "Representative_document"]]
    doc_dets.to_csv(doc_det_output_name)
    #print(doc_dets)
    
    #print(topic_dets)    
    #topics_text_out_str = ', '.join(list(topic_dets["KeyBERT"]))

    topics_text_out_str = str(topic_dets["KeyBERT"])
    #topics_scores_out_str = str(doc_dets["Probability"][0])
    
    output_text = "Topics: " + topics_text_out_str #+ "\n\nProbability scores: " + topics_scores_out_str

    # Outputs
    embedding_file_name = data_file_name_no_ext + '_' + 'embeddings.npz'
    np.savez_compressed(embedding_file_name, embeddings_out)

    topic_model_save_name = data_file_name_no_ext + "_topics_" + today_rev + ".pkl"
    topic_model.save(topic_model_save_name, serialization='pickle', save_embedding_model=False, save_ctfidf=False)

    # Visualise the topics:
    topics_vis = topic_model.visualize_documents(label_col, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True)
    
    return output_text, [doc_det_output_name, topic_det_output_name, embedding_file_name, topic_model_save_name], topics_vis


# ## Gradio app - extract topics

block = gr.Blocks(theme = gr.themes.Base())

with block:

    data_state = gr.State(pd.DataFrame())
 
    gr.Markdown(
    """
    # Extract topics from text
    Enter open text below to get topics. You can copy and paste text directly, or upload a file and specify the column that you want to topics.
    """)    
   
    #with gr.Accordion("I will copy and paste my open text", open = False):
    #    in_text = gr.Textbox(label="Copy and paste your open text here", lines = 5)
        
    with gr.Tab("Load files and find topics"):
        with gr.Accordion("Load data file", open = True):
            in_files = gr.File(label="Input text from file", file_count="multiple")
            with gr.Row():
                in_colnames = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column to find topics (first will be chosen if multiple selected).")
                in_label = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column to for labelling documents in the output visualisation.")

        with gr.Accordion("I have my own list of topics. File should have at least one column with a header and topic keywords in cells below. Topics will be taken from the first column of the file", open = False):
            candidate_topics = gr.File(label="Input topics from file (csv)")
            
        with gr.Row():
            min_docs_slider = gr.Slider(minimum = 1, maximum = 1000, value = 15, step = 1, label = "Minimum number of documents needed to create topic")
            max_topics_slider = gr.Slider(minimum = 2, maximum = 500, value = 3, step = 1, label = "Maximum number of topics")

        with gr.Row():
            topics_btn = gr.Button("Extract topics")
            
        with gr.Row():
            output_single_text = gr.Textbox(label="Output example (first example in dataset)")
            output_file = gr.File(label="Output file")

        plot = gr.Plot(label="Visualise your topics here:")
    
    with gr.Tab("Load and data processing options"):
        with gr.Accordion("Process data on load", open = True):
            anonymise_drop = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Anonymise data on file load.")
            return_intermediate_files = gr.Dropdown(label = "Return intermediate processing files from file preparation. Files can be loaded in to save processing time in future.", value="No", choices=["Yes", "No"])
            embedding_super_compress = gr.Dropdown(label = "Round embeddings to three dp for smaller files with less accuracy.", value="No", choices=["Yes", "No"])
            low_resource_mode_opt = gr.Dropdown(label = "Low resource mode (non-AI embeddings, no LLM-generated topic names).", value=low_resource_mode, choices=["Yes", "No"])
                

    # Update column names dropdown when file uploaded
    in_files.upload(fn=put_columns_in_df, inputs=[in_files], outputs=[in_colnames, in_label, data_state])    
    in_colnames.change(dummy_function, in_colnames, None)

    topics_btn.click(fn=extract_topics, inputs=[data_state, in_files, min_docs_slider, in_colnames, max_topics_slider, candidate_topics, in_label, anonymise_drop, return_intermediate_files, embedding_super_compress, low_resource_mode_opt], outputs=[output_single_text, output_file, plot], api_name="topics")

block.queue().launch(debug=True)#, server_name="0.0.0.0", ssl_verify=False, server_port=7860)

