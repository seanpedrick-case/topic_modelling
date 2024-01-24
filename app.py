import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
import time
#from sklearn.cluster import KMeans
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


# Load embeddings
#embedding_model_name = "BAAI/bge-small-en-v1.5"
#embedding_model = SentenceTransformer(embedding_model_name)

# Pinning a Jina revision for security purposes: https://www.baseten.co/blog/pinning-ml-model-revisions-for-compatibility-and-security/
# Save Jina model locally as described here: https://huggingface.co/jinaai/jina-embeddings-v2-base-en/discussions/29
embeddings_name = "jinaai/jina-embeddings-v2-small-en"
local_embeddings_location = "model/jina/"
revision_choice = "b811f03af3d4d7ea72a7c25c802b21fc675a5d99"

if low_resource_mode == "No":
    try:
        embedding_model = AutoModel.from_pretrained(local_embeddings_location, revision = revision_choice, trust_remote_code=True,local_files_only=True, device_map="auto")
    except:
        embedding_model = AutoModel.from_pretrained(embeddings_name, revision = revision_choice, trust_remote_code=True, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-small-en")

    embedding_model_pipe = pipeline("feature-extraction", model=embedding_model, tokenizer=tokenizer)

elif low_resource_mode == "Yes":
    embedding_model_pipe = make_pipeline(
                TfidfVectorizer(),
                TruncatedSVD(2) # 100 # set to 2 to be compatible with zero shot topics - can't be higher than number of topics
                )

# Model used for representing topics
hf_model_name =  'TheBloke/phi-2-orange-GGUF' #'NousResearch/Nous-Capybara-7B-V1.9-GGUF' # 'second-state/stablelm-2-zephyr-1.6b-GGUF'
hf_model_file =   'phi-2-orange.Q5_K_M.gguf' #'Capybara-7B-V1.9-Q5_K_M.gguf' # 'stablelm-2-zephyr-1_6b-Q5_K_M.gguf'


def extract_topics(in_files, in_file, min_docs_slider, in_colnames, max_topics_slider, candidate_topics, in_label, anonymise_drop, return_intermediate_files, embeddings_super_compress, low_resource_mode, create_llm_topic_labels, save_topic_model, visualise_topics, reduce_outliers):
    
    all_tic = time.perf_counter()

    output_list = []
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
        anon_tic = time.perf_counter()
        time_out = f"Creating visualisation took {all_toc - vis_tic:0.1f} seconds"
        in_files_anon_col, anonymisation_success = anon.anonymise_script(in_files, in_colnames_list_first, anon_strat="replace")
        in_files[in_colnames_list_first] = in_files_anon_col[in_colnames_list_first]
        anonymise_data_name = "anonymised_data.csv"
        in_files.to_csv(anonymise_data_name)
        output_list.append(anonymise_data_name)

        anon_toc = time.perf_counter()
        time_out = f"Anonymising text took {anon_toc - anon_tic:0.1f} seconds"

    docs = list(in_files[in_colnames_list_first].str.lower())
    label_col = in_files[in_label_list_first]

    # Check if embeddings are being loaded in
    ## Load in pre-embedded file if exists
    file_list = [string.name for string in in_file]

    print("Low resource mode: ", low_resource_mode)

    if low_resource_mode == "No":
        print("Using high resource Jina transformer model")
        try:
            embedding_model = AutoModel.from_pretrained(local_embeddings_location, revision = revision_choice, trust_remote_code=True,local_files_only=True, device_map="auto")
        except:
            embedding_model = AutoModel.from_pretrained(embeddings_name, revision = revision_choice, trust_remote_code=True, device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-small-en")

        embedding_model_pipe = pipeline("feature-extraction", model=embedding_model, tokenizer=tokenizer)

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

    embeddings_out, reduced_embeddings = make_or_load_embeddings(docs, file_list, data_file_name_no_ext, embedding_model, return_intermediate_files, embeddings_super_compress, low_resource_mode, create_llm_topic_labels)

    vectoriser_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.1)
    
    from funcs.prompts import capybara_prompt, capybara_start, open_hermes_prompt, open_hermes_start, stablelm_prompt, stablelm_start
    from funcs.representation_model import create_representation_model, llm_config, chosen_start_tag

    print("Create LLM topic labels:", create_llm_topic_labels)
    representation_model = create_representation_model(create_llm_topic_labels, llm_config, hf_model_name, hf_model_file, chosen_start_tag, low_resource_mode)  


    if not candidate_topics:
        
        # Generate representation model here if topics won't be changed later
        if reduce_outliers == "No":
            topic_model = BERTopic( embedding_model=embedding_model_pipe,
                                    vectorizer_model=vectoriser_model,
                                    umap_model=umap_model,
                                    min_topic_size = min_docs_slider,
                                    nr_topics = max_topics_slider,
                                    representation_model=representation_model,
                                    verbose = True)
        else:
            topic_model = BERTopic( embedding_model=embedding_model_pipe,
                                    vectorizer_model=vectoriser_model,
                                    umap_model=umap_model,
                                    min_topic_size = min_docs_slider,
                                    nr_topics = max_topics_slider,
                                    verbose = True)

        topics_text, probs = topic_model.fit_transform(docs, embeddings_out)   


    # Do this if you have pre-defined topics
    else:
        if low_resource_mode == "Yes":
            error_message = "Zero shot topic modelling currently not compatible with low-resource embeddings. Please change this option to 'No' on the options tab and retry."
            print(error_message)

            return error_message, output_list, None

        zero_shot_topics = read_file(candidate_topics.name)
        zero_shot_topics_lower = list(zero_shot_topics.iloc[:, 0].str.lower())

        # Generate representation model here if topics won't be changed later
        if reduce_outliers == "No":
            topic_model = BERTopic( embedding_model=embedding_model_pipe,
                                    vectorizer_model=vectoriser_model,
                                    umap_model=umap_model,
                                    min_topic_size = min_docs_slider,
                                    nr_topics = max_topics_slider,
                                    zeroshot_topic_list = zero_shot_topics_lower,
                                    zeroshot_min_similarity = 0.5,#0.7,
                                    representation_model=representation_model,
                                    verbose = True)
        else:
            topic_model = BERTopic( embedding_model=embedding_model_pipe,
                                    vectorizer_model=vectoriser_model,
                                    umap_model=umap_model,
                                    min_topic_size = min_docs_slider,
                                    nr_topics = max_topics_slider,
                                    zeroshot_topic_list = zero_shot_topics_lower,
                                    zeroshot_min_similarity = 0.5,#0.7,
                                    verbose = True)
        
        topics_text, probs = topic_model.fit_transform(docs, embeddings_out)

    if not topics_text:
        return "No topics found.", data_file_name, None
        
    else: 
        print("Preparing topic model outputs.")

    # Reduce outliers if required
    if reduce_outliers == "Yes":
        print("Reducing outliers.")
        # Calculate the c-TF-IDF representation for each outlier document and find the best matching c-TF-IDF topic representation using cosine similarity.
        topics_text = topic_model.reduce_outliers(docs, topics_text, strategy="embeddings")
        # Then, update the topics to the ones that considered the new data
        topic_model.update_topics(docs, topics=topics_text, vectorizer_model=vectoriser_model, representation_model=representation_model)
        print("Finished reducing outliers.")

    topic_dets = topic_model.get_topic_info()
    #print(topic_dets.columns)

    if topic_dets.shape[0] == 1:
        topic_det_output_name = "topic_details_" + data_file_name_no_ext + "_" + today_rev + ".csv"
        topic_dets.to_csv(topic_det_output_name)
        output_list.append(topic_det_output_name)

        return "No topics found, original file returned", output_list, None

    # Replace original labels with LLM labels
    if "Mistral" in topic_model.get_topic_info().columns:
        llm_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["Mistral"].values()]
        topic_model.set_topic_labels(llm_labels)
    else:
        topic_model.set_topic_labels(list(topic_dets["Name"]))

    # Outputs
    
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
        topic_model_save_name_folder = "output_model/" + data_file_name_no_ext + "_topics_" + today_rev# + ".safetensors"
        topic_model_save_name_zip = topic_model_save_name_folder + ".zip"

        # Clear folder before replacing files
        delete_files_in_folder(topic_model_save_name_folder)

        topic_model.save(topic_model_save_name_folder, serialization='pytorch', save_embedding_model=True, save_ctfidf=False)

        # Zip file example
        
        zip_folder(topic_model_save_name_folder, topic_model_save_name_zip)
        output_list.append(topic_model_save_name_zip)

    if return_intermediate_files == "Yes":
        print("Saving embeddings to file")
        if low_resource_mode == "Yes":
            embeddings_file_name = data_file_name_no_ext + '_' + 'tfidf_embeddings.npz'
        else:
            embeddings_file_name = data_file_name_no_ext + '_' + 'ai_embeddings.npz'

        np.savez_compressed(embeddings_file_name, embeddings_out)

        output_list.append(embeddings_file_name)

    if visualise_topics == "Yes":
        # Visualise the topics:
        vis_tic = time.perf_counter()
        print("Creating visualisation")
        topics_vis = topic_model.visualize_documents(label_col, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False, custom_labels=True)

        all_toc = time.perf_counter()
        time_out = f"Creating visualisation took {all_toc - vis_tic:0.1f} seconds"
        print(time_out)

        
        time_out = f"All processes took {all_toc - all_tic:0.1f} seconds"
        print(time_out)

        return output_text, output_list, topics_vis

    all_toc = time.perf_counter()
    time_out = f"All processes took {all_toc - all_tic:0.1f} seconds"
    print(time_out)

    return output_text, output_list, None

# , topic_model_save_name

# ## Gradio app - extract topics

block = gr.Blocks(theme = gr.themes.Base())

with block:

    data_state = gr.State(pd.DataFrame())
 
    gr.Markdown(
    """
    # Topic modeller
    Generate topics from open text in tabular data. Upload a file (csv, xlsx, or parquet), then specify the columns that you want to use to generate topics and use for labels in the visualisation. If you have an embeddings .npz file of the text made using the 'jina-embeddings-v2-small-en' model, you can load this in at the same time to skip the first modelling step. If you have a pre-defined list of topics, you can upload this as a csv file under 'I have my own list of topics...'. Further configuration options are available under the 'Options' tab.
    """)    
          
    with gr.Tab("Load files and find topics"):
        with gr.Accordion("Load data file", open = True):
            in_files = gr.File(label="Input text from file", file_count="multiple")
            with gr.Row():
                in_colnames = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column to find topics (first will be chosen if multiple selected).")
                in_label = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column to for labelling documents in the output visualisation.")

        with gr.Accordion("I have my own list of topics (zero shot topic modelling).", open = False):
            candidate_topics = gr.File(label="Input topics from file (csv). File should have at least one column with a header and topic keywords in cells below. Topics will be taken from the first column of the file. Currently not compatible with low-resource embeddings.")
            
        with gr.Row():
            min_docs_slider = gr.Slider(minimum = 2, maximum = 1000, value = 15, step = 1, label = "Minimum number of similar documents needed to make a topic.")
            max_topics_slider = gr.Slider(minimum = 2, maximum = 500, value = 10, step = 1, label = "Maximum number of topics")

        with gr.Row():
            topics_btn = gr.Button("Extract topics")
            
        with gr.Row():
            output_single_text = gr.Textbox(label="Output example (first example in dataset)")
            output_file = gr.File(label="Output file")

        plot = gr.Plot(label="Visualise your topics here. Go to the 'Options' tab to enable.")
    
    with gr.Tab("Options"):
        with gr.Accordion("Data load and processing options", open = True):
            with gr.Row():
                anonymise_drop = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Anonymise data on file load. Names and other details are replaced with tags e.g. '<person>'.")
                return_intermediate_files = gr.Dropdown(label = "Return intermediate processing files from file preparation. Files can be loaded in to save processing time in future.", value="Yes", choices=["Yes", "No"])
                embedding_super_compress = gr.Dropdown(label = "Round embeddings to three dp for smaller files with less accuracy.", value="No", choices=["Yes", "No"])
            with gr.Row():
                low_resource_mode_opt = gr.Dropdown(label = "Use low resource embeddings and processing.", value="No", choices=["Yes", "No"])
                create_llm_topic_labels = gr.Dropdown(label = "Create LLM-generated topic labels.", value="No", choices=["Yes", "No"])
                reduce_outliers = gr.Dropdown(label = "Reduce outliers by selecting closest topic.", value="No", choices=["Yes", "No"])
            with gr.Row():
                save_topic_model = gr.Dropdown(label = "Save topic model to file.", value="Yes", choices=["Yes", "No"])
                visualise_topics = gr.Dropdown(label = "Create a visualisation to map topics.", value="No", choices=["Yes", "No"])

    # Update column names dropdown when file uploaded
    in_files.upload(fn=put_columns_in_df, inputs=[in_files], outputs=[in_colnames, in_label, data_state])    
    in_colnames.change(dummy_function, in_colnames, None)

    topics_btn.click(fn=extract_topics, inputs=[data_state, in_files, min_docs_slider, in_colnames, max_topics_slider, candidate_topics, in_label, anonymise_drop, return_intermediate_files, embedding_super_compress, low_resource_mode_opt, create_llm_topic_labels, save_topic_model, visualise_topics, reduce_outliers], outputs=[output_single_text, output_file, plot], api_name="topics")

block.queue().launch(debug=True)#, server_name="0.0.0.0", ssl_verify=False, server_port=7860)

