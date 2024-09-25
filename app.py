import os
import gradio as gr
import pandas as pd
import numpy as np

from funcs.topic_core_funcs import pre_clean, optimise_zero_shot, extract_topics, reduce_outliers, represent_topics, visualise_topics, save_as_pytorch_model, change_default_vis_col
from funcs.helper_functions import initial_file_load, custom_regex_load, ensure_output_folder_exists, output_folder, get_connection_params, get_or_create_env_var
from funcs.embeddings import make_or_load_embeddings
from sklearn.feature_extraction.text import CountVectorizer
from funcs.auth import authenticate_user, download_file_from_s3

min_word_occurence_slider_default = 0.01
max_word_occurence_slider_default = 0.95

ensure_output_folder_exists()

# Gradio app

block = gr.Blocks(theme = gr.themes.Base())

with block:

    original_data_state  = gr.State(pd.DataFrame())
    data_state = gr.State(pd.DataFrame())
    embeddings_state = gr.State(np.array([]))
    embeddings_type_state = gr.State("")
    topic_model_state = gr.State()
    assigned_topics_state = gr.State([])
    custom_regex_state = gr.State(pd.DataFrame())
    docs_state = gr.State()
    data_file_name_no_ext_state = gr.State()
    label_list_state = gr.State(pd.DataFrame())
    vectoriser_state = gr.State(CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_word_occurence_slider_default, max_df=max_word_occurence_slider_default))

    session_hash_state = gr.State("")
    s3_output_folder_state = gr.State("")
 
    gr.Markdown(
    """
    # Topic modeller
    Generate topics from open text in tabular data, based on [BERTopic](https://maartengr.github.io/BERTopic/). Upload a data file (csv, xlsx, or parquet), then specify the open text column that you want to use to generate topics. Click 'Extract topics' after you have selected the minimum similar documents per topic and maximum total topics. Duplicate this space, or clone to your computer to avoid queues here!
    
    Uses fast TF-IDF-based embeddings by default, which are fast but does not lead to high quality clusering. Change to higher quality [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) model embeddings (512 dimensions) for better results but slower processing time. If you have an embeddings .npz file previously made using this model, you can load this in at the same time to skip the first modelling step. If you have a pre-defined list of topics for zero-shot modelling, you can upload this as a csv file under 'I have my own list of topics...'. Further configuration options are available such as maximum topics allowed, minimum documents per topic etc.. Topic representation with LLMs currently based on [Phi-3.1-mini-128k-instruct-GGUF](https://huggingface.co/bartowski/Phi-3.1-mini-128k-instruct-GGUF), which is quite slow on CPU, so use a GPU-enabled computer if possible, building from the requirements_gpu.txt file in the base folder.

    For small datasets, consider breaking up your text into sentences under 'Clean data' -> 'Split open text...' before topic modelling.

    I suggest [Wikipedia mini dataset](https://huggingface.co/datasets/rag-datasets/mini_wikipedia/tree/main/data) for testing the tool here, choose the passages.parquet file for download.
    """)    
          
    with gr.Tab("Load files and find topics"):
        with gr.Accordion("Load data file", open = True):
            in_files = gr.File(label="Input text from file", file_count="multiple")
            with gr.Row():
                in_colnames = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column to find topics (first will be chosen if multiple selected).")                

        with gr.Accordion("Clean data", open = False):
            with gr.Row():
                clean_text = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Remove html, URLs, non-ASCII, multiple digits, emails, postcodes (UK).")
                drop_duplicate_text = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Remove duplicate text, drop < 50 character strings.")
                anonymise_drop = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Anonymise data on file load. Personal details are redacted - not 100% effective and slow!")
                #with gr.Row():
                split_sentence_drop = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Split text into sentences. Useful for small datasets.")
                #additional_custom_delimiters_drop = gr.Dropdown(choices=["and", ",", "as well as", "also"], multiselect=True, label="Additional custom delimiters to split sentences.")
                min_sentence_length_num = gr.Number(value=5, label="Min char length of split sentences")
            
            with gr.Row():
                custom_regex = gr.UploadButton(label="Import custom regex removal file", file_count="multiple")    
                gr.Markdown("""Import custom regex - csv table with one column of regex patterns with no header. Strings matching this pattern will be removed. Example pattern: (?i)roosevelt for case insensitive removal of this term.""")
                custom_regex_text = gr.Textbox(label="Custom regex load status")
            clean_btn = gr.Button("Clean data")

        with gr.Accordion("I have my own list of topics (zero shot topic modelling).", open = False):
            candidate_topics = gr.File(label="Input topics from file (csv). File should have at least one column with a header and topic keywords in cells below. Topics will be taken from the first column of the file. Currently not compatible with low-resource embeddings.")
            
            with gr.Row():
                zero_shot_similarity = gr.Slider(minimum = 0.2, maximum = 1, value = 0.55, step = 0.001, label = "Minimum similarity value for document to be assigned to zero-shot topic. You may need to set this very low to get documents assigned to your topics!", scale=2)
                zero_shot_optimiser_btn = gr.Button("Optimise settings to keep only zero-shot topics", scale=1)

        with gr.Row():
            with gr.Accordion("Topic modelling settings - change documents per topic, max topics, frequency of terms", open = False):
                
                with gr.Row():
                    min_docs_slider = gr.Slider(minimum = 2, maximum = 1000, value = 5, step = 1, label = "Minimum number of similar documents needed to make a topic.")
                    max_topics_slider = gr.Slider(minimum = 0, maximum = 500, value = 0, step = 1, label = "Maximum number of topics. If set to 0, then will choose topics to merge automatically.")
                with gr.Row():
                    min_word_occurence_slider = gr.Slider(minimum = 0.001, maximum = 0.9, value = min_word_occurence_slider_default, step = 0.001, label = "Keep terms that appear in this minimum proportion of documents. Avoids creating topics with very uncommon words.")
                    max_word_occurence_slider = gr.Slider(minimum = 0.1, maximum = 1.0, value =max_word_occurence_slider_default, step = 0.01, label = "Keep terms that appear in less than this maximum proportion of documents. Avoids very common words in topic names.")

            quality_mode_drop = gr.Dropdown(label = "Use high-quality transformers-based embeddings (slower)", value="No", choices=["Yes", "No"])

        with gr.Row():
            topics_btn = gr.Button("Extract topics", variant="primary")
            
        with gr.Row():
            output_single_text = gr.Textbox(label="Output topics")
            output_file = gr.File(label="Output file")

        with gr.Accordion("Post processing options.", open = True):
            with gr.Row():
                representation_type =  gr.Dropdown(label = "Method for generating new topic labels", value="Default", choices=["Default", "MMR", "KeyBERT", "LLM"]) 
                represent_llm_btn = gr.Button("Change topic labels")
            with gr.Row():
                reduce_outliers_btn = gr.Button("Reduce outliers (will create new topic labels)")
                save_pytorch_btn = gr.Button("Save model in Pytorch format")
                
    with gr.Tab("Visualise"):
        with gr.Row():
            visualisation_type_radio = gr.Radio(label="Visualisation type", choices=["Topic document graph", "Hierarchical view"], value="Topic document graph")
            in_label = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column for labelling documents in output visualisations.")
        sample_slide = gr.Slider(minimum = 0.01, maximum = 1, value = 0.1, step = 0.01, label = "Proportion of data points to show on output visualisations.")
        legend_label = gr.Textbox(label="Custom legend column (optional, any column from the topic details output)", visible=False)
            
        plot_btn = gr.Button("Visualise topic model")
        with gr.Row():
            vis_output_single_text = gr.Textbox(label="Visualisation output text")
            out_plot_file = gr.File(label="Output plots to file", file_count="multiple")
        plot = gr.Plot(label="Visualise your topics here.")
        plot_2 = gr.Plot(label="Visualise your topics here.")
    
    with gr.Tab("Options"):
        with gr.Accordion("Data load and processing options", open = True):
            with gr.Row():
                seed_number = gr.Number(label="Random seed to use in processing", minimum=0, step=1, value=42, precision=0)
                calc_probs = gr.Dropdown(label="Calculate all topic probabilities", value="No", choices=["Yes", "No"])
            with gr.Row():
                embedding_super_compress = gr.Dropdown(label = "Round embeddings to three dp: smaller files but lower quality.", value="No", choices=["Yes", "No"])
                return_intermediate_files = gr.Dropdown(label = "Return intermediate processing files from file preparation.", value="Yes", choices=["Yes", "No"])
                save_topic_model = gr.Dropdown(label = "Save topic model to BERTopic format pkl file.", value="No", choices=["Yes", "No"])

    # Load in data. Update column names dropdown when file uploaded
    in_files.upload(fn=initial_file_load, inputs=[in_files], outputs=[in_colnames, in_label, data_state, output_single_text, topic_model_state, embeddings_state, data_file_name_no_ext_state, label_list_state, original_data_state])

    # When topic modelling column is chosen, change the default visualisation column to the same
    in_colnames.change(fn=change_default_vis_col, inputs=[in_colnames],outputs=[in_label])

    # Clean data
    custom_regex.upload(fn=custom_regex_load, inputs=[custom_regex], outputs=[custom_regex_text, custom_regex_state])
    clean_btn.click(fn=pre_clean, inputs=[data_state, in_colnames, data_file_name_no_ext_state, custom_regex_state, clean_text, drop_duplicate_text, anonymise_drop, split_sentence_drop, min_sentence_length_num, embeddings_state], outputs=[output_single_text, output_file, data_state, data_file_name_no_ext_state, embeddings_state], api_name="clean")

    # Optimise for keeping only zero-shot topics
    zero_shot_optimiser_btn.click(fn=optimise_zero_shot, outputs=[quality_mode_drop, min_docs_slider, max_topics_slider, min_word_occurence_slider, max_word_occurence_slider, zero_shot_similarity])

    # Extract topics
    topics_btn.click(fn=extract_topics, inputs=[data_state, in_files, min_docs_slider, in_colnames, max_topics_slider, candidate_topics, data_file_name_no_ext_state, label_list_state, return_intermediate_files, embedding_super_compress, quality_mode_drop, save_topic_model, embeddings_state, embeddings_type_state, zero_shot_similarity, calc_probs, vectoriser_state, min_word_occurence_slider, max_word_occurence_slider, split_sentence_drop, seed_number], outputs=[output_single_text, output_file, embeddings_state, embeddings_type_state, data_file_name_no_ext_state, topic_model_state, docs_state, vectoriser_state, assigned_topics_state], api_name="topics")

    # Reduce outliers
    reduce_outliers_btn.click(fn=reduce_outliers, inputs=[topic_model_state, docs_state, embeddings_state, data_file_name_no_ext_state, assigned_topics_state, vectoriser_state, save_topic_model, split_sentence_drop, data_state], outputs=[output_single_text, output_file, topic_model_state], api_name="reduce_outliers")

    # Re-represent topic labels
    represent_llm_btn.click(fn=represent_topics, inputs=[topic_model_state, docs_state, data_file_name_no_ext_state, quality_mode_drop, save_topic_model, representation_type, vectoriser_state, split_sentence_drop, data_state], outputs=[output_single_text, output_file, topic_model_state], api_name="represent_llm")

    # Save in Pytorch format
    save_pytorch_btn.click(fn=save_as_pytorch_model, inputs=[topic_model_state, data_file_name_no_ext_state], outputs=[output_single_text, output_file], api_name="pytorch_save")

    # Visualise topics
    plot_btn.click(fn=visualise_topics, inputs=[topic_model_state, data_state, data_file_name_no_ext_state, quality_mode_drop, embeddings_state, in_label, in_colnames, legend_label, sample_slide, visualisation_type_radio, seed_number], outputs=[vis_output_single_text, out_plot_file, plot, plot_2], api_name="plot")

    # Get session hash from connection parameters
    block.load(get_connection_params, inputs=None, outputs=[session_hash_state, s3_output_folder_state])

COGNITO_AUTH = get_or_create_env_var('COGNITO_AUTH', '0')
print(f'The value of COGNITO_AUTH is {COGNITO_AUTH}')


if __name__ == "__main__":
    if os.environ['COGNITO_AUTH'] == "1":
        block.queue().launch(show_error=True, auth=authenticate_user)
    else:
        block.queue().launch(show_error=True, inbrowser=True)