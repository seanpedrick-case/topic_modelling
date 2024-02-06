# Dendrograms will not work with the latest version of scipy (1.12.0), so run the following here/in your environment if you come across issues
# import os
# os.system("pip install scipy==1.11.4")

import gradio as gr
import pandas as pd
import numpy as np

from funcs.topic_core_funcs import pre_clean, extract_topics, reduce_outliers, represent_topics, visualise_topics, save_as_pytorch_model
from funcs.helper_functions import dummy_function, initial_file_load, custom_regex_load
from sklearn.feature_extraction.text import CountVectorizer

# Gradio app

block = gr.Blocks(theme = gr.themes.Base())

with block:

    data_state = gr.State(pd.DataFrame())
    embeddings_state = gr.State(np.array([]))
    topic_model_state = gr.State()
    custom_regex_state = gr.State(pd.DataFrame())
    docs_state = gr.State()
    data_file_name_no_ext_state = gr.State()
    label_list_state = gr.State(pd.DataFrame())
    vectoriser_state = gr.State(CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.1, max_df=0.95))
 
    gr.Markdown(
    """
    # Topic modeller
    Generate topics from open text in tabular data, based on [BERTopic](https://maartengr.github.io/BERTopic/). Upload a data file (csv, xlsx, or parquet), then specify the open text column that you want to use to generate topics. Click 'Extract topics' after you have selected the minimum similar documents per topic and maximum total topics. Duplicate this space, or clone to your computer to avoid queues here!
    
    Uses fast TF-IDF-based embeddings by default, which are fast but not very performant in terms of cluster. Change to [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) model embeddings on the options page for topics of much higher quality, but slower processing time. If you have an embeddings .npz file previously made using this model, you can load this in at the same time to skip the first modelling step. If you have a pre-defined list of topics for zero-shot modelling, you can upload this as a csv file under 'I have my own list of topics...'. Further configuration options are available under the 'Options' tab. Topic representation with LLMs currently based on [StableLM-2-Zephyr-1.6B-GGUF](https://huggingface.co/second-state/stablelm-2-zephyr-1.6b-GGUF) - this works locally, but unfortunately this doesn't yet seem to work on the Huggingface website, I'm working on it!

    I suggest [Wikipedia mini dataset](https://huggingface.co/datasets/rag-datasets/mini_wikipedia/tree/main/data) for testing the tool here, choose passages.parquet.
    """)    
          
    with gr.Tab("Load files and find topics"):
        with gr.Accordion("Load data file", open = True):
            in_files = gr.File(label="Input text from file", file_count="multiple")
            with gr.Row():
                in_colnames = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column to find topics (first will be chosen if multiple selected).")                

        with gr.Accordion("Clean data", open = False):
            with gr.Row():
                clean_text = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Clean data - remove html, numbers with > 1 digits, emails, postcodes (UK).")
                drop_duplicate_text = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Remove duplicate text, drop < 50 char strings. May make old embedding files incompatible due to differing lengths.")
                anonymise_drop = gr.Dropdown(value = "No", choices=["Yes", "No"], multiselect=False, label="Anonymise data on file load. Personal details are redacted - not 100% effective. This is slow!")
            with gr.Row():
                gr.Markdown("""Import custom regex - csv table with one column of raw text regex patterns with header. Example pattern: r'example'""")
                custom_regex = gr.UploadButton(label="Import custom regex file", file_count="multiple")             
            clean_btn = gr.Button("Clean data")

        with gr.Accordion("I have my own list of topics (zero shot topic modelling).", open = False):
            candidate_topics = gr.File(label="Input topics from file (csv). File should have at least one column with a header and topic keywords in cells below. Topics will be taken from the first column of the file. Currently not compatible with low-resource embeddings.")
            zero_shot_similarity = gr.Slider(minimum = 0.5, maximum = 1, value = 0.65, step = 0.001, label = "Minimum similarity value for document to be assigned to zero-shot topic.")

        with gr.Row():
            min_docs_slider = gr.Slider(minimum = 2, maximum = 1000, value = 5, step = 1, label = "Minimum number of similar documents needed to make a topic.")
            max_topics_slider = gr.Slider(minimum = 2, maximum = 500, value = 50, step = 1, label = "Maximum number of topics")

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
                reduce_outliers_btn = gr.Button("Reduce outliers")
                save_pytorch_btn = gr.Button("Save model in Pytorch format")
                
    with gr.Tab("Visualise"):
        with gr.Row():
            visualisation_type_radio = gr.Radio(label="Visualisation type", choices=["Topic document graph", "Hierarchical view"])
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
                seed_number = gr.Number(label="Random seed to use for dimensionality reduction.", minimum=0, step=1, value=42, precision=0)
                calc_probs = gr.Dropdown(label="Calculate all topic probabilities", value="No", choices=["Yes", "No"])
            with gr.Row():
                low_resource_mode_opt = gr.Dropdown(label = "Use low resource (TF-IDF) embeddings and processing.", value="Yes", choices=["Yes", "No"])
                embedding_super_compress = gr.Dropdown(label = "Round embeddings to three dp for smaller files with less accuracy.", value="No", choices=["Yes", "No"])
            with gr.Row(): 
                return_intermediate_files = gr.Dropdown(label = "Return intermediate processing files from file preparation.", value="Yes", choices=["Yes", "No"])
                save_topic_model = gr.Dropdown(label = "Save topic model to BERTopic format pkl file.", value="No", choices=["Yes", "No"])

    # Load in data. Update column names dropdown when file uploaded
    in_files.upload(fn=initial_file_load, inputs=[in_files], outputs=[in_colnames, in_label, data_state, output_single_text, topic_model_state, embeddings_state, data_file_name_no_ext_state, label_list_state])    
    in_colnames.change(dummy_function, in_colnames, None)

    # Clean data
    custom_regex.upload(fn=custom_regex_load, inputs=[custom_regex], outputs=[custom_regex_state])
    clean_btn.click(fn=pre_clean, inputs=[data_state, in_colnames, data_file_name_no_ext_state, custom_regex_state, clean_text, drop_duplicate_text, anonymise_drop], outputs=[output_single_text, output_file, data_state, data_file_name_no_ext_state], api_name="clean")

    # Extract topics
    topics_btn.click(fn=extract_topics, inputs=[data_state, in_files, min_docs_slider, in_colnames, max_topics_slider, candidate_topics, data_file_name_no_ext_state, label_list_state, return_intermediate_files, embedding_super_compress, low_resource_mode_opt, save_topic_model, embeddings_state, zero_shot_similarity, seed_number, calc_probs, vectoriser_state], outputs=[output_single_text, output_file, embeddings_state, data_file_name_no_ext_state, topic_model_state, docs_state, vectoriser_state], api_name="topics")

    # Reduce outliers
    reduce_outliers_btn.click(fn=reduce_outliers, inputs=[topic_model_state, docs_state, embeddings_state, data_file_name_no_ext_state, save_topic_model], outputs=[output_single_text, output_file, topic_model_state], api_name="reduce_outliers")

    # Re-represent topic labels
    represent_llm_btn.click(fn=represent_topics, inputs=[topic_model_state, docs_state, data_file_name_no_ext_state, low_resource_mode_opt, save_topic_model, representation_type, vectoriser_state], outputs=[output_single_text, output_file, topic_model_state], api_name="represent_llm")

    # Save in Pytorch format
    save_pytorch_btn.click(fn=save_as_pytorch_model, inputs=[topic_model_state, data_file_name_no_ext_state], outputs=[output_single_text, output_file])

    # Visualise topics
    plot_btn.click(fn=visualise_topics, inputs=[topic_model_state, data_state, data_file_name_no_ext_state, low_resource_mode_opt, embeddings_state, in_label, in_colnames, legend_label, sample_slide, visualisation_type_radio, seed_number], outputs=[vis_output_single_text, out_plot_file, plot, plot_2], api_name="plot")

block.queue().launch(debug=True)#, server_name="0.0.0.0", ssl_verify=False, server_port=7860)