import argparse
import pandas as pd
import numpy as np
from funcs.topic_core_funcs import pre_clean, extract_topics
from funcs.helper_functions import custom_regex_load, initial_file_load, output_folder
from sklearn.feature_extraction.text import CountVectorizer

print("Output folder:", output_folder)

def main():

    parser = argparse.ArgumentParser(description="Run pre_clean and extract_topics from command line.")
    
    # Arguments for pre_clean
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file (csv, xlsx, or parquet).')
    parser.add_argument('--in_colnames', type=str, required=True, help='Column name to find topics.')
    parser.add_argument('--custom_regex_file', type=str, help='Path to custom regex removal file.', default=None)
    parser.add_argument('--clean_text', type=str, choices=['Yes', 'No'], default='No', help='Remove html, URLs, etc.')
    parser.add_argument('--drop_duplicate_text', type=str, choices=['Yes', 'No'], default='No', help='Remove duplicate text.')
    parser.add_argument('--anonymise_drop', type=str, choices=['Yes', 'No'], default='No', help='Redact personal information.')
    parser.add_argument('--split_sentence_drop', type=str, choices=['Yes', 'No'], default='No', help='Split text into sentences.')
    parser.add_argument('--min_sentence_length_num', type=int, default=5, help='Min char length of split sentences.')

    parser.add_argument('--min_docs_slider', type=int, default=5, help='Minimum number of similar documents needed to make a topic.')
    parser.add_argument('--max_topics_slider', type=int, default=0, help='Maximum number of topics.')
    parser.add_argument('--min_word_occurence_slider', type=float, default=0.01, help='Minimum word occurrence proportion.')
    parser.add_argument('--max_word_occurence_slider', type=float, default=0.95, help='Maximum word occurrence proportion.')
    parser.add_argument('--embeddings_high_quality_mode', type=str, choices=['Yes', 'No'], default='No', help='Use high-quality embeddings.')
    parser.add_argument('--zero_shot_similarity', type=float, default=0.55, help='Minimum similarity for zero-shot topic assignment.')
    parser.add_argument('--seed_number', type=int, default=42, help='Random seed for processing.')
    parser.add_argument('--return_only_embeddings_drop', type=str, default="No", help='Return only embeddings from the function, do not assign topics.')
    parser.add_argument('--output_folder', type=str, default=output_folder, help='Output folder for results.')

    args = parser.parse_args()

    # Load data
    #data = pd.read_csv(args.data_file) if args.data_file.endswith('.csv') else pd.read_excel(args.data_file)
    #custom_regex = pd.read_csv(args.custom_regex_file) if args.custom_regex_file else pd.DataFrame()

    in_colnames_all, in_label, data, output_single_text, topic_model_state, embeddings_state, data_file_name_no_ext, label_list_state, original_data_state = initial_file_load(args.data_file)
    custom_regex_output_text, custom_regex = custom_regex_load(args.custom_regex_file) if args.custom_regex_file else pd.DataFrame()

    print("data_file_name_no_ext:", data_file_name_no_ext)

    # Pre-clean data
    pre_clean_output = pre_clean(
        data=data,
        in_colnames=[args.in_colnames],
        data_file_name_no_ext=data_file_name_no_ext,
        custom_regex=custom_regex,
        clean_text=args.clean_text,
        drop_duplicate_text=args.drop_duplicate_text,
        anonymise_drop=args.anonymise_drop,
        sentence_split_drop=args.split_sentence_drop,
        min_sentence_length=args.min_sentence_length_num,
        embeddings_state=np.array([]),
        output_folder=output_folder
    )

    # Extract topics
    extract_topics_output = extract_topics(
        data=pre_clean_output[2],
        in_files=args.data_file,
        min_docs_slider=args.min_docs_slider,
        in_colnames=[args.in_colnames],
        max_topics_slider=args.max_topics_slider,
        candidate_topics=[],
        data_file_name_no_ext=data_file_name_no_ext,
        custom_labels_df=pd.DataFrame(),
        return_intermediate_files='Yes',
        embeddings_super_compress='No',
        high_quality_mode=args.embeddings_high_quality_mode,
        save_topic_model='No',
        embeddings_out=np.array([]),
        embeddings_type_state='',
        zero_shot_similarity=args.zero_shot_similarity,
        calc_probs='No',
        vectoriser_state=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=args.min_word_occurence_slider, max_df=args.max_word_occurence_slider),
        min_word_occurence_slider=args.min_word_occurence_slider,
        max_word_occurence_slider=args.max_word_occurence_slider,
        split_sentence_drop=args.split_sentence_drop,
        random_seed=args.seed_number,
        return_only_embeddings_drop=args.return_only_embeddings_drop,
        output_folder=output_folder
    )

if __name__ == "__main__":
    main()