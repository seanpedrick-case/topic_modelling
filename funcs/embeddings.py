import time
import numpy as np
from torch import cuda
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

if cuda.is_available():
    torch_device = "gpu"
else: 
    torch_device =  "cpu"

def make_or_load_embeddings(docs, file_list, data_file_name_no_ext, embedding_model, return_intermediate_files, embeddings_super_compress, low_resource_mode_opt, reduce_embeddings="Yes"):

    embeddings_file_names = [string.lower() for string in file_list if "embedding" in string.lower()]  

    if embeddings_file_names:
        print("Loading embeddings from file.")
        embeddings_out = np.load(embeddings_file_names[0])['arr_0']

        # If embedding files have 'super_compress' in the title, they have been multiplied by 100 before save
        if "compress" in embeddings_file_names[0]:
            embeddings_out /= 100

        # print("embeddings loaded: ", embeddings_out)

    if not embeddings_file_names:
        tic = time.perf_counter()
        print("Starting to embed documents.")

        # Custom model
        # If on CPU, don't resort to embedding models
        if low_resource_mode_opt == "Yes":
            print("Creating simplified 'sparse' embeddings based on TfIDF")
            embedding_model = make_pipeline(
            TfidfVectorizer(),
            TruncatedSVD(100)
            )

            # Fit the pipeline to the text data
            embedding_model.fit(docs)

            # Transform text data to embeddings
            embeddings_out = embedding_model.transform(docs)

            #embeddings_out = embedding_model.encode(sentences=docs, show_progress_bar = True, batch_size = 32)

        elif low_resource_mode_opt == "No":
            print("Creating dense embeddings based on transformers model")

            #print("Embedding model is: ", embedding_model)

            embeddings_out = embedding_model.encode(sentences=docs, max_length=1024, show_progress_bar = True, batch_size = 32) # For Jina # # 

            #import torch
            #from torch.nn.utils.rnn import pad_sequence

            # Assuming embeddings_out is a list of tensors
            #embeddings_out = [torch.tensor(embedding) for embedding in embeddings_out]

            # Pad the sequences
            # Set batch_first=True if you want the batch dimension to be the first dimension
            #embeddings_out = pad_sequence(embeddings_out, batch_first=True, padding_value=0)


        toc = time.perf_counter()
        time_out = f"The embedding took {toc - tic:0.1f} seconds"
        print(time_out)

        # If you want to save your files for next time
        if return_intermediate_files == "Yes":
            if embeddings_super_compress == "No":
                semantic_search_file_name = data_file_name_no_ext + '_' + 'embeddings.npz'
                np.savez_compressed(semantic_search_file_name, embeddings_out)
            else:
                semantic_search_file_name = data_file_name_no_ext + '_' + 'embedding_compress.npz'
                embeddings_out_round = np.round(embeddings_out, 3) 
                embeddings_out_round *= 100 # Rounding not currently used
                np.savez_compressed(semantic_search_file_name, embeddings_out_round)

    # Pre-reduce embeddings for visualisation purposes
    if reduce_embeddings == "Yes":
        reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings_out)
        return embeddings_out, reduced_embeddings

    return embeddings_out, None