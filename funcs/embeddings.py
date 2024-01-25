import time
import numpy as np
from torch import cuda
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

random_seed = 42

if cuda.is_available():
    torch_device = "gpu"
else: 
    torch_device =  "cpu"

def make_or_load_embeddings(docs, file_list, data_file_name_no_ext, embeddings_out, embedding_model, return_intermediate_files, embeddings_super_compress, low_resource_mode_opt, reduce_embeddings="Yes"):

    # If no embeddings found, make or load in
    if embeddings_out.size == 0:
        print("Embeddings not found. Loading or generating new ones.")

        embeddings_file_names = [string.lower() for string in file_list if "embedding" in string.lower()]  

        if embeddings_file_names:
            print("Loading embeddings from file.")
            embeddings_out = np.load(embeddings_file_names[0])['arr_0']

            # If embedding files have 'super_compress' in the title, they have been multiplied by 100 before save
            if "compress" in embeddings_file_names[0]:
                embeddings_out /= 100

        if not embeddings_file_names:
            tic = time.perf_counter()
            print("Starting to embed documents.")

            # Custom model
            # If on CPU, don't resort to embedding models
            if low_resource_mode_opt == "Yes":
                print("Creating simplified 'sparse' embeddings based on TfIDF")

                embedding_model = make_pipeline(
                TfidfVectorizer(),
                TruncatedSVD(100, random_state=random_seed)
                )

                # Fit the pipeline to the text data
                embedding_model.fit(docs)

                # Transform text data to embeddings
                embeddings_out = embedding_model.transform(docs)

                #embeddings_out = embedding_model.encode(sentences=docs, show_progress_bar = True, batch_size = 32)

            elif low_resource_mode_opt == "No":
                print("Creating dense embeddings based on transformers model")

                embeddings_out = embedding_model.encode(sentences=docs, max_length=1024, show_progress_bar = True, batch_size = 32) # For Jina # # 

            toc = time.perf_counter()
            time_out = f"The embedding took {toc - tic:0.1f} seconds"
            print(time_out)

           # If the user has chosen to go with super compressed embedding files to save disk space
            if embeddings_super_compress == "Yes":
                embeddings_out = np.round(embeddings_out, 3) 
                embeddings_out *= 100

    else:
        print("Found pre-loaded embeddings.")

    # Pre-reduce embeddings for visualisation purposes
    if reduce_embeddings == "Yes":
        if low_resource_mode_opt == "No":
            reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=random_seed).fit_transform(embeddings_out)
            return embeddings_out, reduced_embeddings
        else:
            reduced_embeddings = TruncatedSVD(2, random_state=random_seed).fit_transform(embeddings_out)
            return embeddings_out, reduced_embeddings

    return embeddings_out, None