import time
import numpy as np
from torch import cuda, backends, version

# Check for torch cuda
# If you want to disable cuda for testing purposes
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Is CUDA enabled? ", cuda.is_available())
print("Is a CUDA device available on this computer?", backends.cudnn.enabled)
if cuda.is_available():
    torch_device = "gpu"
    print("Cuda version installed is: ", version.cuda)
    high_quality_mode = "Yes"
    #os.system("nvidia-smi")
else: 
    torch_device =  "cpu"
    high_quality_mode = "No"

print("Device used is: ", torch_device)



def make_or_load_embeddings(docs: list, file_list: list, embeddings_out: np.ndarray, embedding_model, embeddings_super_compress: str, high_quality_mode_opt: str) -> np.ndarray:
    """
    Create or load embeddings for the given documents.

    Args:
        docs (list): List of documents to embed.
        file_list (list): List of file names to check for existing embeddings.
        embeddings_out (np.ndarray): Array to store the embeddings.
        embedding_model: Model used to generate embeddings.
        embeddings_super_compress (str): Option to super compress embeddings ("Yes" or "No").
        high_quality_mode_opt (str): Option for high quality mode ("Yes" or "No").

    Returns:
        np.ndarray: The generated or loaded embeddings.
    """

    # If no embeddings found, make or load in
    if embeddings_out.size == 0:
        print("Embeddings not found. Loading or generating new ones.")

        embeddings_file_names = [string for string in file_list if "embedding" in string.lower()]  
        
        if embeddings_file_names:
            embeddings_file_name = embeddings_file_names[0]
            print("Loading embeddings from file.")
            embeddings_out = np.load(embeddings_file_name)['arr_0']

            # If embedding files have 'super_compress' in the title, they have been multiplied by 100 before save
            if "compress" in embeddings_file_name:
                embeddings_out /= 100

        if not embeddings_file_names:
            tic = time.perf_counter()
            print("Starting to embed documents.")

            # Custom model
            # If on CPU, don't resort to embedding models
            if high_quality_mode_opt == "No":
                print("Creating simplified 'sparse' embeddings based on TfIDF")

                # Fit the pipeline to the text data
                embedding_model.fit(docs)

                # Transform text data to embeddings
                embeddings_out = embedding_model.transform(docs)

            elif high_quality_mode_opt == "Yes":
                print("Creating dense embeddings based on transformers model")

                embeddings_out = embedding_model.encode(sentences=docs, show_progress_bar = True, batch_size = 32)#, precision="int8") # For large

            toc = time.perf_counter()
            time_out = f"The embedding took {toc - tic:0.1f} seconds"
            print(time_out)

           # If the user has chosen to go with super compressed embedding files to save disk space
            if embeddings_super_compress == "Yes":
                embeddings_out = np.round(embeddings_out, 3) 
                embeddings_out *= 100

        return embeddings_out

    else:
        print("Found pre-loaded embeddings.")

        return embeddings_out