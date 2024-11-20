from huggingface_hub import hf_hub_download

# Define the repository and files to download
repo_id = "mixedbread-ai/mxbai-embed-xsmall-v1" #"sentence-transformers/all-MiniLM-L6-v2" #"BAAI/bge-small-en-v1.5"
files_to_download = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "vocab.txt"
]

# Download each file and save it to the /model/bge directory
for file_name in files_to_download:
    print("Checking for file", file_name)
    hf_hub_download(repo_id=repo_id, filename=file_name, local_dir="/model/embed") #"/model/bge"