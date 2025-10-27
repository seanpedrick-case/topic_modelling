import os
from bertopic.representation import LlamaCPP

from pydantic import BaseModel

from huggingface_hub import hf_hub_download
from gradio import Warning

from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, BaseRepresentation
from funcs.embeddings import torch_device
from funcs.prompts import phi3_prompt, phi3_start
from funcs.helper_functions import get_or_create_env_var

chosen_prompt = phi3_prompt #open_hermes_prompt # stablelm_prompt 
chosen_start_tag =  phi3_start #open_hermes_start # stablelm_start

random_seed = 42

RUNNING_ON_AWS = get_or_create_env_var('RUNNING_ON_AWS', '0')
print(f'The value of RUNNING_ON_AWS is {RUNNING_ON_AWS}')

USE_GPU = get_or_create_env_var('USE_GPU', '0')
print(f'The value of USE_GPU is {USE_GPU}')

# from torch import cuda, backends, version, get_num_threads

# print("Is CUDA enabled? ", cuda.is_available())
# print("Is a CUDA device available on this computer?", backends.cudnn.enabled)
# if cuda.is_available():
#     torch_device = "gpu"
#     print("Cuda version installed is: ", version.cuda)
#     high_quality_mode = "Yes"
#     os.system("nvidia-smi")
# else: 
#     torch_device =  "cpu"
#     high_quality_mode = "No"

if USE_GPU == "1":
    print("Using GPU for representation functions")
    torch_device = "gpu"
    print("Cuda version installed is: ", version.cuda)
    high_quality_mode = "Yes"
    os.system("nvidia-smi")
else:
    print("Using CPU for representation functions")
    torch_device =  "cpu"
    high_quality_mode = "No"

# Currently set n_gpu_layers to 0 even with cuda due to persistent bugs in implementation with cuda
print("torch device for representation functions:", torch_device)
if torch_device == "gpu":
    low_resource_mode = "No"
    n_gpu_layers = -1 # i.e. all
else: #     torch_device =  "cpu"
    low_resource_mode = "Yes"
    n_gpu_layers = 0

#print("Running on device:", torch_device)
from torch import get_num_threads
n_threads = get_num_threads()
print("CPU n_threads:", n_threads)

# Default Model parameters
temperature: float = 0.1
top_k: int = 3
top_p: float = 1
repeat_penalty: float = 1.1
last_n_tokens_size: int = 128
max_tokens: int = 500
seed: int = random_seed
reset: bool = True
stream: bool = False
n_threads: int = n_threads
n_batch:int = 512
n_ctx:int = 8192 #4096. # Set to 8192 just to avoid any exceeded context window issues
sample:bool = True
trust_remote_code:bool =True

class LLamacppInitConfigGpu(BaseModel):
    last_n_tokens_size: int
    seed: int
    n_threads: int
    n_batch: int
    n_ctx: int
    n_gpu_layers: int
    temperature: float
    top_k: int
    top_p: float
    repeat_penalty: float
    max_tokens: int
    reset: bool
    stream: bool
    stop: str
    trust_remote_code:bool

    def update_gpu(self, new_value: int):
        self.n_gpu_layers = new_value

llm_config = LLamacppInitConfigGpu(last_n_tokens_size=last_n_tokens_size,
    seed=seed,
    n_threads=n_threads,
    n_batch=n_batch,
    n_ctx=n_ctx,
    n_gpu_layers=n_gpu_layers,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    repeat_penalty=repeat_penalty,
    max_tokens=max_tokens,
    reset=reset,
    stream=stream,
    stop=chosen_start_tag,
    trust_remote_code=trust_remote_code)

## Create representation model parameters ##
keybert = KeyBERTInspired(random_state=random_seed)
mmr = MaximalMarginalRelevance(diversity=0.5)
base_rep = BaseRepresentation()

# Find model file
def find_model_file(hf_model_name: str, hf_model_file: str, search_folder: str, sub_folder: str) -> str:
    """
    Finds the specified model file within the given search folder and subfolder.

    Args:
        hf_model_name (str): The name of the Hugging Face model.
        hf_model_file (str): The specific file name of the model to find.
        search_folder (str): The base folder to start the search.
        sub_folder (str): The subfolder within the search folder to look into.

    Returns:
        str: The path to the found model file, or None if the file is not found.
    """

    hf_loc = search_folder #os.environ["HF_HOME"]
    hf_sub_loc = search_folder + sub_folder #os.environ["HF_HOME"] 

    if sub_folder == "/hub/":
        hf_model_name_path = hf_sub_loc + 'models--' + hf_model_name.replace("/","--")
    else:
        hf_model_name_path = hf_sub_loc

    def find_file(root_folder, file_name):
        for root, dirs, files in os.walk(root_folder):
            if file_name in files:
                return os.path.join(root, file_name)
        return None

    # Example usage
    folder_path = hf_model_name_path  # Replace with your folder path
    file_to_find = hf_model_file         # Replace with the file name you're looking for

    print("Searching for model file", hf_model_file, "in:", hf_model_name_path)

    found_file = find_file(folder_path, file_to_find) # os.environ["HF_HOME"]
    
    return found_file

def create_representation_model(representation_type: str, llm_config: dict, hf_model_name: str, hf_model_file: str, chosen_start_tag: str, low_resource_mode: bool) -> dict:
    """
    Creates a representation model based on the specified type and configuration.

    Args:
        representation_type (str): The type of representation model to create (e.g., "LLM", "KeyBERT").
        llm_config (dict): Configuration settings for the LLM model.
        hf_model_name (str): The name of the Hugging Face model.
        hf_model_file (str): The specific file name of the model to find.
        chosen_start_tag (str): The start tag to use for the model.
        low_resource_mode (bool): Whether to enable low resource mode.

    Returns:
        dict: A dictionary containing the created representation model.
    """

    if representation_type == "LLM":
        print("RUNNING_ON_AWS:", RUNNING_ON_AWS)
        if RUNNING_ON_AWS=="1":
            error_message = "LLM representation not available on AWS due to model size restrictions. Returning base representation"
            Warning(error_message, duration=5)
            print(error_message)
            representation_model = {"LLM":base_rep}
            return representation_model
        # Else import Llama
        else:
            from llama_cpp import Llama

        print("Generating LLM representation")
        # Use llama.cpp to load in model

        # Check for HF_HOME environment variable and supply a default value if it's not found (typical location for huggingface models)
        base_folder = "model" #"~/.cache/huggingface/hub"
        hf_home_value = os.getenv("HF_HOME", base_folder)

        # Expand the user symbol '~' to the full home directory path
        if "~" in base_folder:
            hf_home_value = os.path.expanduser(hf_home_value)

        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(hf_home_value):
            os.makedirs(hf_home_value)

        print("Searching base folder for model:", hf_home_value)

        found_file = find_model_file(hf_model_name, hf_model_file,  hf_home_value, "/rep/")

        if found_file:
            print(f"Model file found in model folder: {found_file}")

        else:
            found_file = find_model_file(hf_model_name, hf_model_file,  hf_home_value, "/hub/")

        if not found_file:
            error = "File not found in HF hub directory or in local model file."
            print(error, " Downloading model from hub")

            found_file = hf_hub_download(repo_id=hf_model_name, filename=hf_model_file)#, local_dir=hf_home_value) # cache_dir

            print("Downloaded model from Huggingface Hub to: ", found_file)

        print("Loading representation model with", llm_config.n_gpu_layers, "layers allocated to GPU.")

        #llm_config.n_gpu_layers
        llm = Llama(model_path=found_file, stop=chosen_start_tag, n_gpu_layers=llm_config.n_gpu_layers, n_ctx=llm_config.n_ctx,seed=seed) #**llm_config.model_dump())#  rope_freq_scale=0.5,
        #print(llm.n_gpu_layers)
        #print("Chosen prompt:", chosen_prompt)
        llm_model = LlamaCPP(llm, prompt=chosen_prompt)#, **gen_config.model_dump())

        # All representation models
        representation_model = {
        "LLM": llm_model
        }

    elif representation_type == "KeyBERT":
        print("Generating KeyBERT representation")
        #representation_model = {"mmr": mmr}
        representation_model = {"KeyBERT": keybert}

    elif representation_type == "MMR":
        print("Generating MMR representation")
        representation_model = {"MMR": mmr}

    else:
        print("Generating default representation type")
        representation_model = {"Default":base_rep}
        
    return representation_model


