import os
from bertopic.representation import LlamaCPP
from llama_cpp import Llama
from pydantic import BaseModel
import torch.cuda
from huggingface_hub import hf_hub_download, snapshot_download

from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, BaseRepresentation
from funcs.prompts import capybara_prompt, capybara_start, open_hermes_prompt, open_hermes_start, stablelm_prompt, stablelm_start

random_seed = 42

chosen_prompt = open_hermes_prompt # stablelm_prompt 
chosen_start_tag =  open_hermes_start # stablelm_start


# Currently set n_gpu_layers to 0 even with cuda due to persistent bugs in implementation with cuda
if torch.cuda.is_available():
    torch_device = "gpu"
    low_resource_mode = "No"
    n_gpu_layers = 100
else: 
    torch_device =  "cpu"
    low_resource_mode = "Yes"
    n_gpu_layers = 0

#low_resource_mode = "No" # Override for testing

#print("Running on device:", torch_device)
n_threads = torch.get_num_threads()
print("CPU n_threads:", n_threads)

# Default Model parameters
temperature: float = 0.1
top_k: int = 3
top_p: float = 1
repeat_penalty: float = 1.1
last_n_tokens_size: int = 128
max_tokens: int = 500
seed: int = 42
reset: bool = True
stream: bool = False
n_threads: int = n_threads
n_batch:int = 256
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
# KeyBERT
keybert = KeyBERTInspired(random_state=random_seed)
# MMR
mmr = MaximalMarginalRelevance(diversity=0.5)

base_rep = BaseRepresentation()

# Find model file
def find_model_file(hf_model_name, hf_model_file, search_folder):
    hf_loc = search_folder #os.environ["HF_HOME"]
    hf_sub_loc = search_folder + "/hub/" #os.environ["HF_HOME"] 

    hf_model_name_path = hf_sub_loc + 'models--' + hf_model_name.replace("/","--")

    print(hf_model_name_path)

    def find_file(root_folder, file_name):
        for root, dirs, files in os.walk(root_folder):
            if file_name in files:
                return os.path.join(root, file_name)
        return None

    # Example usage
    folder_path = hf_model_name_path  # Replace with your folder path
    file_to_find = hf_model_file         # Replace with the file name you're looking for

    found_file = find_file(folder_path, file_to_find) # os.environ["HF_HOME"]
    if found_file:
        print(f"Model file found: {found_file}")
        return found_file
    else:
        error = "File not found."
        print(error, " Downloading model from hub")

        # Specify your custom directory
        # Get HF_HOME environment variable or default to "~/.cache/huggingface/hub"
        #hf_home_value = search_folder

        # Check if the directory exists, create it if it doesn't
        #if not os.path.exists(hf_home_value):
        #    os.makedirs(hf_home_value)

        
       
        found_file = hf_hub_download(repo_id=hf_model_name, filename=hf_model_file)#, local_dir=hf_home_value) # cache_dir

        #path = snapshot_download(
        #    repo_id=hf_model_name,
        #    allow_patterns="config.json",
        #    local_files_only=False
        #)

        print("Downloaded model to: ", found_file)

        #found_file = find_file(path, file_to_find)
        return found_file


def create_representation_model(representation_type, llm_config, hf_model_name, hf_model_file, chosen_start_tag, low_resource_mode):

    if representation_type == "LLM":
        print("Generating LLM representation")
        # Use llama.cpp to load in model

        #    del os.environ["HF_HOME"]     

        # Check for HF_HOME environment variable and supply a default value if it's not found (typical location for huggingface models)
        # Get HF_HOME environment variable or default to "~/.cache/huggingface/hub"
        base_folder = "." #"~/.cache/huggingface/hub"
        hf_home_value = os.getenv("HF_HOME", base_folder)

        # Expand the user symbol '~' to the full home directory path
        if "~" in base_folder:
            hf_home_value = os.path.expanduser(hf_home_value)

        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(hf_home_value):
            os.makedirs(hf_home_value)

        print(hf_home_value)

        found_file = find_model_file(hf_model_name, hf_model_file,  hf_home_value)

        llm = Llama(model_path=found_file, stop=chosen_start_tag, n_gpu_layers=llm_config.n_gpu_layers, n_ctx=llm_config.n_ctx, rope_freq_scale=0.5, seed=seed) #**llm_config.model_dump())# 
        #print(llm.n_gpu_layers)
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

    # Deprecated example using CTransformers. This package is not really used anymore
    #model = AutoModelForCausalLM.from_pretrained('NousResearch/Nous-Capybara-7B-V1.9-GGUF', model_type='mistral', model_file='Capybara-7B-V1.9-Q5_K_M.gguf', hf=True, **vars(llm_config))
    #tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Capybara-7B-V1.9")
    #generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    # Text generation with Llama 2
    #mistral_capybara = TextGeneration(generator, prompt=capybara_prompt)
    #mistral_hermes = TextGeneration(generator, prompt=open_hermes_prompt)
        
    return representation_model


