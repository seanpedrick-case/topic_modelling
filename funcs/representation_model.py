import os
#from ctransformers import AutoModelForCausalLM
#from transformers import AutoTokenizer, pipeline
from bertopic.representation import LlamaCPP
from llama_cpp import Llama
from pydantic import BaseModel
import torch.cuda

from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from funcs.prompts import capybara_prompt, capybara_start, open_hermes_prompt, open_hermes_start, stablelm_prompt, stablelm_start



chosen_prompt = open_hermes_prompt # stablelm_prompt 
chosen_start_tag =  open_hermes_start # stablelm_start

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
        print(f"File found: {found_file}")
        return found_file
    else:
        error = "File not found."
        print(error, " Downloading model from hub")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=hf_model_name, filename='phi-2-orange.Q5_K_M.gguf')
        found_file = find_file(folder_path, file_to_find)
        return found_file



# Currently set n_gpu_layers to 0 even with cuda due to persistent bugs in implementation with cuda
if torch.cuda.is_available():
    torch_device = "gpu"
    low_resource_mode = "No"
    n_gpu_layers = 100
else: 
    torch_device =  "cpu"
    low_resource_mode = "Yes"
    n_gpu_layers = 0

low_resource_mode = "No" # Override for testing

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
n_ctx:int = 4096
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
keybert = KeyBERTInspired()

def create_representation_model(create_llm_topic_labels, llm_config, hf_model_name, hf_model_file, chosen_start_tag):

    if create_llm_topic_labels == "Yes":
        # Use llama.cpp to load in model

        # Check for HF_HOME environment variable and supply a default value if it's not found (current folder)
        hf_home_value = os.getenv("HF_HOME", '.')

        found_file = find_model_file(hf_model_name, hf_model_file, hf_home_value)

        llm = Llama(model_path=found_file, stop=chosen_start_tag, n_gpu_layers=llm_config.n_gpu_layers, n_ctx=llm_config.n_ctx) #**llm_config.model_dump())# 
        #print(llm.n_gpu_layers)
        llm_model = LlamaCPP(llm, prompt=chosen_prompt)#, **gen_config.model_dump())

        # All representation models
        representation_model = {
        "KeyBERT": keybert,
        "Mistral": llm_model
        }

    elif create_llm_topic_labels == "No":
        representation_model = {"KeyBERT": keybert}

    # Deprecated example using CTransformers. This package is not really used anymore
    #model = AutoModelForCausalLM.from_pretrained('NousResearch/Nous-Capybara-7B-V1.9-GGUF', model_type='mistral', model_file='Capybara-7B-V1.9-Q5_K_M.gguf', hf=True, **vars(llm_config))
    #tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Capybara-7B-V1.9")
    #generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    # Text generation with Llama 2
    #mistral_capybara = TextGeneration(generator, prompt=capybara_prompt)
    #mistral_hermes = TextGeneration(generator, prompt=open_hermes_prompt)
        
    return representation_model


