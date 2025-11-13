import os
import re
import spaces
from bertopic.representation import LlamaCPP

from pydantic import BaseModel

from huggingface_hub import hf_hub_download
from gradio import Warning

from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, BaseRepresentation
from funcs.prompts import phi3_prompt, phi3_start
from funcs.helper_functions import get_or_create_env_var, GPU_SPACE_DURATION


def clean_llm_output_text(text: str) -> str:
    """
    Clean LLM output text by removing special characters.
    Keeps only: letters, numbers, spaces, dashes, and apostrophes (for contractions).
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text with special characters removed
    """
    if not text:
        return ""
    
    # Keep only alphanumeric characters, spaces, dashes, and apostrophes
    # This regex keeps: a-z, A-Z, 0-9, spaces, hyphens/dashes, and apostrophes
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-\']', '', text)
    
    # Clean up multiple spaces and strip
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def patch_llama_create_chat_completion(llama_model):
    """
    Monkey-patch the create_chat_completion method on a Llama model instance
    to use raw completion instead of chat format handler.
    This avoids the "System role not supported" error for models like phi3.
    
    Args:
        llama_model: The Llama model instance to patch
        
    Returns:
        The same llama_model instance with patched create_chat_completion method
    """
    def patched_create_chat_completion(messages, **kwargs):
        """
        Override create_chat_completion to use raw completion.
        This avoids the chat format handler that requires system roles (not supported by phi3).
        BERTopic's LlamaCPP formats messages and uses the prompt template, so we reconstruct
        the full prompt from the messages.
        """
        # Reconstruct the prompt from messages
        # BERTopic's LlamaCPP passes messages in OpenAI format: [{"role": "user", "content": "..."}]
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                # Skip system messages as phi3 doesn't support them
                if role != 'system' and content:
                    prompt_parts.append(content)
            else:
                prompt_parts.append(str(msg))
        
        # Join all message contents into a single prompt
        prompt = '\n'.join(prompt_parts) if prompt_parts else ''
        
        # Use raw completion instead of chat completion
        # This avoids the chat format handler that requires system roles
        # Remove chat-specific kwargs that might cause issues, but enable streaming
        completion_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['messages', 'chat_format', 'chat_handler']}
        
        # Enable streaming to show output in real-time
        completion_kwargs['stream'] = True
        
        # Use create_completion for raw text completion (not chat completion)
        # With stream=True, this returns a generator of CompletionChunk objects
        text_parts = []
        try:
            # Create completion with streaming enabled
            completion_stream = llama_model.create_completion(prompt, **completion_kwargs)
            
            # Iterate through the stream and collect text
            print("\nLLM Output: ", end="", flush=True)  # Print prefix without newline
            for chunk in completion_stream:
                # Extract text from each chunk
                chunk_text = ""
                
                # Handle dictionary chunks (the format returned by llama_cpp)
                if isinstance(chunk, dict):
                    # Extract from chunk['choices'][0]['text'] - this is the standard format
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]
                        if isinstance(choice, dict):
                            chunk_text = choice.get('text', '') or choice.get('content', '')
                        elif hasattr(choice, 'text'):
                            chunk_text = choice.text
                        elif hasattr(choice, 'content'):
                            chunk_text = choice.content
                    elif 'text' in chunk:
                        chunk_text = chunk['text']
                    elif 'content' in chunk:
                        chunk_text = chunk['content']
                
                # Try different ways to extract text from the chunk (object format)
                elif hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'text'):
                        chunk_text = choice.text
                    elif hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        # Some formats use delta.content
                        chunk_text = choice.delta.content or ""
                    elif hasattr(choice, 'content'):
                        chunk_text = choice.content
                    elif isinstance(choice, dict):
                        chunk_text = choice.get('text', '') or choice.get('delta', {}).get('content', '')
                elif hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                elif isinstance(chunk, str):
                    chunk_text = chunk
                elif hasattr(chunk, '__dict__'):
                    # Check various possible attributes
                    chunk_dict = chunk.__dict__
                    if 'text' in chunk_dict:
                        chunk_text = chunk_dict['text']
                    elif 'choices' in chunk_dict:
                        choices = chunk_dict['choices']
                        if choices and len(choices) > 0:
                            if isinstance(choices[0], dict):
                                chunk_text = choices[0].get('text', '') or choices[0].get('delta', {}).get('content', '')
                            elif hasattr(choices[0], 'text'):
                                chunk_text = choices[0].text
                            elif hasattr(choices[0], 'delta'):
                                delta = choices[0].delta
                                if hasattr(delta, 'content'):
                                    chunk_text = delta.content or ""
                
                # Only add non-empty text and filter out debug messages
                if chunk_text and chunk_text.strip():
                    # Filter out llama.cpp debug messages
                    if not any(debug_keyword in chunk_text for debug_keyword in [
                        'llama_perf_context_print', 'Llama.generate', 'load time', 
                        'prompt eval time', 'eval time', 'total time', 'prefix-match hit'
                    ]):
                        text_parts.append(chunk_text)
                        print(chunk_text, end="", flush=True)  # Print without newline, flush immediately
            
            print()  # Newline after streaming is complete
            text = ''.join(text_parts)
            
            # Clean the text to remove special characters
            text = clean_llm_output_text(text)
            
            # If no text was collected, there might be an issue with chunk extraction
            if not text:
                print("Warning: No text extracted from streaming chunks. Chunk structure may be different.")
                print("Falling back to non-streaming mode.")
                raise Exception("No text in stream")
            
        except (AttributeError, TypeError, Exception) as e:
            # Fallback to non-streaming if create_completion doesn't exist or streaming fails
            print(f"\nStreaming failed, falling back to non-streaming mode: {e}")
            completion_kwargs.pop('stream', None)  # Remove stream parameter
            try:
                completion = llama_model.create_completion(prompt, **completion_kwargs)
            except AttributeError:
                completion = llama_model(prompt, **completion_kwargs)
            
            # Extract text from the completion object
            text = ""
            if hasattr(completion, 'choices') and len(completion.choices) > 0:
                # Standard Completion object format
                if hasattr(completion.choices[0], 'text'):
                    text = completion.choices[0].text
                elif hasattr(completion.choices[0], 'content'):
                    text = completion.choices[0].content
            elif hasattr(completion, 'text'):
                # Direct text attribute
                text = completion.text
            elif isinstance(completion, str):
                # Already a string
                text = completion
            elif hasattr(completion, '__dict__'):
                # Try to get text from object attributes
                if 'text' in completion.__dict__:
                    text = completion.__dict__['text']
                elif 'choices' in completion.__dict__:
                    choices = completion.__dict__['choices']
                    if choices and len(choices) > 0:
                        if isinstance(choices[0], dict):
                            text = choices[0].get('text', '')
                        elif hasattr(choices[0], 'text'):
                            text = choices[0].text
            else:
                # Last resort: convert to string (but this might not work well)
                text = str(completion)
        
        # Clean up the text - remove special characters and whitespace
        text = clean_llm_output_text(text) if text else ""
        
        # Create a chat completion response as a dictionary
        # BERTopic accesses it as: response["choices"][0]["message"]["content"]
        # Always return a dictionary to ensure it's subscriptable
        return {
            "choices": [{
                "message": {
                    "content": text,
                    "role": "assistant"
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "id": "custom",
            "created": 0,
            "model": "",
            "object": "chat.completion"
        }
    
    # Replace the method on the instance
    llama_model.create_chat_completion = patched_create_chat_completion
    
    return llama_model

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
    #print("Cuda version installed is: ", version.cuda)
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

@spaces.GPU(duration=GPU_SPACE_DURATION)
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
        # Initialize Llama model - try to disable chat format handler if supported
        # This helps avoid "System role not supported" error for models like phi3
        try:
            llm = Llama(model_path=found_file, stop=chosen_start_tag, n_gpu_layers=llm_config.n_gpu_layers, n_ctx=llm_config.n_ctx, seed=seed, chat_format=None)
        except TypeError:
            # If chat_format parameter doesn't exist, try without it or with chat_handler
            try:
                llm = Llama(model_path=found_file, stop=chosen_start_tag, n_gpu_layers=llm_config.n_gpu_layers, n_ctx=llm_config.n_ctx, seed=seed, chat_handler=None)
            except TypeError:
                # Fall back to basic initialization if chat format parameters don't exist
                llm = Llama(model_path=found_file, stop=chosen_start_tag, n_gpu_layers=llm_config.n_gpu_layers, n_ctx=llm_config.n_ctx, seed=seed)
        
        # Monkey-patch the create_chat_completion method to use raw completion
        # This avoids the chat format handler that requires system roles (not supported by phi3)
        # We patch the instance directly so it still passes isinstance checks in BERTopic
        llm = patch_llama_create_chat_completion(llm)
        
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


