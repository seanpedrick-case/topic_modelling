open_hermes_prompt = """<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant
"""


# Example prompt demonstrating the output we are looking for
capybara_start = "USER:"

capybara_example_prompt = """USER:I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

Topic label: Environmental impacts of eating meat
"""



# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
capybara_main_prompt = """
Now, create a new topic label given the following information.

I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
ASSISTANT:Topic label:"""

capybara_prompt = capybara_example_prompt + capybara_main_prompt

#print("Capybara prompt: ", capybara_prompt)

# System prompt describes information given to all conversations
open_hermes_start="<|im_start|>"
open_hermes_system_prompt = """<|im_start|>system
You are a helpful, respectful and honest assistant for labeling topics.<|im_end|>
"""

# Example prompt demonstrating the output we are looking for
open_hermes_example_prompt = """<|im_start|>user
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

Topic label: Environmental impacts of eating meat
"""
open_hermes_main_prompt = """
Now, create a new topic label given the following information.

I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.<|im_end|>
<|im_start|>assistant
Topic label:
"""
open_hermes_prompt = open_hermes_system_prompt + open_hermes_example_prompt + open_hermes_main_prompt

#print("Open Hermes prompt: ", open_hermes_prompt)

stablelm_start = "<|user|>"
stablelm_example_prompt = """<|user|>
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

Topic label: Environmental impacts of eating meat
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
stablelm_main_prompt = """
Now, create a new topic label given the following information.

I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.<|endoftext|>
<|assistant|>
Topic label:"""

stablelm_prompt = stablelm_example_prompt + stablelm_main_prompt

#print("StableLM prompt: ", stablelm_prompt)


phi3_start = "<|user|>"
phi3_example_prompt = """<|user|>
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

Topic label: Environmental impacts of eating meat
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
phi3_main_prompt = """
Now, create a new topic label given the following information.

I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.<|end|>
<|assistant|>
Topic label:"""

phi3_prompt = phi3_example_prompt + phi3_main_prompt

#print("phi3 prompt: ", phi3_prompt)