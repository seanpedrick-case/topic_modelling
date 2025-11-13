---
title: Topic modelling
emoji: 🚀
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
license: apache-2.0
---

# Topic modeller

Generate topics from open text in tabular data, based on [BERTopic](https://maartengr.github.io/BERTopic/). Upload a data file (csv, xlsx, or parquet), then specify the open text column that you want to use to generate topics. Click 'Extract topics' after you have selected the minimum similar documents per topic and maximum total topics. Duplicate this space, or clone to your computer to avoid queues here!

Uses fast TF-IDF based embeddings by default, which are fast but does not lead to high quality clusering. Change to higher quality [mxbai-embed-xsmall-v1](mixedbread-ai/mxbai-embed-xsmall-v1) model embeddings (384 dimensions) for better results but slower processing time. If you have an embeddings .npz file previously made using this model, you can load this in at the same time to skip the first modelling step. If you have a pre-defined list of topics for zero-shot modelling, you can upload this as a csv file under 'I have my own list of topics...'. Further configuration options are available such as maximum topics allowed, minimum documents per topic etc.. Topic representation with LLMs currently based on [Llama-3.2-3B-Instruct-Q5_K_M.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF), which is quite slow on CPU, so use a GPU-enabled computer if possible, building from the requirements_gpu.txt file in the base folder.

For small datasets, consider breaking up your text into sentences under 'Clean data' -> 'Split open text...' before topic modelling.

I suggest [Wikipedia mini dataset](https://huggingface.co/datasets/rag-datasets/mini_wikipedia/tree/main/data) for testing the tool here, choose the passages.parquet file for download.