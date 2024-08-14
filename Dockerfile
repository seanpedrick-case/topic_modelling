# First stage: build dependencies
#FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm
FROM python:3.11.9-slim-bookworm

# Install Lambda web adapter in case you want to run with with an AWS Lamba function URL (not essential if not using Lambda)
#COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.3 /lambda-adapter /opt/extensions/lambda-adapter

# Install wget, curl, and build-essential
RUN apt-get update && apt-get install -y \
	&& rm -rf /var/lib/apt/lists/*

#wget \
#curl \

# Create a directory for the model
RUN mkdir /model && mkdir /model/rep && mkdir /model/embed

WORKDIR /src

COPY requirements_aws.txt .

RUN pip install --no-cache-dir -r requirements_aws.txt

# Gradio needs to be installed after due to conflict with spacy in requirements
RUN pip install --no-cache-dir gradio==4.41.0

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Change ownership of /home/user directory
RUN chown -R user:user /home/user

# Make output folder
RUN mkdir -p /home/user/app/output && chown -R user:user /home/user/app/output \
&& mkdir -p /home/user/.cache/huggingface/hub && chown -R user:user /home/user/.cache/huggingface/hub \
&& mkdir -p /home/user/.cache/matplotlib && chown -R user:user /home/user/.cache/matplotlib \
&& mkdir -p /home/user/app/model/rep && chown -R user:user /home/user/app/model/rep \
&& mkdir -p /home/user/app/model/embed && chown -R user:user /home/user/app/model/embed \
&& mkdir -p /home/user/app/cache && chown -R user:user /home/user/app/cache

# Download the quantised phi model directly with curl. Changed at it is so big - not loaded
#RUN curl -L -o /home/user/app/model/rep/Phi-3.1-mini-128k-instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Phi-3.1-mini-128k-instruct-GGUF/tree/main/Phi-3.1-mini-128k-instruct-Q4_K_M.gguf

# Download the Mixed bread embedding model during the build process - changed as it is too big for AWS. Not loaded.
#RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
#RUN apt-get install git-lfs -y
#RUN git lfs install
#RUN git clone https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1 /home/user/app/model/embed
#RUN rm -rf /home/user/app/model/embed/.git

# Download the embedding model - Create a directory for the model and download specific files using huggingface_hub
COPY download_model.py /src/download_model.py
RUN python /src/download_model.py

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_SERVER_PORT=7860 \
	GRADIO_THEME=huggingface \
	AWS_STS_REGIONAL_ENDPOINT=regional \
	GRADIO_OUTPUT_FOLDER='output/' \
	NUMBA_CACHE_DIR=/home/user/app/cache \
	SYSTEM=spaces
 
# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

CMD ["python", "app.py"]