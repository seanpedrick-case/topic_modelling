# Stage 1: Build dependencies and download models
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm AS builder

# Install Lambda web adapter in case you want to run with with an AWS Lamba function URL (not essential if not using Lambda)
#COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Install system dependencies
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Create directories (if needed for model download script)
RUN mkdir -p /model/rep /model/embed /install

WORKDIR /src

# Copy requirements file and install dependencies. Sentence transformers and Bertopic are installed without dependencies so that torch is not reinstalled.
COPY requirements_aws.txt .

RUN pip install torch>=2.6.0+cpu --target=/install --index-url https://download.pytorch.org/whl/cpu \
&& pip install --no-cache-dir --target=/install sentence-transformers>=5.1.2 --no-deps \
&& pip install --no-cache-dir --target=/install bertopic==0.17.3 --no-deps \
&& pip install --no-cache-dir --target=/install -r requirements_aws.txt \
&& pip install --no-cache-dir --target=/install gradio==5.49.1

# Add /install to the PYTHONPATH
ENV PYTHONPATH="/install:${PYTHONPATH}"

# Download models (using your download_model.py script)
COPY download_model.py /src/download_model.py
RUN python /src/download_model.py

RUN rm requirements_aws.txt download_model.py

# Stage 2: Final runtime image
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm

# Create a non-root user
RUN useradd -m -u 1000 user

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Create necessary directories and set ownership
RUN mkdir -p /home/user/app/output /home/user/.cache/huggingface/hub /home/user/.cache/matplotlib /home/user/app/cache \
    && chown -R user:user /home/user

# Download the quantised phi model directly with curl. Changed at it is so big - not loaded
#RUN curl -L -o /home/user/app/model/rep/Llama-3.2-3B-Instruct-Q5_K_M.gguf https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf

# Copy models from the builder stage
COPY --from=builder /model/rep /home/user/app/model/rep
COPY --from=builder /model/embed /home/user/app/model/embed

# Switch to the non-root user
USER user

# Set environment variables
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

# Set working directory and copy application code
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Command to run your application
CMD ["python", "app.py"]