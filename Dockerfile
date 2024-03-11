# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.


# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh


# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

ARG PINECONE_API_KEY
ENV PINECONE_API_KEY=${PINECONE_API_KEY}

ARG COHERE_API_KEY
ENV COHERE_API_KEY=${COHERE_API_KEY}

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /handler.py
