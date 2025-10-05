#!/bin/bash

# 1. Create a dedicated, guaranteed data directory for spaCy
mkdir -p .spacy_data

# 2. Set environment variable to force spaCy to look in this directory
export SPACY_DATA=./.spacy_data

# 3. Download and install the model package directly into the custom directory
# --no-deps ensures only the model package itself is installed here
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz -t .spacy_data --no-deps

# 4. Create the symbolic link inside the custom directory
# This resolves the E050 "Can't find model" error
python -m spacy link .spacy_data/en_core_web_sm-3.8.0 en_core_web_sm --force