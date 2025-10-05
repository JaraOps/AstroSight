#!/bin/bash

# 1. Download and install the model package directly
# This ensures the model files are present
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz

# 2. Link the installed package to the name 'en_core_web_sm'
# This resolves the [E050] "Can't find model" error
python -m spacy link en_core_web_sm en_core_web_sm --force