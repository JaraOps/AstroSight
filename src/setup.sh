#!/bin/bash

# Download and install the model package directly
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz

# Link the installed package to the name 'en_core_web_sm'
python -m spacy link en_core_web_sm en_core_web_sm --forcece