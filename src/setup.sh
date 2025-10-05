#!/bin/bash

# 1. NLTK PUNKT FIX: Download NLTK resources directly to the application path
# This is the most reliable fix for the "NLTK tokenizers are missing" error.
python -m nltk.downloader punkt

# 2. SPAcY FIX: Link the model installed via requirements.txt
# This ensures spacy can find the en_core_web_sm model
python -m spacy link en_core_web_sm en_core_web_sm --force