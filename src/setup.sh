#!/bin/bash

# 1. NLTK PUNKT FIX (Most reliable method)
# Downloads the necessary NLTK data before the app starts.
python -m nltk.downloader punkt

# 2. SPAcY FIX
# Links the model installed via requirements.txt, making it callable by spacy.load().
python -m spacy link en_core_web_sm en_core_web_sm --force