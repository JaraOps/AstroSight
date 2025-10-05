#!/bin/bash

# 1. NLTK PUNKT FIX (Already in code, but good to keep as backup)
python -m nltk.downloader punkt

# 2. SPAcY FIX (Essential for Knowledge Graph)
python -m spacy link en_core_web_sm en_core_web_sm --force