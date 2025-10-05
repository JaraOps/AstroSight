#!/bin/bash

# 1. NLTK PUNKT FIX (Essential for Summarization)
python -m nltk.downloader punkt

# 2. SPAcY FIX (Essential for Knowledge Graph)
python -m spacy link en_core_web_sm en_core_web_sm --force