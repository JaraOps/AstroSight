#!/bin/bash

# This is for spaCy linkage, which is a known good practice.
python -m spacy link en_core_web_sm en_core_web_sm --force

# We are no longer adding NLTK here, as the Python code now handles Gensim instead.