AstroSight: AI Research Navigator 🇨🇱
Team: The Chilean Orbital

Project Overview
AstroSight is an advanced, high-performance web application designed to combat information overload in scientific research. I built it using Python and a suite of NLP tools to transform vast collections of 
unstructured astronomical literature (sourced from NASA repositories) into a cohesive, searchable, and interactive knowledge base.

AstroSight empowers researchers to go from a document corpus to quantifiable insights in seconds.

Key Features (The AI Cockpit)
The application provides three distinct layers of AI-driven analysis, all optimized for speed and local execution:

1. Tag Overlap Engine (Performance Solution)
What it does: Instantly identifies and ranks the top 5 most similar articles to a selected document.

Innovation: I replaced slow, vector-based semantic search with a custom function using set theory (set intersection) on pre-extracted keywords. This guarantees millisecond-level performance and precise, quantifiable similarity ranking.

2. Knowledge Graph (KG) Viewer
What it does: Visually maps the entire research landscape.

Technology: Uses spaCy's Named Entity Recognition (NER) to extract connections between documents and entities (Organizations, Events, etc.). The graph is constructed with networkx and rendered interactively with pyvis, revealing complex relationships and networks within the data.

3. LexRank Summarization
What it does: Generates concise, high-quality extractive summaries for any document on demand.

Benefit: Provides instant document triage, allowing users to verify relevance quickly without relying on expensive, external cloud APIs.

Technical Stack
AstroSight is built entirely on open-source Python libraries, prioritizing local execution and cost efficiency.

Category	          Tools & Libraries	Purpose
Application Core:	  Streamlit, pandas, pathlib	Web framework, data management, and secure file path handling.
NLP Engines:      	spaCy, sumy (LexRank)	Named Entity Recognition (NER) and high-performance summarization.
Feature Extraction:	TfidfVectorizer (scikit-learn), nltk	Keyword generation, stopword removal, and text normalization.
Visualization:	    networkx, pyvis	Graph construction and interactive visualization.


Installation and Local Setup
To run AstroSight locally, follow these steps.

Prerequisites
You need Python 3.9+ installed.

1. Clone the Repository
Bash

git clone https://github.com/JaraOps/AstroSight
cd AstroSight

2. Install Dependencies
Install all required Python packages from your environment. You will need to create a requirements.txt file listing all the libraries used (e.g., streamlit, pandas, spacy, pyvis, networkx, sumy, scikit-learn, etc.).

Bash

pip install -r requirements.txt

3. Download the spaCy Model
The Knowledge Graph requires the small English language model. You must link this model for spaCy to function correctly:

Bash

python -m spacy download en_core_web_sm
python -m spacy link en_core_web_sm en_core_web_sm --force

4. Data File
Ensure the project data file, final_analyzed_data.csv, is located in the same directory as streamlit_AstroSight.py.

5. Run the Application
Start the Streamlit application from your terminal:

Bash

streamlit run streamlit_AstroSight.py
The application will open in your default web browser.

Solo Developer & Data Source
This project was developed entirely by [Your Name] as a solo effort.

Data Source
The data used for this project consists of astronomical and astrophysical papers sourced from open-access repositories, including those maintained by NASA (e.g., PMC, ADS).

Future Work
Integrate with a live API (e.g., NASA ADS) for real-time data ingestion.

Develop a feature to track the temporal evolution of entities and keywords over time.

Implement a full-text search index (like Lucene/Elasticsearch) for even faster document querying.
