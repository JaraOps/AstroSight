# create web application with streamlit including knowledge graph, add AI for purpose driven summaries

import os
from typing import List
from pathlib import Path

from pyvis.network import Network
import networkx as nx
import numpy as np
import collections

import pandas as pd
import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import spacy
from pathlib import Path

model_path = Path(__file__).parent / "en_core_web_sm"
nlp = spacy.load(model_path)


SPACY_AVAILABLE = True


# CONFIG
DEFAULT_DATAFILE = "final_analyzed_data.csv"
TEXT_COL = "Clean_Text_for_NLP"
KEYWORDS_COL = "Top_Keywords"
THEME_COL = "Theme_category"
TITLE_COL = "Title"
AUTHORS_COL = "Authors"
YEAR_COL = "Year"
PMC_COL = "PMC_ID"


# HELPER FUNCTIONS
def get_absolute_path(filename):
    BASE_DIR = Path(__file__).parent
    return BASE_DIR / filename


@st.cache_data
def load_data():
    # Robust path check: looks in the current script's directory first
    full_path = get_absolute_path(DEFAULT_DATAFILE)

    if not full_path.exists():
        st.error(f"Data file not found: {full_path}. Ensure '{DEFAULT_DATAFILE}' is in the same folder as the script.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(full_path)

        # Fix Theme Category and ensure column is present
        df = df.rename(columns={
            "Theme_Category": "Theme_category",
        }, inplace=False)

        # Drop rows where Title is missing to prevent key errors in selectbox
        df.dropna(subset=[TITLE_COL], inplace=True)

        return df
    except Exception as e:
        st.error(f"Error loading or renaming data: {e}")
        return pd.DataFrame()


@st.cache_data
def summarize_with_lexrank(text, sentence_count=3):
    """Generates a summary using the LexRank algorithm (local & free)."""

    if not text:
        return "No text available for summarization."

    if len(text.split('.')) < sentence_count:
        sentence_count = len(text.split('.'))

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary])

    except Exception as e:
        return f"Error during local summarization: {e}"


# --- NEW STABLE KEYWORD ANALYSIS FUNCTION ---
@st.cache_data
def get_top_keywords(df, keywords_column):
    all_keywords = []

    for keywords in df[keywords_column].fillna(''):
        if isinstance(keywords, str):
            # Split by semicolon ';' based on your data format
            all_keywords.extend([k.strip() for k in keywords.split(';') if k.strip()])
        elif isinstance(keywords, list):
            all_keywords.extend(keywords)

    keyword_counts = collections.Counter(all_keywords)

    top_keywords_df = pd.DataFrame(keyword_counts.most_common(10),
                                   columns=['Keyword', 'Frequency'])
    return top_keywords_df


# --- TAG OVERLAP ENGINE (The Stable Discovery Feature) ---
def get_tag_similarity(df, reference_title):
    # Step 1: Ensure Keywords are split by semicolon ';'
    ref_keywords_raw = df[df[TITLE_COL] == reference_title][KEYWORDS_COL].iloc[0]
    ref_keywords = set(ref_keywords_raw.split(';') if isinstance(ref_keywords_raw, str) else [])
    ref_keywords = {k.strip() for k in ref_keywords if k.strip()}  # Clean whitespace

    similarity_scores = {}

    for index, row in df.iterrows():
        title = row[TITLE_COL]
        if title != reference_title:
            other_keywords_raw = row[KEYWORDS_COL]
            other_keywords = set(other_keywords_raw.split(';') if isinstance(other_keywords_raw, str) else [])
            other_keywords = {k.strip() for k in other_keywords if k.strip()}  # Clean whitespace

            score = len(ref_keywords.intersection(other_keywords))
            similarity_scores[title] = score

    TOP_N_RESULTS = 5
    top_similar = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)[:TOP_N_RESULTS]
    return top_similar


@st.cache_data
def build_entity_graph(texts: List[str], titles: List[str], entity_labels: List[str], top_n_entities=5):
    # Use Spacy to extract entities and build a graph.
    st.write("Starting entity graph build...")

    if not SPACY_AVAILABLE:
        return None

    G = nx.Graph()

    COLOR_MAP = {
        "PERSON": "#FF5733", "ORG": "#33FF57", "GPE": "#3357FF", "NORP": "#FF33A1",
        "EVENT": "#33FFF3", "PRODUCT": "#F3FF33", "DOC": '#8fb9a8'
    }

    for i, txt in enumerate(texts):
        title = titles[i]

        G.add_node(title, type="doc", color=COLOR_MAP['DOC'])

        doc = nlp(txt[:5000])  # Limit text length for speed

        ents_by_label = {}
        for ent in doc.ents:
            if ent.label_ in entity_labels:  # Filter by selected labels
                norm_ent = ent.text.strip()
                if norm_ent not in ents_by_label:
                    ents_by_label[norm_ent] = ent.label_

        # adds top N entities
        for ent_text, ent_label in list(ents_by_label.items())[:top_n_entities]:
            if not G.has_node(ent_text):
                G.add_node(ent_text, type="entity", label=ent_text, color=COLOR_MAP.get(ent_label, '#CCCCCC'),
                           title=f"{ent_text} ({ent_label})")

            G.add_edge(title, ent_text)

    return G


def draw_pyvis_graph(G: nx.Graph, height="600px", width="100%"):
    net = Network(height=height, width=width, notebook=False, cdn_resources='local', directed=False)

    for n, attrs in G.nodes(data=True):
        label = attrs.get('label', n)
        title = attrs.get('title', n)
        color = attrs.get('color', '#97C2DE')

        if attrs.get("type") == "doc":
            net.add_node(n, label=label[:60], title=title, color=color, size=20)
        else:
            net.add_node(n, label=label[:40], title=title, color=color, size=10)

    for a, b in G.edges():
        net.add_edge(a, b)

    net.repulsion(node_distance=200, spring_length=300)

    return net


# UI
st.title("AstroSight by The Chilean Orbital")

# Sidebar: Data files and quick settings
st.sidebar.header("Data & Settings")
datafile = st.sidebar.text_input("Data CSV path", value=DEFAULT_DATAFILE)
df = load_data()

if df.empty:
    st.stop()

if TEXT_COL not in df.columns:
    st.error(f"Required text column '{TEXT_COL}' not found in data.")
    st.stop()

# Filters
st.sidebar.subheader("Filters")
query = st.sidebar.text_input("Search (title, abstract, keywords)")
years = None
if YEAR_COL in df.columns:
    years = sorted(df[YEAR_COL].dropna().unique())
    selected_years = st.sidebar.multiselect("Year", options=years, default=None)
else:
    selected_years = None

# Filter DataFrame
filtered = df.copy()
if query:
    q = query.lower()
    mask = filtered[TITLE_COL].fillna('').str.lower().str.contains(q) | filtered[KEYWORDS_COL].fillna(
        '').str.lower().str.contains(q) | filtered[TEXT_COL].fillna('').str.lower().str.contains(q)
    filtered = filtered[mask]

if selected_years and YEAR_COL in filtered.columns:
    filtered = filtered[filtered[YEAR_COL].isin(selected_years)]

st.sidebar.markdown(f"**Results:** {len(filtered)} publications")


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


# Main layout: list on left, details on right
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Publications")
    # show a selectable list
    titles = filtered[TITLE_COL].fillna('Untitled').tolist()
    sel = st.selectbox("Choose an article (by title)", options=titles)

    # Get the index of the selected article
    if not filtered.empty and sel in filtered[TITLE_COL].fillna('Untitled').tolist():
        idx = filtered[filtered[TITLE_COL].fillna('Untitled') == sel].index[0]

        # small metadata
        st.markdown(f"**Authors:** {filtered.loc[idx, AUTHORS_COL] if AUTHORS_COL in filtered.columns else 'N/A'}")
        if YEAR_COL in filtered.columns:
            st.markdown(f"**Year:** {filtered.loc[idx, YEAR_COL]}")
        if PMC_COL in filtered.columns:
            st.markdown(f"**PMCID:** {filtered.loc[idx, PMC_COL]}")
        st.markdown(f"**Theme:** {filtered.loc[idx, THEME_COL] if THEME_COL in filtered.columns else 'N/A'}")
    else:
        st.warning("Please select a title.")
        idx = -1  # Set a safe index value

    st.markdown("---")
    st.subheader("Export / Batch")

    to_export = st.multiselect("Select rows to export (titles)", options=filtered[TITLE_COL].tolist(),
                               key='export_titles_selector')

    if to_export:
        # FIX: The df definition must be inside the if block to avoid NameError
        outdf = filtered[filtered[TITLE_COL].isin(to_export)]

        csv_data = convert_df_to_csv(outdf)

        st.download_button(
            label="Download Selected CSV",
            data=csv_data,
            file_name='export_selected_data.csv',
            mime='text/csv',
            key='download_csv'
        )
    else:
        st.info("Select titles above to enable the download button.")

with col2:
    st.subheader("Article details & tools")
    st.markdown(f"### {sel}")

    if idx != -1:  # Only proceed if an article is selected
        text = filtered.loc[idx, TEXT_COL]
        st.markdown("**Keywords (TF-IDF):**")
        st.write(filtered.loc[idx, KEYWORDS_COL])
        st.markdown("**Abstract / Selected text (cleaned):**")

        st.text_area("Article text", value=text if pd.notna(text) else "", height=300, label_visibility="collapsed")

        st.markdown("---")

        # --- Summarization Section ---
        st.header("Article Summarization")
        titles_list = df[TITLE_COL].unique().tolist()
        selected_title = st.selectbox(
            "Select an Article to Summarize:",
            options=titles_list,
            key='summary_title_selector'
        )
        SENTENCE_COUNT = 3

        selected_text = df[df[TITLE_COL] == selected_title][TEXT_COL].iloc[0]

        if st.button("Generate Summary"):
            summary = summarize_with_lexrank(selected_text, sentence_count=SENTENCE_COUNT)
            st.subheader(f"Article Summary ({SENTENCE_COUNT} Sentences, LexRank Model)")
            st.info(summary)

        st.markdown("---")

        # --- KNOWLEDGE GRAPH SECTION ---
        st.subheader("Knowledge Graph (entities)")
        if not SPACY_AVAILABLE:
            st.error("Graph build failed (spaCy not available or model error)")
        else:
            st.write(
                "Visualizes connections between the selected documents and extracted entities (e.g., ORGs, Events).")

            NER_LABELS = ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART"]
            selected_labels = st.multiselect(
                "Entities to include (NER Labels):",
                options=NER_LABELS,
                default=["ORG", "GPE", "EVENT"],
                key='ner_labels_multiselect'
            )

            max_docs = st.slider("Number of documents to index for graph",
                                 1,
                                 min(50, len(filtered)),
                                 value=min(20, len(filtered)),
                                 key='max_docs_slider')

            sample = filtered.head(max_docs)

            if st.button("Build entity graph", key='build_graph_button'):
                G = build_entity_graph(
                    sample[TEXT_COL].fillna('').tolist(),
                    sample[TITLE_COL].fillna('').tolist(),
                    selected_labels
                )

                if G is None:
                    # This error handles the case where nlp = spacy.load fails internally
                    st.error("Graph build failed (Check console for detailed spaCy error)")
                else:
                    net = draw_pyvis_graph(G)
                    tmpfile = "graph_vis.html"
                    net.save_graph(tmpfile)
                    st.components.v1.html(open(tmpfile, 'r', encoding='utf-8').read(), height=600)

    st.markdown("---")

    # --- GLOBAL KEYWORD TABLE (Stable Analysis) ---
    st.header("🌐 Global Keyword Landscape (Top 10)")
    st.write("A quantitative view of the most frequent concepts extracted across all papers.")
    top_keywords_table = get_top_keywords(df, KEYWORDS_COL)
    if not top_keywords_table.empty:
        st.dataframe(top_keywords_table, use_container_width=True)

    st.markdown("---")

    # --- TAG OVERLAP ENGINE (The Stable Discovery Feature) ---
    st.header("🔑 Inter-document Tag Overlap Engine")
    st.write("Instantly links papers based on shared core topics extracted by our NLP pipeline.")

    # Define the selectbox for user input
    reference_title = st.selectbox(
        "Select an Article to find look-alikes:",
        options=df[TITLE_COL].tolist(),
        key='tag_ref_selector'
    )