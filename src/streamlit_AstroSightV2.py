# create web application with streamlit including knowledge graph, add AI for purpose driven summaries

import os
from typing import List
from pathlib import Path
import collections

# CRITICAL: Importing necessary Hugging Face libraries
try:
    from transformers import pipeline

    # We will load the model only once, using st.cache_resource
    HF_PIPELINE_AVAILABLE = True
except ImportError:
    HF_PIPELINE_AVAILABLE = False

import pandas as pd
import streamlit as st
from operator import itemgetter

# NLP/Graph
try:
    import spacy
    from pyvis.network import Network
    import networkx as nx

    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

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
    full_path = get_absolute_path(DEFAULT_DATAFILE)
    # ... (rest of load_data remains the same)
    if not full_path.exists():
        if not Path(DEFAULT_DATAFILE).exists():
            st.error(f"Data file not found: {full_path}. Check file location or path.")
            return pd.DataFrame()
        df = pd.read_csv(DEFAULT_DATAFILE)
    else:
        df = pd.read_csv(full_path)

    try:
        df = df.rename(columns={
            "Theme_Category": "Theme_category",
        }, inplace=False)
        df.dropna(subset=[TITLE_COL], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# --- CRITICAL NEW FUNCTION: TRANSFORMERS SUMMARIZATION ---
@st.cache_resource
def get_summarizer_pipeline():
    """Loads a small, efficient summarization model once."""
    if not HF_PIPELINE_AVAILABLE:
        return None

    # We use 'facebook/bart-large-cnn' as it's a common, high-quality choice
    # You can change this to a smaller model like 'sshleifer/distilbart-cnn-6-6' if installation fails
    try:
        return pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6"  # Smaller, faster model
        )
    except Exception as e:
        st.error(f"Failed to load Hugging Face model: {e}")
        return None


def summarize_with_transformers(text, min_length=40, max_length=150):
    """Generates a summary using a pre-trained Hugging Face model."""
    summarizer = get_summarizer_pipeline()

    if summarizer is None:
        return "High-quality summarization is unavailable (model failed to load)."

    if not text or len(text.split()) < 50:
        return "Text too short for transformer summarization (min 50 words recommended)."

    try:
        # Truncate text if it's excessively long (models have token limits)
        MAX_INPUT_WORDS = 500
        text_list = text.split()
        if len(text_list) > MAX_INPUT_WORDS:
            text = " ".join(text_list[:MAX_INPUT_WORDS])

        result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return result[0]['summary_text']

    except Exception as e:
        return f"Error during transformer summarization: {e}"


# -----------------------------------------------------------


# --- TAG OVERLAP ENGINE (Remains the same) ---
def get_tag_similarity(df, reference_title):
    # ... (rest of get_tag_similarity remains the same)
    ref_keywords_raw = df[df[TITLE_COL] == reference_title][KEYWORDS_COL].iloc[0]

    if isinstance(ref_keywords_raw, str):
        split_char = ';' if ';' in ref_keywords_raw else ','
        ref_keywords = set([k.strip() for k in ref_keywords_raw.split(split_char) if k.strip()])
    else:
        ref_keywords = set()

    similarity_scores = {}

    for index, row in df.iterrows():
        title = row[TITLE_COL]
        if title != reference_title:
            other_keywords_raw = row[KEYWORDS_COL]

            if isinstance(other_keywords_raw, str):
                split_char = ';' if ';' in other_keywords_raw else ','
                other_keywords = set([k.strip() for k in other_keywords_raw.split(split_char) if k.strip()])
            else:
                other_keywords = set()

            score = len(ref_keywords.intersection(other_keywords))
            similarity_scores[title] = score

    TOP_N_RESULTS = 5
    top_similar = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)[:TOP_N_RESULTS]
    return top_similar


# --- GLOBAL KEYWORD TABLE (Remains the same) ---
@st.cache_data
def get_top_keywords(df, keywords_column):
    # ... (rest of get_top_keywords remains the same)
    all_keywords = []

    for keywords in df[keywords_column].fillna(''):
        if isinstance(keywords, str):
            if ';' in keywords:
                all_keywords.extend([k.strip() for k in keywords.split(';') if k.strip()])
            else:
                all_keywords.extend([k.strip() for k in keywords.split(',') if k.strip()])
        elif isinstance(keywords, list):
            all_keywords.extend(keywords)

    keyword_counts = collections.Counter(all_keywords)

    top_keywords_df = pd.DataFrame(keyword_counts.most_common(10),
                                   columns=['Keyword', 'Frequency'])
    return top_keywords_df


@st.cache_data
def build_entity_graph(texts: List[str], titles: List[str], entity_labels: List[str], top_n_entities=5):
    # ... (rest of build_entity_graph remains the same)
    if not SPACY_AVAILABLE:
        return None

    try:
        # Load the spacy model here
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        return None

    G = nx.Graph()

    COLOR_MAP = {
        "PERSON": "#FF5733", "ORG": "#33FF57", "GPE": "#3357FF", "NORP": "#FF33A1",
        "EVENT": "#33FFF3", "PRODUCT": "#F3FF33", "DOC": '#8fb9a8'
    }

    for i, txt in enumerate(texts):
        title = titles[i]
        G.add_node(title, type="doc", color=COLOR_MAP['DOC'])
        doc = nlp(txt[:5000])

        ents_by_label = {}
        for ent in doc.ents:
            if ent.label_ in entity_labels:
                norm_ent = ent.text.strip()
                if norm_ent not in ents_by_label:
                    ents_by_label[norm_ent] = ent.label_

        for ent_text, ent_label in list(ents_by_label.items())[:top_n_entities]:
            if not G.has_node(ent_text):
                net_label = ent_text
                net_title = f"{ent_text} ({ent_label})"
                net_color = COLOR_MAP.get(ent_label, '#CCCCCC')
                G.add_node(ent_text, type="entity", label=net_label, color=net_color, title=net_title)

            G.add_edge(title, ent_text)

    return G


def draw_pyvis_graph(G: nx.Graph, height="600px", width="100%"):
    # ... (rest of draw_pyvis_graph remains the same)
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
st.title("AstroSight 🇨🇱 by The Chilean Orbital")

# Sidebar: Data files and quick settings
st.sidebar.header("Data & Settings")
datafile = st.sidebar.text_input("Data CSV path", value=DEFAULT_DATAFILE)
df = load_data()

if df.empty:
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
    titles = filtered[TITLE_COL].fillna('Untitled').tolist()
    sel = st.selectbox("Choose an article (by title)", options=titles)

    try:
        idx = filtered[filtered[TITLE_COL].fillna('Untitled') == sel].index[0]
        st.markdown(f"**Authors:** {filtered.loc[idx, AUTHORS_COL] if AUTHORS_COL in filtered.columns else 'N/A'}")
        if YEAR_COL in filtered.columns:
            st.markdown(f"**Year:** {filtered.loc[idx, YEAR_COL]}")
        if PMC_COL in filtered.columns:
            st.markdown(f"**PMCID:** {filtered.loc[idx, PMC_COL]}")
        st.markdown(f"**Theme:** {filtered.loc[idx, THEME_COL] if THEME_COL in filtered.columns else 'N/A'}")

        current_keywords = filtered.loc[idx, KEYWORDS_COL]
        current_text = filtered.loc[idx, TEXT_COL]

    except IndexError:
        st.warning("No article selected or filter is too strict.")
        idx = -1
        current_keywords = ""
        current_text = ""

    st.markdown("---")
    st.subheader("Export / Batch")

    to_export = st.multiselect("Select rows to export (titles)", options=filtered[TITLE_COL].tolist(),
                               key='export_titles_selector')

    if to_export:
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

    if idx != -1:
        st.markdown("**Keywords (TF-IDF):**")
        st.write(current_keywords)
        st.markdown("**Abstract / Selected text (cleaned):**")

        st.text_area("Article text", value=current_text if pd.notna(current_text) else "", height=300,
                     label_visibility="collapsed")

        st.markdown("---")

        # --- Summarization Section ---
        st.header("Article Summarization")

        if st.button("Generate AI Summary (BART-CNN)"):
            if HF_PIPELINE_AVAILABLE:
                with st.spinner("Generating summary with transformer model..."):
                    summary = summarize_with_transformers(current_text)
                    st.subheader(f"Article Summary (BART-CNN Model)")
                    st.info(summary)
            else:
                st.error(
                    "High-quality summarization dependencies are missing or failed to load. Please check installation logs for `transformers` and `torch`.")

        st.markdown("---")

        # --- KNOWLEDGE GRAPH SECTION ---
        st.subheader("Knowledge Graph (entities)")
        if not SPACY_AVAILABLE:
            st.info("Graph is disabled (spaCy/pyvis dependencies missing or model not linked).")
        else:
            # ... (rest of graph code remains the same)
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
                    st.error("Graph build failed (Check console for detailed spaCy error)")
                else:
                    net = draw_pyvis_graph(G)
                    tmpfile = "graph_vis.html"
                    net.save_graph(tmpfile)
                    st.components.v1.html(open(tmpfile, 'r', encoding='utf-8').read(), height=600)

        st.markdown("---")

        # --- GLOBAL KEYWORD TABLE ---
        st.header("🌐 Global Keyword Landscape (Top 10)")
        top_keywords_table = get_top_keywords(df, KEYWORDS_COL)
        if not top_keywords_table.empty:
            st.dataframe(top_keywords_table, width='stretch')

        st.markdown("---")

        # --- TAG OVERLAP ENGINE ---
        st.header("🔑 Inter-document Tag Overlap Engine")

        reference_title = st.selectbox(
            "Select an Article to find look-alikes:",
            options=df[TITLE_COL].tolist(),
            key='tag_ref_selector'
        )

        if reference_title:
            top_similar = get_tag_similarity(df, reference_title)

            results_list = []
            for title, score in top_similar:
                theme = df[df[TITLE_COL] == title][THEME_COL].iloc[0]

                results_list.append({
                    "Title": title,
                    "Theme": theme,
                    "Shared Keywords": score
                })

            st.dataframe(results_list, width='stretch')