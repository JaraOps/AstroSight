# create web application with streamlit including knowledge graph, add AI for purpose driven summaries

import os
from typing import List
from pathlib import Path
import collections
import re  # Core Python library for sentence splitting
from operator import itemgetter  # For sorting sentences

# Removed all external NLP libraries: nltk, sumy, gensim, scipy

import pandas as pd
import streamlit as st

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
        # Drop rows where Title is missing to prevent key errors in selectbox
        df.dropna(subset=[TITLE_COL], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data
def summarize_with_keyword_rank(text, keywords_str, sentence_count=3):
    """
    Generates a summary using a simple, dependency-free keyword-ranking algorithm.
    It ranks sentences based on the number of keywords they contain.
    """
    if not text:
        return "No text available for summarization."

    # 1. Prepare keywords
    if isinstance(keywords_str, str):
        split_char = ';' if ';' in keywords_str else ','
        keywords = set([k.strip().lower() for k in keywords_str.split(split_char) if k.strip()])
    else:
        keywords = set()

    if not keywords:
        return "No keywords available to rank the text."

    # 2. Split text into sentences using regex (dependency-free sentence tokenizer)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text.strip())

    # Ensure we don't try to summarize more sentences than exist
    if len(sentences) < sentence_count:
        sentence_count = len(sentences)

    # 3. Score sentences
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0
        for keyword in keywords:
            if keyword in sentence.lower():
                score += 1
        scored_sentences.append({'sentence': sentence, 'score': score, 'order': i})

    # 4. Select top sentences by score
    # Sort by score (descending), then by original order (ascending)
    top_sentences = sorted(scored_sentences, key=itemgetter('score', 'order'), reverse=True)

    # 5. Take the top N, then sort them back into original order
    final_summary_sentences = sorted(top_sentences[:sentence_count], key=itemgetter('order'))

    return " ".join([item['sentence'] for item in final_summary_sentences])


# --- TAG OVERLAP ENGINE (CRITICAL BUG FIX IMPLEMENTED) ---
# NOTE: This function does not change.
def get_tag_similarity(df, reference_title):
    # 1. Get the keywords for the selected article
    ref_keywords_raw = df[df[TITLE_COL] == reference_title][KEYWORDS_COL].iloc[0]

    # FIX: Check if string, then split and clean (The necessary fix for string keywords)
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

            # Score is the count of shared tags
            score = len(ref_keywords.intersection(other_keywords))
            similarity_scores[title] = score

    TOP_N_RESULTS = 5
    top_similar = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)[:TOP_N_RESULTS]
    return top_similar


# NEW HELPER: For Global Keywords Table
# NOTE: This function does not change.
@st.cache_data
def get_top_keywords(df, keywords_column):
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
    # NOTE: This function remains the same, assuming spaCy installation is now stable.
    if not SPACY_AVAILABLE:
        return None

    try:
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
    # show a selectable list
    titles = filtered[TITLE_COL].fillna('Untitled').tolist()
    sel = st.selectbox("Choose an article (by title)", options=titles)

    try:
        idx = filtered[filtered[TITLE_COL].fillna('Untitled') == sel].index[0]
        # small metadata
        st.markdown(f"**Authors:** {filtered.loc[idx, AUTHORS_COL] if AUTHORS_COL in filtered.columns else 'N/A'}")
        if YEAR_COL in filtered.columns:
            st.markdown(f"**Year:** {filtered.loc[idx, YEAR_COL]}")
        if PMC_COL in filtered.columns:
            st.markdown(f"**PMCID:** {filtered.loc[idx, PMC_COL]}")
        st.markdown(f"**Theme:** {filtered.loc[idx, THEME_COL] if THEME_COL in filtered.columns else 'N/A'}")

        # Get keywords for the summary function
        current_keywords = filtered.loc[idx, KEYWORDS_COL]
        current_text = filtered.loc[idx, TEXT_COL]

    except IndexError:
        st.warning("No article selected or filter is too strict.")
        idx = -1  # Set a safe index value
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

    if idx != -1:  # Only proceed if an article is selected
        st.markdown("**Keywords (TF-IDF):**")
        st.write(current_keywords)
        st.markdown("**Abstract / Selected text (cleaned):**")

        st.text_area("Article text", value=current_text if pd.notna(current_text) else "", height=300,
                     label_visibility="collapsed")

        st.markdown("---")

        # --- Summarization Section ---
        st.header("Article Summarization")

        SENTENCE_COUNT = 3

        if st.button("Generate Summary"):
            # Use the simple, dependency-free function
            summary = summarize_with_keyword_rank(current_text, current_keywords, sentence_count=SENTENCE_COUNT)
            st.subheader(f"Article Summary (Keyword Rank Model)")
            st.info(summary)

        st.markdown("---")

        # --- KNOWLEDGE GRAPH SECTION ---
        st.subheader("Knowledge Graph (entities)")
        if not SPACY_AVAILABLE:
            st.info("Graph is disabled (spaCy/pyvis dependencies missing or model not linked).")
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
                    st.error("Graph build failed (Check console for detailed spaCy error)")
                else:
                    net = draw_pyvis_graph(G)
                    tmpfile = "graph_vis.html"
                    net.save_graph(tmpfile)
                    st.components.v1.html(open(tmpfile, 'r', encoding='utf-8').read(), height=600)

        st.markdown("---")

        # --- GLOBAL KEYWORD TABLE (New Stable Feature) ---
        st.header("🌐 Global Keyword Landscape (Top 10)")
        st.write("A quantitative view of the most frequent concepts extracted across all papers.")
        top_keywords_table = get_top_keywords(df, KEYWORDS_COL)
        if not top_keywords_table.empty:
            st.dataframe(top_keywords_table, width='stretch')

        st.markdown("---")

        # --- TAG OVERLAP ENGINE (The Stable Discovery Feature) ---
        st.header("🔑 Inter-document Tag Overlap Engine")
        st.write("Instantly links papers based on shared core topics extracted by our NLP pipeline.")

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