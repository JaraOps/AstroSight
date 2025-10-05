#create web application with streamlit including knowledge graph, add AI for purpose driven summaries

import os
from typing import List

import pandas as pd
import streamlit as st

#AI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
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
TITLE_COL = "Article_Title"
AUTHORS_COL = "Authors"
YEAR_COL = "Year"
PMC_COL = "PMC_ID"

#HELPER FUNCTIONS
def load_data(path = DEFAULT_DATAFILE):
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv("final_analyzed_data.csv")
    df.rename(columns={
        "Theme_Category": "Theme_category",
        "Title": "Article_Title"
    }, inplace=True)

    return df


def summarize_with_openai(text: str, purpose: str = "short summary (3 sentences)") -> str:
    if not OPENAI_AVAILABLE:
        return "OpenAI library not installed. Install `openai` to enable AI summaries."

    key = os.environ.get("OPENAI_API_KEY")

    if not key:
        return "OpenAI API key not set. Set the OPENAI_API_KEY environment to enable AI summaries."

    try:
        client = openai.OpenAI(api_key=key)

        prompt = f"Provide a {purpose} of the following scientific text. Be concise and factual.\n\nText:\n{text[:4000]}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        # La forma de acceder al contenido también cambia
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI request failed: {e}"

@st.cache_data
def build_entity_graph(texts: List[str], titles: List[str], entity_labels: List[str], top_n_entities=5):
    # Use Spacy to extract entities and build a graph.
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

    # use the existing node attributes for labels and colors
    for n, attrs in G.nodes(data=True):
        label = attrs.get('label', n)
        title = attrs.get('title', n)
        color = attrs.get('color', '#97C2DE')  #use existing color or default

        if attrs.get("type") == "doc":
            # Node for document
            net.add_node(n, label=label[:60], title=title, color=color, size=20)
        else:
            # entity nodes
            net.add_node(n, label=label[:40], title=title, color=color, size=10)

    for a, b in G.edges():
        net.add_edge(a, b)

    net.repulsion(node_distance=200, spring_length=300)

    return net

# UI
st.title("NASA Space Biology - Explorer & Summarizer") #Change to AstroSight

#Sidebar: Data files and quick settings
st.sidebar.header("Data & Settings")
datafile = st.sidebar.text_input("Data CSV path", value=DEFAULT_DATAFILE)
df = load_data(datafile)

if df.empty:
    st.stop()

#Basic columns

if TEXT_COL in df.columns:
    possible = [c for c in df.columns if "text" in c.lower() or "clean" in c.lower()]
    if possible:
        TEXT_COL = possible[0]
    else:
        st.error("Couldn't find a text column to show article content.")
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
    filtered=filtered[filtered[YEAR_COL].isin(selected_years)]

st.sidebar.markdown(f"**Results:** {len(filtered)} publications")

#Main layout: list on left, details on right
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Publications")
    # show a selectable list
    titles = filtered[TITLE_COL].fillna('Untitled').tolist()
    sel = st.selectbox("Choose an article (by title)", options=titles)
    idx = filtered[filtered[TITLE_COL].fillna('Untitled') == sel].index[0]
    # small metadata
    st.markdown(f"**Authors:** {filtered.loc[idx, AUTHORS_COL] if AUTHORS_COL in filtered.columns else 'N/A'}")
    if YEAR_COL in filtered.columns:
        st.markdown(f"**Year:** {filtered.loc[idx, YEAR_COL]}")
    if PMC_COL in filtered.columns:
        st.markdown(f"**PMCID:** {filtered.loc[idx, PMC_COL]}")
    st.markdown(f"**Theme:** {filtered.loc[idx, THEME_COL] if THEME_COL in filtered.columns else 'N/A'}")
    st.markdown("---")
    st.subheader("Export / Batch")
    to_export = st.multiselect("Select rows to export (titles)", options=titles)
    if st.button("Export selected to CSV"):
        if not to_export:
            st.warning("No items selected")
    else:
        outdf = filtered[filtered[TITLE_COL].isin(to_export)]
    outpath = st.text_input("Output filename", value="export_selected.csv")
    outdf.to_csv(outpath, index=False)
    st.success(f"Exported {len(outdf)} rows to {outpath}")

with col2:
    st.subheader("Article details & tools")
    st.markdown(f"### {sel}")
    text = filtered.loc[idx, TEXT_COL]
    st.markdown("**Keywords (TF-IDF):**")
    st.write(filtered.loc[idx, KEYWORDS_COL])
    st.markdown("**Abstract / Selected text (cleaned):**")

    st.text_area("Article text", value=text if pd.notna(text) else "", height=300, label_visibility="collapsed")

    st.markdown("---")
    st.subheader("AI Summary & Q&A")
    if not OPENAI_AVAILABLE:
        st.info("OpenAI python library not installed. Install `openai` to enable in-app summarization.")
    else:
        st.write("Set OPENAI_API_KEY as an environment variable to enable generation.")
    if st.button("Generate short AI summary (3 sentences)"):
        summary = summarize_with_openai(text, purpose="short summary (3 sentences)")
        st.info(summary)

    st.markdown("---")
    st.subheader("Knowledge Graph (entities)")
    if not SPACY_AVAILABLE:
        st.info("spaCy/pyvis not installed. Install `spacy`, `pyvis`, and download `en_core_web_sm` to enable graph.")
    else:
        max_docs = st.slider("Number of documents to index for graph", 1, min(50, len(filtered)),
                             value=min(20, len(filtered)))

    NER_LABELS = ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART"]
    selected_labels = st.multiselect(
        "Entities to include (NER Labels):",
        options=NER_LABELS,
        default=["ORG", "GPE", "EVENT"]
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
            st.error("Graph build failed (spaCy not available or model error)")
        else:
            net = draw_pyvis_graph(G)
            tmpfile = "graph_vis.html"
            net.save_graph(tmpfile)
            st.components.v1.html(open(tmpfile, 'r', encoding='utf-8').read(), height=600)




# Created and edited by: JaraOps