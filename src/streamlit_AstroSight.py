#create web application with streamlit including knowledge graph, add AI for purpose driven summaries

from typing import List
from pathlib import Path
import pandas as pd
import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

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

#HELPER FUNCTIONS
def get_absolute_path(filename):
    # This gets the directory where the current script (streamlit_AstroSight.py) is located
    BASE_DIR = Path(__file__).parent
    # This constructs the full path: /AstroSight/src/final_analyzed_data.csv
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

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data
def summarize_with_lexrank(text, sentence_count=3):


    if not text:
        return "No text available for summarization."

    # Check for minimum text length to avoid errors
    if len(text.split('.')) < sentence_count:
        sentence_count = len(text.split('.'))

    try:
        # 1. Set up the parser and tokenizer
        parser = PlaintextParser.from_string(text, Tokenizer("english"))

        # 2. Instantiate the summarizer
        summarizer = LexRankSummarizer()

        # 3. Generate the summary (e.g., top 3 sentences)
        summary = summarizer(parser.document, sentence_count)

        # 4. Join the sentences back into a single string
        return " ".join([str(sentence) for sentence in summary])

    except Exception as e:
        # Catch any potential NLTK/Sumy errors gracefully
        return f"Error during local summarization: {e}"

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
st.title("AstroSight by The Chilean Orbital")

#Sidebar: Data files and quick settings
st.sidebar.header("Data & Settings")
datafile = st.sidebar.text_input("Data CSV path", value=DEFAULT_DATAFILE)
df = load_data()

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
st.sidebar.markdown("github.com/JaraOps | The Chilean Orbital")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')



def get_tag_similarity(df, reference_title):

    ref_keywords_raw = df[df['Title'] == reference_title]['Top_Keywords'].iloc[0]

    if isinstance(ref_keywords_raw, str):
        ref_keywords = set([k.strip() for k in ref_keywords_raw.split(';') if k.strip()])
    else:
        ref_keywords = set()

    similarity_scores = {}


    for index, row in df.iterrows():
        title = row['Title']
        if title != reference_title:
            other_keywords_raw = row['Top_Keywords']

            if isinstance(other_keywords_raw, str):
                other_keywords = set([k.strip() for k in other_keywords_raw.split(';') if k.strip()])
            else:
                other_keywords = set()

            score = len(ref_keywords.intersection(other_keywords))
            similarity_scores[title] = score

    TOP_N_RESULTS = 5
    top_similar = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)[:TOP_N_RESULTS]
    return top_similar



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

    to_export = st.multiselect("Select rows to export (titles)", options=filtered['Title'].tolist(), key='export_titles_selector')

    outpath_filename = st.text_input("Output filename", value="export_selected.csv")

    if to_export:
        outdf = filtered[filtered['Title'].isin(to_export)]

        csv_data = convert_df_to_csv(outdf)

        st.download_button(
            label="Download Selected CSV",
            data=csv_data,
            file_name='export_selected_data.csv',
            mime='text/csv',
            key='download_csv'  # Unique key for the widget
        )
    else:
        st.info("Select titles above to enable the download button.")

with col2:
    st.subheader("Article details & tools")
    st.markdown(f"### {sel}")
    text = filtered.loc[idx, TEXT_COL]
    st.markdown("**Keywords (TF-IDF):**")
    st.write(filtered.loc[idx, KEYWORDS_COL])
    st.markdown("**Abstract / Selected text (cleaned):**")

    st.text_area("Article text", value=text if pd.notna(text) else "", height=300, label_visibility="collapsed")

    st.header("Article Details & Summarization")
    titles_list = df['Title'].unique().tolist()
    selected_title = st.selectbox(
        "Select an Article to Summarize:",
        options=titles_list,
        key='summary_title_selector'
    )
    st.markdown("---")
    st.subheader("AI Summary & Q&A")

    selected_text = df[df['Title'] == selected_title]['Clean_Text_for_NLP'].iloc[0]
    SENTENCE_COUNT = 3

    if st.button("Generate Summary"):
        summary = summarize_with_lexrank(selected_text, sentence_count=SENTENCE_COUNT)

        st.subheader(f"Article Summary ({SENTENCE_COUNT} Sentences, LexRank Model)")
        st.info(summary)

    st.markdown("---")
    st.subheader("Knowledge Graph")
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

    st.markdown("---")
    st.header("Inter-document Tag Overlap Engine")
    st.write("Instantly links papers based on shared core topics extracted by our NLP pipeline.")

    # Define the selectbox for user input
    reference_title = st.selectbox(
        "Select an Article to find look-alikes:",
        options=df['Title'].tolist(),
        key='tag_ref_selector'
    )

    if reference_title:
        top_similar = get_tag_similarity(df, reference_title)

        results_list = []
        for title, score in top_similar:
            # **MUST FIX:** Ensure 'Theme_category' is correct (lowercase 'c')
            # This line relies on your corrected column name!
            theme = df[df['Title'] == title]['Theme_category'].iloc[0]

            results_list.append({
                "Title": title,
                "Theme": theme,
                "Shared Keywords": score
            })

        st.dataframe(results_list)

# Created and edited by: JaraOps