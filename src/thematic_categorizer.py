#create thematic categorizer using thematic words and giving labels to each
import pandas as pd

#config
INPUT_CSV_FILE = "publications_with_keyboard.csv"
OUTPUT_CSV_FILE = "final_analyzed_data.csv"
KEYWORD_COLUMN ="TFIDF_Keywords_Top_10"

#define keyword rules based on expected space biology topics
#the first matching category is assigned

THEMATIC_RULES = {
    "Human Factors & Health": [
        "human", "bone", "muscle", "cardio", "vision", "psych", "nutrition", "behavior", "sleep"
        ,"inmune", "countemeasure", "exercise", "vascular", "radiation", "cosmic", "hze",
        "cns", "cognitive"
    ],

    "Cell/Molecular Biology": [
        "gene", "rna", "protein", "dna", "expression", "molecular", "cell", "epigenetic", "signaling",
        "transcriptome", "proteome"
    ],

    "Plant & Food Production": [
        "plant", "root", "seed", "growth", "photosynthesis", "crop", "lettuce", "arabidopsis", "substrate", "soil"
    ],

    "Microbiology & Ecology": [
        "microbe", "bacteria", "fungi", "yeast", "biofilm", "pathogen", "viability", "ecosystem", "environment"
    ],

    "Engineering & Hardware": [
        "system", "hardware", "device", "reactor", "design", "sensor", "material", "monitor", "testbed"
    ],

    "General Space Biology": [
        "microgravity", "spaceflight", "iss", "gravity", "earth", "flight"
    ]
}

def assign_theme(keywords_str):
    if pd.isna(keywords_str):
        return "Unclassified"

    keywords = [k.strip().lower() for k in keywords_str.split(";")]

    assigned_themes = []

    for theme,rule_keywords in THEMATIC_RULES.items():
        if any(kw in keywords for kw in rule_keywords):
            assigned_themes.append(theme)
    if not assigned_themes:
        return "Other/Unclassified"
    return " | ".join(assigned_themes)

#def main_categorization_pipeline():
