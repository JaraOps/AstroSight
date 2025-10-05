#create thematic categorizer using thematic words and giving labels to each
import pandas as pd

#config
INPUT_CSV_FILE = "publications_with_keywords.csv"
OUTPUT_CSV_FILE = "final_analyzed_data.csv"
KEYWORD_COLUMN ="Top_Keywords"

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

def main_categorization_pipeline():
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        print(f"Data loaded successfully: {len(df)} publications.")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{INPUT_CSV_FILE}'.")
        return

    print("Assigning thematic categories...")

    # Apply the thematic assignment function
    df['Theme_Category'] = df[KEYWORD_COLUMN].apply(assign_theme)

    # Save final DataFrame
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')

    # Display results and top categories
    theme_counts = df['Theme_Category'].value_counts()

    print("\n\n Phase 4, Part A Complete")
    print(f"Final data saved to **{OUTPUT_CSV_FILE}**.")
    print("Top Thematic Breakdown")
    print(theme_counts.head().to_string())

if __name__ == "__main__":
    main_categorization_pipeline()

#Created and edited by: JaraOps