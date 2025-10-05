import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

#CONFIG
INPUT_XML_FILE = "raw_pmc_data.xml"
OUTPUT_CSV_FILE = "cleaned_publications_data.csv"

def extract_section_text(article_soup, section_title_patterns):

    body = article_soup.find("body")
    if not body:
        return ""

    for sec in body.find_all("sec"):
        title_tag = sec.find("title")
        if title_tag:
            title_text = title_tag.get_text(strip=True).lower()

            for pattern in section_title_patterns:
                if pattern in title_text:
                    return sec.get_text(separator='', strip=True)
    return ""


def parse_pmc_article(xml_string):
    data = {'PMC_ID': '', 'Title': '', 'Abstract': '', 'Results_Conclusion_Text': '', 'Clean_Text_for_NLP': ''}

    # Use the lxml parser for speed and robustness with XML
    soup = BeautifulSoup(xml_string, 'lxml')


    pmc_id_tag = soup.find('article-id', {'pub-id-type': 'pmc'})
    if pmc_id_tag:
        data['PMC_ID'] = pmc_id_tag.get_text(strip=True)
    if not pmc_id_tag:
        id_tag = soup.find('article-id', text=re.compile(r'PMC\d+'))
        if id_tag:
            pmc_id_tag = id_tag.get_text(strip=True)

    if not pmc_id_tag:
        ext_id_tag = soup.find('ext-link', {'ext-link-type': 'pmc'})
        if ext_id_tag:
            # Extract only the PMC number from the URL/link text
            match = re.search(r'(PMC\d+)', ext_id_tag.get_text(strip=True))
            if match:
                pmc_id_tag = match.group(0)

    # Store the final result
    data['PMC_ID'] = pmc_id_tag

    # Title
    title_tag = soup.find('article-title')
    if title_tag:
        data['Title'] = title_tag.get_text(strip=True)

    # Abstract
    abstract_tag = soup.find('abstract')
    if abstract_tag:
        data['Abstract'] = abstract_tag.get_text(separator=' ', strip=True)

    conclusion_patterns = ["conclusion", "conclusions", "summary and conclusions"]
    results_patterns = ["results", "experimental results"]

    conclusion_text = extract_section_text(soup, conclusion_patterns)
    results_text = extract_section_text(soup, results_patterns)

    data["Results_Conclusion_Text"] = (results_text + "" + conclusion_text).strip()


    nlp_text = data["Results_Conclusion_Text"]
    if not nlp_text and data["Abstract"]:
        nlp_text = data["Abstract"]

    nlp_text = re.sub(r"\(Fig\.\s\d+\)", "", nlp_text)
    nlp_text = re.sub(r"\s{2,}", "", nlp_text)
    data["Clean_Text_for_NLP"] = nlp_text.strip()

    return data

def main_parsing_pipeline():
    try:
        print(f" Reading raw XML data from {INPUT_XML_FILE}...")
        with open(INPUT_XML_FILE, 'r', encoding='utf-8') as f:
            raw_xml_data = f.read()

    except FileNotFoundError:
        print(f" Error: Input XML file not found at '{INPUT_XML_FILE}'. Did phase 1 complete successfully?")
        return

    soup = BeautifulSoup(f"<root>{raw_xml_data}</root>", "lxml")
    articles = soup.find_all("article")

    print(f" Found {len(articles)} articles to process.  Starting Parsing...")

    parsed_data_list = []

    for article in tqdm(articles, desc = "Parsing Articles"):
        article_data = parse_pmc_article(str(article))
        parsed_data_list.append(article_data)

    final_df = pd.DataFrame(parsed_data_list)

    final_df['Text_Length'] = final_df['Clean_Text_for_NLP'].str.len()

    final_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")

    print(f"\n\n Phase 2 Complete")
    print(f"Cleaned, structured data saved to **{OUTPUT_CSV_FILE}**")
    print(f"Total articles processed: {len(final_df)}")
    print(f"Average NLP Text Length: {final_df['Text_Length'].mean():.0f} characters")
    print("Ready for Phase 3: NLP and Keyphrase Extraction")

    return final_df

if __name__ == "__main__":
    main_parsing_pipeline()


#Created and edited by: JaraOps

