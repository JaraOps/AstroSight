import pandas as pd
import time
from Bio import Entrez
from tqdm import tqdm


Entrez.email = "ignaciojara731@gmail.com"  # Replace with your email
Entreztool = "AstroSightforSpaceBiology"

InputFile = "SB_publication_PMC.csv"
NCBI_DB = "pmc"

BATCH_SIZE = 200

def loadandprepdata(csv_file):

    try:
        df = pd.read_csv(csv_file, header = None, names=["Title", "PMC_Link"])
        print(f" Data loaded successfully with {len(df)} publications found.")

        df["PMC_ID"] = df["PMC_Link"].str.extract(r"(PMC\d+)")

        df.dropna(subset=["PMC_ID"], inplace=True)
        print(f" PMC IDs extracted. {len(df)} valid entries remaining.")
        return df

    except FileNotFoundError:
        print(f" Error: File not found at '{csv_file}'. Check the file path.")
        return None
    except Exception as e:
        print(f" An error occurred during data loading: {e}")
        return None

def fetch_article_data(pmc_ids):
    if not pmc_ids:
        return ""
    print(f" Fetching data for {len(pmc_ids)} articles from NCBI...")
    try:
        handle = Entrez.efetch(db = NCBI_DB, id=pmc_ids, rettype="full", retmode= "xml")
        xml_data_bytes = handle.read()
        handle.close()

        xml_data = xml_data_bytes.decode("utf-8")
        time.sleep(1) # To respect NCBI rate limits
        return xml_data

    except Exception as e:
        print(f"\n Error fetching batch with IDs Starting at {pmc_ids[0]}: {e}")

        time.sleep(10) # Wait before retrying
        return ""

def main_api_pipeline():
    df = loadandprepdata(InputFile)

    if df is None:
        return

    #get list of PMC IDs
    pmcid_list = df["PMC_ID"].tolist()

    #List to hold all fetched XML data
    all_xml_data = []

    #process in batches
    for i in tqdm(range (0, len(pmcid_list), BATCH_SIZE), desc = "Fetching Batches"):
        batch_ids = pmcid_list[i:i+BATCH_SIZE]

        #Convert to list of ids coma separated string
        id_string = ",".join(batch_ids)

        xml_result = fetch_article_data(id_string)

        if xml_result:
            all_xml_data.append(xml_result)

    with open("raw_pmc_data.xml", "w", encoding= "utf-8") as f:
        f.write("".join(all_xml_data))

    print("\n\n Phase 1 Complete")
    print("All data fetched and saved to **raw_pmc_data.xml**")
    print("Proceed to Phase 2 to parse and clean the data.")

if __name__ == "__main__":
    main_api_pipeline()

# Created and edited by: JaraOps




