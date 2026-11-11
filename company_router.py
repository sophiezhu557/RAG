import csv
import re
from pathlib import Path
from rapidfuzz import fuzz

csv_path = Path(__file__).resolve().parent / "company_name.csv"

def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()

def load_company_mapping():
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        company_mapping = list(csv.DictReader(f))

    valid_sha1 = {
        path.stem 
        for path in (Path(__file__).resolve().parent / "PDFs").glob("*.pdf")
    }

    company_mapping = [
        row for row in company_mapping
        if row["sha1"] in valid_sha1
    ]
    return company_mapping


def identify_companies(query, company_mapping, threshold=80):
    query = f"{normalize(query)}"
    matchs = []

    for row in company_mapping:
        company = row["company_name"]

        match_score = fuzz.partial_ratio(normalize(company), query)

        if match_score >= threshold:
            matchs.append({
                "company_name": company,
                "sha1": row["sha1"]
            })
    return matchs


if __name__ == "__main__":
    companies = load_company_mapping()
    # question = "What was ENRG Elements' revenue in 2022?"

    # company_query_list = identify_companies(question, companies)