# fallback_handler.py

from fuzzywuzzy import process
import pandas as pd

class FallbackSearch:
    def __init__(self, csv_path='occupations.csv'):
        self.df = pd.read_csv(csv_path)
        self.job_titles = self.df['Job Title'].fillna("").tolist()

    def search(self, query, top_k=10):
        matches = process.extract(query, self.job_titles, limit=top_k)
        results = []

        for match in matches:
            job_title = match[0]
            score = match[1]
            row = self.df[self.df['Job Title'] == job_title].iloc[0]
            results.append({
                "Code": row["Code"],
                "Job Title": job_title,
                "Score": f"Fuzzy {score}"
            })

        return results
