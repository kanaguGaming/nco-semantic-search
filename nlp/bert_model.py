# bert_model.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BERTSemanticSearch:
    def __init__(self, csv_path='occupations.csv'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = pd.read_csv(csv_path)
        self.df.fillna("", inplace=True)

        def format_code(code):
            code_str = str(code) if pd.notnull(code) else ""
            if '.' in code_str:
                before, after = code_str.split('.')
                return f"{before.zfill(4)}.{after.ljust(4, '0')}"
            elif code_str:
                return f"{code_str.zfill(4)}.0000"
            else:
                return "0000.0000"

        self.df['Code'] = self.df['Code'].apply(format_code)
        # Combine all useful text into a single field
        self.df['combined'] = self.df.apply(
            lambda row: f"{row['Job Title']} {row['Keywords']} {row['Description']}", axis=1
        )

        # Precompute embeddings
        self.embeddings = self.model.encode(self.df['combined'].tolist(), show_progress_bar=True)

    def search(self, query, top_k=25):
        query_embedding = self.model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = scores.argsort()[::-1][:top_k]
        results = []

        for idx in top_indices:
            results.append({
                "Code": self.df.iloc[idx]['Code'],
                "Job Title": self.df.iloc[idx]['Job Title'],
                "Score": round(float(scores[idx]), 4)
            })

        return results
