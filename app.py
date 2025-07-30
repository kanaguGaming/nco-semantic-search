# app.py

from flask import Flask, render_template, request
from nlp.bert_model import BERTSemanticSearch
from nlp.translator import SmartTranslator
from nlp.fallback_handler import FallbackSearch

app = Flask(__name__)

# Load models once at startup
bert_search = BERTSemanticSearch('occupations.csv')
fallback_search = FallbackSearch('occupations.csv')
translator = SmartTranslator()  # Initialize translator

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        user_query = request.form['query']
        translated_query = translator.translate_to_english(user_query)

        results = bert_search.search(translated_query, top_k=25)

        # Optional fallback if all results are weak
        if all(float(r['Score']) < 0.4 for r in results):
            results = fallback_search.search(translated_query, top_k=10)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
