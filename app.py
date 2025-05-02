from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from recommendation import get_recommendations  # our helper

app = Flask(__name__)

# --- Data Loading and TF-IDF (runs once at startup) ---

# Load songs dataset
df = pd.read_csv('data/data.csv')
df['combined'] = df['artist'] + " " + df['song_title']

# Build TF-IDF matrix on combined artist+title text
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# --- Model Metrics ---
n_songs = df.shape[0]
vocab_size = len(vectorizer.vocabulary_)
# Average number of non-zero terms per document
terms_per_doc = (tfidf_matrix > 0).sum(axis=1)
avg_terms_per_doc = float(terms_per_doc.mean())

# Define your group members as a list of dictionaries
group_members = [
    {"name": "Chuwe Terrence", "reg": "R216888K", "program": "HAI"},
    {"name": "Pennyvia Michael Kanengoni", "reg": "R216896T", "program": "HAI"},
    {"name": "Emmah Zvakafa", "reg": "R216905E", "program": "HAI"},
    {"name": "Natasha Taundi", "reg": "R2110558", "program": "HAI"},
    {"name": "Tadiwanashe Musora", "reg": "R219287X", "program": "HAI"},
    {"name": "Tanatswa Shamu", "reg": "R216894Z", "program": "HAI"}
]


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renders the search form and, on POST, displays recommended songs,
    plus model metrics and team info.
    """
    recommendations = None
    query = None

    if request.method == 'POST':
        query = request.form.get('song_title', '').strip()
        if query:
            # Get top 10 recommendations as list of (artist, title) tuples
            recs = get_recommendations(query, df, tfidf_matrix, vectorizer, top_n=10)
            # Format as list of dicts for templating
            recommendations = [
                {'artist': artist, 'song_title': title}
                for artist, title in recs
            ]

    return render_template(
        'index.html',
        recommendations=recommendations,
        query=query,
        # pass metrics
        n_songs=n_songs,
        vocab_size=vocab_size,
        avg_terms_per_doc=round(avg_terms_per_doc, 1),
        # pass team info
        group_members=group_members
    )


if __name__ == '__main__':
    app.run(debug=True)
