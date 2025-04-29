import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_recommendations(query_title, df, tfidf_matrix, vectorizer, top_n=10):
    """
    Given a song title (string), return top_n similar songs from df
    as a list of (artist, song_title) tuples.
    """
    query_title = query_title.strip()

    # Try exact (case-insensitive) match on 'song_title'
    matches = df[df['song_title'].str.contains(query_title, case=False, na=False)]

    if not matches.empty:
        # Use the TF-IDF vector of the first matching row
        idx = matches.index[0]
        query_vec = tfidf_matrix[idx]
    else:
        idx = None
        # Vectorize the query string itself
        query_vec = vectorizer.transform([query_title])

    # Compute cosine similarity against all songs
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Exclude the exact match from results if found
    if idx is not None:
        cosine_sim[idx] = -np.inf

    # Get indices of top_n similar songs
    top_idx = np.argsort(cosine_sim)[-top_n:][::-1]

    # Select both 'artist' and 'song_title' columns for these indices
    recommended = df.iloc[top_idx][['artist', 'song_title']]

    # Return a list of (artist, song_title) tuples
    return list(recommended.itertuples(index=False, name=None))
