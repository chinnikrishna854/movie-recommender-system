# Movie Recommendation System

A content-based movie recommendation system using TF-IDF vectorization and cosine similarity on the TMDB movies dataset, with fuzzy title matching for user input.

---

## Overview

This project builds a movie recommender that suggests similar films based on shared genres, keywords, taglines, cast, and director. Given any movie title (with fuzzy matching for typos), it returns the 30 most similar movies ranked by cosine similarity.

---

## Dataset

**File:** `movies.csv`  
**Size:** 4,803 movies × 24 columns

**Key columns used for recommendations:**

| Column | Description |
|--------|-------------|
| `genres` | Genre tags (e.g. Action, Adventure) |
| `keywords` | Plot keywords |
| `tagline` | Movie tagline |
| `cast` | Main cast members |
| `director` | Director name |

Other columns include `title`, `overview`, `popularity`, `vote_average`, `vote_count`, `budget`, `runtime`, etc.

---

## How It Works

### 1. Preprocessing
- Load CSV into a pandas DataFrame
- Drop rows with null values (5 missing in `track_name`, `track_artist`, `track_album_name`)
- Fill null values in the 5 selected feature columns with empty strings

### 2. Feature Engineering
Concatenate the 5 selected text features into a single combined string per movie:
```
combined = genres + keywords + tagline + cast + director
```

### 3. TF-IDF Vectorization
Transform the combined text into a TF-IDF feature matrix — each movie becomes a vector in a high-dimensional term space.

### 4. Cosine Similarity
Compute pairwise cosine similarity across all 4,803 movies, producing a 4803×4803 similarity matrix.

### 5. Recommendation
Given a movie title input:
1. Use `difflib.get_close_matches` to find the closest title match (handles typos/partial names)
2. Look up the movie's row in the similarity matrix
3. Sort all movies by similarity score (descending)
4. Return the top 30 results

---

## Example Output

```
Input: "Iron Man"

Movies suggested for you:
1. Iron Man
2. Iron Man 2
3. Iron Man 3
4. Avengers: Age of Ultron
5. The Avengers
6. Captain America: Civil War
7. Captain America: The Winter Soldier
8. Ant-Man
9. X-Men
10. Guardians of the Galaxy
...
```

---

## Visualizations

- **Pairplot** — pairwise relationships across all numeric features
- **Correlation heatmap** — 10×8 annotated heatmap of feature correlations
- **Word cloud** — most frequent words in movie titles

---

## Usage

```python
movie_name = input('Enter your favourite movie name: ')

list_of_all_titles = movies_data['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

for i, movie in enumerate(sorted_similar_movies[:30], 1):
    title = movies_data[movies_data.index == movie[0]]['title'].values[0]
    print(i, '.', title)
```

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
difflib  (standard library)
```

---

## Notes

- The fuzzy matching via `difflib.get_close_matches` handles case-insensitive input and partial titles (e.g. "iron man" → "Iron Man").
- Recommendations are purely content-based — no user ratings or collaborative filtering involved.
- Similarity is driven mainly by shared cast and director, since those fields tend to have strong overlap within franchises (e.g. Marvel movies cluster together naturally).
- Extending the feature set to include `overview` (plot summary) via TF-IDF could improve variety in recommendations.
