import pandas as pd
import pickle

movies = pd.read_csv("movies.csv")      # movieId, title, genres
links = pd.read_csv("links.csv")        # movieId, imdbId, tmdbId

movies_full = movies.merge(links, on="movieId", how="left")

with open("movies.pkl", "wb") as f:
    pickle.dump(movies_full, f)

print("movies.pkl rebuilt successfully!")
print(movies_full.head())
print(movies_full.columns)
