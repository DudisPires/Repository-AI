import pandas as pd

df = pd.read_csv("/home/eduardo-monteiro/faculdade/IA/Repository-AI/machine-learning/MaxEclat/Trabalho2/Pre-processamento/amazon_prime_movies.csv")  # substitua pelo nome real

df = df[['movie', 'plot', 'maturity_rating']].dropna()

print(df.head())
