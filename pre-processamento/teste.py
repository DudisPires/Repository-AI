# Recomendação de Filmes - Pré-processamento IMDb

# 1. Importação de bibliotecas
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Carregamento da base de dados
df = pd.read_csv("world_imdb_movies_top_movies_per_year.csv")

# 3. Visualização inicial
print("Formato da base:", df.shape)
print("Colunas disponíveis:", df.columns)
print(df.head())

# 4. Seleção das colunas de interesse
df = df[['title', 'year', 'rating_imdb', 'genre', 'language']]

# 5. Remoção de valores ausentes
df.dropna(subset=['title', 'year', 'rating_imdb', 'genre', 'language'], inplace=True)
df.reset_index(drop=True, inplace=True)

# 6. Transformar 'genre' em lista de gêneros
df['genre'] = df['genre'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

# 7. Visualização da distribuição das notas
plt.figure(figsize=(8, 5))
sns.histplot(df['rating_imdb'], bins=20, kde=True, color='salmon')
plt.title('Distribuição das notas IMDb')
plt.xlabel('Nota IMDb')
plt.ylabel('Frequência')
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_1.png", dpi=300, bbox_inches='tight')
plt.close()

# 8. Visualização de idiomas mais comuns
plt.figure(figsize=(10, 5))
top_languages = df['language'].value_counts().head(10)
sns.barplot(x=top_languages.index, y=top_languages.values, palette='pastel')
plt.title('Idiomas mais frequentes nos filmes')
plt.ylabel('Quantidade de Filmes')
plt.xlabel('Idioma')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("grafico_2.png", dpi=300, bbox_inches='tight')
plt.close()

# 9. Visualizar o DataFrame final
print("\nPré-processamento concluído! Exibindo os dados finais:")
print(df.head())
print("\nFormato final:", df.shape)
