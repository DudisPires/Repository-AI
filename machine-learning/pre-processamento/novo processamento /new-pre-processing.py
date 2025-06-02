import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# 2. Carregamento da base de dados
arquivo_entrada = "machine-learning/pre-processamento/world_imdb_movies_top_movies_per_year.csv"
try:
    df = pd.read_csv(arquivo_entrada)
except FileNotFoundError:
    print(f"Erro: O arquivo '{arquivo_entrada}' não foi encontrado. Verifique o nome e o caminho do arquivo.")
    exit() 

# 3. Visualização inicial
print("Formato da base:", df.shape)
print("Colunas disponíveis:", df.columns)
print("Primeiras 5 linhas da base original:")
print(df.head())

# 4. Seleção das colunas de interesse (incluindo 'star' e 'director')
colunas_interesse = ['title', 'year', 'rating_imdb', 'genre', 'language', 'star', 'director']
colunas_faltantes = [col for col in colunas_interesse if col not in df.columns]
if colunas_faltantes:
    print(f"Aviso: As seguintes colunas de interesse não foram encontradas na base de dados e serão ignoradas: {colunas_faltantes}")
    colunas_interesse = [col for col in colunas_interesse if col in df.columns]

if not colunas_interesse:
    print("Erro: Nenhuma das colunas de interesse especificadas foi encontrada no arquivo CSV. Encerrando o script.")
    exit()

df = df[colunas_interesse]

# 5. Remoção de valores ausentes
linhas_antes_dropna = len(df)
df.dropna(subset=colunas_interesse, inplace=True) 
df.reset_index(drop=True, inplace=True)
linhas_depois_dropna = len(df)
print(f"\n{linhas_antes_dropna - linhas_depois_dropna} linhas com valores ausentes foram removidas.")

# 6. Transformar 'genre' em lista de gêneros
if 'genre' in df.columns:
    df['genre'] = df['genre'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
else:
    print("Aviso: A coluna 'genre' não está presente no DataFrame após a seleção e remoção de NaNs. O passo de transformação de gênero será ignorado.")

# Cria um diretório para salvar os gráficos, se não existir
output_dir_graficos = "graficos_imdb"
if not os.path.exists(output_dir_graficos):
    os.makedirs(output_dir_graficos)
    print(f"Diretório '{output_dir_graficos}' criado para salvar os gráficos.")

# 7. Gráfico da distribuição das notas IMDb
if 'rating_imdb' in df.columns and not df['rating_imdb'].empty:
    plt.figure(figsize=(8, 5))
    sns.histplot(df['rating_imdb'], bins=20, kde=True, color='salmon')
    plt.title('Distribuição das notas IMDb')
    plt.xlabel('Nota IMDb')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.tight_layout()
    caminho_grafico1 = os.path.join(output_dir_graficos, "grafico_distribuicao_notas.png")
    plt.savefig(caminho_grafico1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico 'Distribuição das notas IMDb' salvo como '{caminho_grafico1}'")
else:
    print("\nAviso: Não foi possível gerar o gráfico de distribuição de notas.")

# 8. Gráfico dos idiomas mais comuns
if 'language' in df.columns and not df['language'].empty:
    plt.figure(figsize=(12, 6)) 
    top_languages = df['language'].value_counts().nlargest(10) 
    sns.barplot(x=top_languages.index, y=top_languages.values, palette='pastel')
    plt.title('Top 10 Idiomas Mais Frequentes nos Filmes')
    plt.ylabel('Quantidade de Filmes')
    plt.xlabel('Idioma')
    plt.xticks(rotation=45, ha="right") 
    plt.tight_layout()
    caminho_grafico2 = os.path.join(output_dir_graficos, "grafico_idiomas_frequentes.png")
    plt.savefig(caminho_grafico2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico 'Idiomas mais frequentes nos filmes' salvo como '{caminho_grafico2}'")
else:
    print("\nAviso: Não foi possível gerar o gráfico de idiomas mais comuns.")

# 9. Gráfico dos diretores mais frequentes
if 'director' in df.columns and not df['director'].empty:
    plt.figure(figsize=(12, 6))
    top_directors = df['director'].value_counts().nlargest(10)
    sns.barplot(x=top_directors.index, y=top_directors.values, palette='muted')
    plt.title('Top 10 Diretores Mais Frequentes')
    plt.ylabel('Quantidade de Filmes')
    plt.xlabel('Diretor')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    caminho_grafico3 = os.path.join(output_dir_graficos, "grafico_diretores_frequentes.png")
    plt.savefig(caminho_grafico3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico 'Diretores mais frequentes' salvo como '{caminho_grafico3}'")
else:
    print("\nAviso: Não foi possível gerar o gráfico de diretores mais frequentes.")

# 10. Visualizar o DataFrame final
print("\nPré-processamento concluído! Exibindo as primeiras 5 linhas dos dados finais:")
print(df.head())
print("\nFormato final do DataFrame:", df.shape)

# 11. Salvar o DataFrame pré-processado
nome_arquivo_saida = "world_imdb_movies_preprocessed.csv"
try:
    df.to_csv(nome_arquivo_saida, index=False, encoding='utf-8') 
    print(f"\nDataFrame pré-processado salvo com sucesso como '{nome_arquivo_saida}' no diretório atual.")
except Exception as e:
    print(f"\nErro ao salvar o DataFrame: {e}")
