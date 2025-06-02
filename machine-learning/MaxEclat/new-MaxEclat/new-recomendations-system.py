import pandas as pd
import ast
from tabulate import tabulate 
import textwrap
import squarify
from collections import Counter
import matplotlib.pyplot as plt

def safe_eval(x):
    try:
        evaluated = ast.literal_eval(x)
        return evaluated
    except (ValueError, SyntaxError, TypeError):
        if isinstance(x, str):
            x_stripped = x.strip()
            if x_stripped.startswith('[') and x_stripped.endswith(']'):
                try:
                    items = [item.strip().replace("'", "").replace('"', '') for item in x_stripped[1:-1].split(',')]
                    return [item for item in items if item]
                except Exception:
                    return []
            elif '[' not in x and ']' not in x and x:
                return [x.strip().replace("'", "").replace('"', '')]
        return []

def support(itemset, transactions):
    if not transactions:
        return 0
    return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)

def max_eclat_recursive(prefix, items, transactions, min_sup, maximal_itemsets_dict):
    for i in range(len(items)):
        current_item = items[i]
        new_itemset = prefix.union({current_item})
        new_sup = support(new_itemset, transactions)
        if new_sup >= min_sup:
            suffix = [items[j] for j in range(i + 1, len(items))]
            max_eclat_recursive(new_itemset, suffix, transactions, min_sup, maximal_itemsets_dict)
    
    if prefix and all(support(prefix.union({item}), transactions) < min_sup for item in items):
        f_prefix = frozenset(prefix)
        if f_prefix not in maximal_itemsets_dict or len(prefix) > len(maximal_itemsets_dict[f_prefix]):
            maximal_itemsets_dict[f_prefix] = prefix

def max_eclat(transactions, min_sup):
    items = sorted(list(set(item for t in transactions for item in t)))
    frequent_single_items = [item for item in items if support({item}, transactions) >= min_sup]
    items = frequent_single_items
    maximal_itemsets_dict = {}
    max_eclat_recursive(set(), items, transactions, min_sup, maximal_itemsets_dict)
    
    collected_itemsets = list(maximal_itemsets_dict.values())
    truly_maximal_itemsets = []
    for i in range(len(collected_itemsets)):
        is_maximal_candidate = True
        for j in range(len(collected_itemsets)):
            if i == j:
                continue
            if collected_itemsets[i].issubset(collected_itemsets[j]) and collected_itemsets[i] != collected_itemsets[j]:
                is_maximal_candidate = False
                break
        if is_maximal_candidate:
            truly_maximal_itemsets.append(collected_itemsets[i])
    return truly_maximal_itemsets

def get_movies_with_itemset(df, itemset):
    return df[df['Itemset'].apply(lambda s: itemset.issubset(s))]

def get_user_profile(df):
    profile = set()
    for row in df.itertuples():
        if isinstance(row.Itemset, set):
            profile.update(row.Itemset)
    return profile

def process_itemset_columns(df):
    df['Gêneros_list'] = df['genre'].apply(safe_eval)
    df['Stars_list'] = df['star'].apply(safe_eval)
    df['Directors_list'] = df['director'].apply(safe_eval)
    
    df['Itemset'] = df.apply(lambda row: set(row.Gêneros_list or []) | set(row.Stars_list or []) | set(row.Directors_list or []), axis=1)
    df['title_normalized'] = df['title'].astype(str).str.strip().str.lower()

def plot_itemset_treemap(relevant_itemsets):
    item_counter = Counter()
    for info in relevant_itemsets:
        item_counter.update(info['itemset'])

    labels = list(item_counter.keys())
    sizes = list(item_counter.values())

    plt.figure(figsize=(12, 7))
    squarify.plot(sizes=sizes, label=labels, alpha=0.8)
    plt.axis('off')
    plt.title("Composição dos conjuntos frequentes recomendados")
    caminho_grafico = '/home/eduardo-monteiro/faculdade/IA/Repository-AI/graficos_imdb/composicao_conjuntos.png'
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico 'Composição dos Conjuntos' salvo como '{caminho_grafico}'")


def plot_affinity_vs_rating(relevant_itemsets, df, user_titles):
    affinities = []
    avg_ratings = []

    for info in relevant_itemsets:
        itemset = info['itemset']
        candidates_df = get_movies_with_itemset(df, itemset)
        new_recs = candidates_df[~candidates_df['title_normalized'].isin(user_titles)]

        if not new_recs.empty:
            affinities.append(info['score'])
            avg_ratings.append(new_recs['rating_imdb'].mean())

    plt.figure(figsize=(8, 5))
    plt.scatter(affinities, avg_ratings, alpha=0.7)
    plt.xlabel("Afinidade com o perfil")
    plt.ylabel("Média do rating IMDb dos recomendados")
    plt.title("Afinidade vs Qualidade das Recomendações")
    plt.grid(True)
    plt.tight_layout()
    caminho_grafico = '/home/eduardo-monteiro/faculdade/IA/Repository-AI/graficos_imdb/afinidade_vs_rating.png'
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico 'Afinidade vs Rating' salvo como '{caminho_grafico}'\n")


def main():
    try:
        main_db_path = '/home/eduardo-monteiro/faculdade/IA/Repository-AI/machine-learning/MaxEclat/new-MaxEclat/world_imdb_movies_preprocessed.csv'
        df = pd.read_csv(main_db_path)
    except Exception as e:
        print(f"Erro ao carregar a base principal: {e}")
        return

    required_cols = {'genre', 'title', 'director', 'star'}
    if not required_cols.issubset(df.columns):
        print(f"Erro: A base principal precisa conter as colunas: {required_cols}")
        return

    process_itemset_columns(df)
    transactions = df['Itemset'].tolist()
    
    min_support = 0.01
    print("\n🔍 Calculando conjuntos frequentes maximais com MaxEclat...")
    maximal_itemsets_global = max_eclat(transactions, min_support)
    print(f"✅ {len(maximal_itemsets_global)} conjuntos frequentes encontrados.\n")

    user_file = input("📂 Digite o nome do arquivo CSV dos seus filmes assistidos ou o path caso o arquivo esteja em outro diretorio:\n> ")
    user_path = '/home/eduardo-monteiro/faculdade/IA/Repository-AI/machine-learning/MaxEclat/new-MaxEclat/' + user_file

    try:
        colunas_user = ['title', 'year', 'rating_imdb', 'genre', 'language', 'star', 'director']
        user_df = pd.read_csv(user_path, header=None, names=colunas_user, dtype={'year': str})
        process_itemset_columns(user_df)
    except Exception as e:
        print(f"❌ Erro ao carregar os dados do usuário: {e}")
        return

    user_titles = set(user_df['title_normalized'])
    user_profile = get_user_profile(user_df)

    if not user_profile:
        print("⚠️ Perfil do usuário vazio.")
        return

    perfil_list = sorted(user_profile)
    print(f"\nSeu perfil inclui: {len(perfil_list)} itens.\n")
    print("------------------------------------------------------------")
    print("🎯 === Recomendações Personalizadas ===")

    relevant_itemsets = []
    for itemset in maximal_itemsets_global:
        score = len(user_profile.intersection(itemset))
        if score > 0:
            relevant_itemsets.append({'itemset': itemset, 'score': score})

    if not relevant_itemsets:
        print("⚠️ Nenhum conjunto relevante encontrado.")
        return

    relevant_itemsets.sort(key=lambda x: x['score'], reverse=True)

    MAX_TOTAL_RECOMMENDATIONS = 20
    MAX_RECS_PER_ITEMSET = 5
    count = 0

    for info in relevant_itemsets:
        if count >= MAX_TOTAL_RECOMMENDATIONS:
            break
        itemset = info['itemset']
        print(f"\n🔹 Afinidade: {info['score']} | Itens do conjunto: {', '.join(sorted(itemset))}")
        print("------------------------------------------------------------")
        candidates_df = get_movies_with_itemset(df, itemset)
        new_recs = candidates_df[~candidates_df['title_normalized'].isin(user_titles)]

        if not new_recs.empty:
            top_recs = new_recs.sort_values(by='rating_imdb', ascending=False).head(MAX_RECS_PER_ITEMSET)
            top_recs['Itemset'] = top_recs['Itemset'].apply(
                lambda x: textwrap.fill(', '.join(sorted(x)), width=55) if isinstance(x, set) else str(x)
            )
            print(tabulate(
                top_recs[['title', 'year', 'rating_imdb', 'director','Itemset']],
                headers='keys',
                tablefmt='psql',
                showindex=False
            ))
            count += len(top_recs)
        else:
            print("🔸 Nenhuma nova recomendação encontrada para este conjunto.")

    if count == 0:
        print("\n🚫 Nenhuma recomendação disponível.")

    plot_itemset_treemap(relevant_itemsets)
    plot_affinity_vs_rating(relevant_itemsets, df, user_titles)



if __name__ == "__main__":
    main()
