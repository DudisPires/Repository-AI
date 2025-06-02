import pandas as pd
import ast
from tabulate import tabulate 
import textwrap 

# --------------------------
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

# --------------------------
def support(itemset, transactions):
    if not transactions:
        return 0
    return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)

# --------------------------
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

# --------------------------
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

# --------------------------
def get_movies_with_exact_genres(df, target_genre_set):
    return df[df['Gêneros'].apply(lambda movie_genre_set: target_genre_set.issubset(movie_genre_set))]

# --------------------------
def get_user_profile_from_watched_df(processed_watched_df):
    user_genres = set()
    if 'Gêneros' not in processed_watched_df.columns:
        return user_genres
    for genres_set in processed_watched_df['Gêneros']:
        if isinstance(genres_set, set):
            user_genres.update(genres_set)
    return user_genres

# --------------------------
# MAIN
def main():
    try:
        main_db_path = 'machine-learning/MaxEclat/world_imdb_movies_preprocessed.csv'
        df = pd.read_csv(main_db_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo da base de dados principal '{main_db_path}' não encontrado.")
        return
    except Exception as e:
        print(f"Erro ao ler a base de dados principal: {e}")
        return

    if 'genre' not in df.columns or 'title' not in df.columns:
        print("Erro: Coluna 'genre' ou 'title' não encontrada na base de dados principal.")
        return
        
    df['Gêneros_list'] = df['genre'].apply(safe_eval)
    df['Gêneros'] = df['Gêneros_list'].apply(set)
    df['title_normalized'] = df['title'].astype(str).str.strip().str.lower()

    transactions = df['Gêneros'].tolist()
    if not transactions:
        print("Nenhuma transação de gênero pôde ser criada a partir da base principal.")
        return

    min_support = 0.01
    print("Calculando conjuntos de gêneros frequentes e maximais da base de dados principal...")
    maximal_itemsets_global = max_eclat(transactions, min_support)
    print(f"Encontrados {len(maximal_itemsets_global)} conjuntos maximais de gêneros globalmente frequentes.\n")

    if not maximal_itemsets_global:
        print(f"Nenhum conjunto maximal de gêneros encontrado com suporte mínimo de {min_support}.")
        return
        
    user_csv_filename2 = input("Digite o NOME DO ARQUIVO CSV dos seus filmes assistidos (ex: meus_filmes.csv)\n(este arquivo deve estar no subdiretório 'machine-learning/MaxEclat/' e ter o mesmo formato da base de dados, sem cabeçalho):\n> ")
    user_csv_filename = 'machine-learning/MaxEclat/' + user_csv_filename2 #caminho correto 

    try:
        column_names_user_csv = ['title', 'year', 'rating_imdb', 'genre', 'language']
        user_watched_df = pd.read_csv(user_csv_filename, header=None, names=column_names_user_csv, dtype={'year': str, 'title': str})
        
        if 'genre' not in user_watched_df.columns or 'title' not in user_watched_df.columns:
             print(f"Erro: As colunas 'title' ou 'genre' não foram carregadas corretamente do arquivo {user_csv_filename}.")
             return
        user_watched_df['Gêneros_list'] = user_watched_df['genre'].apply(safe_eval)
        user_watched_df['Gêneros'] = user_watched_df['Gêneros_list'].apply(set)
        user_watched_df['title_normalized'] = user_watched_df['title'].astype(str).str.strip().str.lower()
        
    except FileNotFoundError:
        print(f"Erro: Arquivo CSV do usuário '{user_csv_filename}' não encontrado.")
        return
    except pd.errors.EmptyDataError:
        print(f"Erro: O arquivo CSV do usuário '{user_csv_filename}' está vazio.")
        return
    except Exception as e:
        print(f"Erro ao ler ou processar o CSV do usuário '{user_csv_filename}': {e}")
        return

    user_watched_titles_normalized = set(user_watched_df['title_normalized'])
    print("-" * 30) 

    user_profile_genres = get_user_profile_from_watched_df(user_watched_df)

    if not user_profile_genres:
        print("\nNão foi possível identificar um perfil de gêneros a partir dos seus filmes assistidos.")
        return
    
    print(f"\nSeu perfil de gêneros (baseado nos filmes assistidos): {sorted(list(user_profile_genres))}\n")
    print(f"=== Recomendações para você, baseadas nos seus filmes assistidos ===")
    
    relevant_scored_itemsets = []
    for itemset in maximal_itemsets_global:
        intersection_score = len(user_profile_genres.intersection(itemset))
        if intersection_score > 0:
            relevant_scored_itemsets.append({'itemset': itemset, 'score': intersection_score})

    if not relevant_scored_itemsets:
        print("Não foram encontrados conjuntos de gêneros frequentes relevantes para o seu perfil.")
        return

    relevant_scored_itemsets.sort(key=lambda x: x['score'], reverse=True)
    recommendations_made_count = 0
    MAX_TOTAL_RECOMMENDATIONS = 20
    MAX_RECS_PER_ITEMSET = 5

    for scored_itemset_info in relevant_scored_itemsets:
        if recommendations_made_count >= MAX_TOTAL_RECOMMENDATIONS:
            break
        current_itemset = scored_itemset_info['itemset']
        current_score = scored_itemset_info['score']
        print(f"\n--- Baseado no conjunto de gêneros (afinidade: {current_score}): {sorted(list(current_itemset))} ---")
        
        potential_recommendations_df = get_movies_with_exact_genres(df, current_itemset)

        if not potential_recommendations_df.empty:
            potential_recs_titles_normalized = set(potential_recommendations_df['title_normalized'])

            new_recommendations_df = potential_recommendations_df[
                ~potential_recommendations_df['title_normalized'].isin(user_watched_titles_normalized)
            ]

            if not new_recommendations_df.empty:
                new_recommendations_df = new_recommendations_df.sort_values(by='rating_imdb', ascending=False)
                num_to_show = min(len(new_recommendations_df), MAX_RECS_PER_ITEMSET, MAX_TOTAL_RECOMMENDATIONS - recommendations_made_count)
                if num_to_show > 0:
                    cols_to_display = ['title', 'year', 'rating_imdb', 'Gêneros_list']
                    if 'Gêneros_list' not in new_recommendations_df.columns: 
                        cols_to_display[-1] = 'genre'
                    
                    df_to_print = new_recommendations_df[cols_to_display].head(num_to_show).copy() 

                    title_max_w = 35
                    year_max_w = 6       
                    rating_max_w = 8     
                    genres_text_w = 55   

                    genre_col_name = 'Gêneros_list' if 'Gêneros_list' in df_to_print.columns else 'genre'
                    
                    df_to_print[genre_col_name] = df_to_print[genre_col_name].apply(
                        lambda x: textwrap.fill(', '.join(x) if isinstance(x, list) else str(x), 
                                               width=genres_text_w, 
                                               break_long_words=False, 
                                               replace_whitespace=False) 
                    )
                    
                    col_widths_for_tabulate = [title_max_w, year_max_w, rating_max_w, genres_text_w + 2] 

                    print(tabulate(df_to_print, headers='keys', tablefmt='psql', 
                                   showindex=False, missingval="N/A",
                                   maxcolwidths=col_widths_for_tabulate))
                                        
                    recommendations_made_count += num_to_show
            else:
                print("Todos os filmes deste conjunto já foram assistidos por você ou não há filmes novos após o filtro.")
        else:
            print("Nenhum filme encontrado na base de dados principal para este conjunto exato de gêneros.")
            
    if recommendations_made_count == 0:
        print("\nNão foram encontradas novas recomendações de filmes para você no momento com base no seu histórico.")

if __name__ == "__main__":
    main()