# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import combinations

def get_frequent_items(transactions, min_support_count):
    """
    Calcula os itens frequentes e suas contagens de suporte.

    Args:
        transactions (list of set): Lista de transações, onde cada transação é um conjunto de itens.
        min_support_count (int): Contagem mínima de suporte para um item ser considerado frequente.

    Returns:
        dict: Dicionário onde as chaves são os itens frequentes e os valores são suas contagens de suporte.
    """
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1

    frequent_items = {
        item: count
        for item, count in item_counts.items()
        if count >= min_support_count
    }
    return frequent_items

def build_tid_lists(transactions, frequent_items):
    """
    Constrói as TID-lists (Transaction ID lists) para os itens frequentes.

    Args:
        transactions (list of set): Lista de transações.
        frequent_items (dict): Dicionário de itens frequentes e suas contagens.

    Returns:
        dict: Dicionário onde as chaves são os itens e os valores são conjuntos de IDs de transação
              onde o item aparece.
    """
    tid_lists = defaultdict(set)
    for i, transaction in enumerate(transactions):
        for item in transaction:
            if item in frequent_items:
                tid_lists[item].add(i)
    return tid_lists

def get_maximal_frequent_itemsets_recursive(
    current_itemset, current_tid_list, tid_lists, min_support_count, all_frequent_itemsets
):
    """
    Função recursiva para encontrar itemsets frequentes maximais.

    Args:
        current_itemset (frozenset): O itemset atual sendo processado.
        current_tid_list (set): A TID-list do itemset atual.
        tid_lists (dict): Dicionário de TID-lists para itens individuais.
        min_support_count (int): Contagem mínima de suporte.
        all_frequent_itemsets (list): Lista para armazenar todos os itemsets frequentes encontrados (para verificação de maximalidade).

    Returns:
        list: Lista de itemsets frequentes maximais encontrados a partir do itemset atual.
    """
    maximal_itemsets = []
    is_maximal = True # Assume que é maximal até que um superconjunto frequente seja encontrado

    # Ordena os itens para garantir uma ordem canônica e evitar duplicatas/combinações desnecessárias
    sorted_items = sorted([item for item in tid_lists.keys() if item not in current_itemset])

    for item in sorted_items:
        # Considera apenas itens que aparecem "depois" do último item no current_itemset
        # para evitar gerar permutações do mesmo conjunto (se current_itemset não estiver vazio)
        if current_itemset and item <= max(current_itemset, default=""): # type: ignore
            continue

        new_tid_list = current_tid_list.intersection(tid_lists[item])
        if len(new_tid_list) >= min_support_count:
            new_itemset = current_itemset.union({item})
            all_frequent_itemsets.append(new_itemset) # Adiciona à lista de todos os frequentes

            # Se encontrarmos um superconjunto frequente, o atual não é maximal
            # No entanto, a lógica de maximalidade é mais complexa e geralmente verificada no final.
            # Esta chamada recursiva explora mais a fundo.
            sub_maximal_itemsets = get_maximal_frequent_itemsets_recursive(
                new_itemset, new_tid_list, tid_lists, min_support_count, all_frequent_itemsets
            )
            maximal_itemsets.extend(sub_maximal_itemsets)

            # Se a chamada recursiva encontrou algum itemset maximal,
            # significa que `current_itemset` é um subconjunto de um itemset frequente maior.
            # Portanto, `current_itemset` em si não é maximal se `sub_maximal_itemsets` não estiver vazio.
            if sub_maximal_itemsets:
                is_maximal = False


    # Adiciona o current_itemset se ele for frequente e nenhum superconjunto frequente foi encontrado
    # a partir desta ramificação da recursão.
    # A verificação final de maximalidade acontece fora desta função recursiva.
    if is_maximal and len(current_tid_list) >= min_support_count and current_itemset:
        # Esta condição é uma heurística. A verdadeira verificação de maximalidade
        # compara com todos os outros itemsets frequentes.
        # Por enquanto, vamos adicionar e filtrar depois.
        # Ou, melhor, a lógica de maximalidade deve ser: se não há extensões frequentes,
        # então este é um candidato a maximal.
        # A forma mais robusta é coletar todos os frequentes e depois filtrar os maximais.
        pass # A decisão de adicionar é adiada para a função principal `max_eclat`

    return maximal_itemsets


def max_eclat(transactions, min_support):
    """
    Encontra todos os itemsets frequentes maximais usando o algoritmo Max Eclat.

    Args:
        transactions (list of set): Lista de transações.
        min_support (float): Suporte mínimo (proporção entre 0 e 1).

    Returns:
        list: Lista de frozensets, onde cada frozenset é um itemset frequente maximal.
    """
    num_transactions = len(transactions)
    if num_transactions == 0:
        return []
    min_support_count = min_support * num_transactions

    # 1. Encontrar itens frequentes de tamanho 1 e suas contagens
    frequent_1_itemsets_counts = get_frequent_items(transactions, min_support_count)
    if not frequent_1_itemsets_counts:
        return []

    # 2. Construir TID-lists para itens frequentes de tamanho 1
    tid_lists = build_tid_lists(transactions, frequent_1_itemsets_counts)

    # Lista para armazenar todos os itemsets frequentes encontrados
    all_frequent_itemsets_with_support = []

    # Adiciona os itemsets de tamanho 1 que são frequentes
    for item, tid_list in tid_lists.items():
        if len(tid_list) >= min_support_count:
            all_frequent_itemsets_with_support.append(
                (frozenset({item}), len(tid_list))
            )

    # 3. Geração recursiva de itemsets frequentes
    # Ordena os itens para processamento consistente
    sorted_items = sorted(frequent_1_itemsets_counts.keys())

    # Usaremos uma lista para coletar todos os itemsets frequentes e depois filtrar os maximais
    collected_frequent_itemsets = {
        frozenset({item}): tid_list
        for item, tid_list in tid_lists.items()
        if len(tid_list) >= min_support_count
    }

    # Fila para a busca em profundidade (DFS) ou largura (BFS)
    # Cada elemento da fila: (itemset, tid_list_do_itemset)
    queue = [
        (frozenset({item}), tid_lists[item])
        for item in sorted_items
        if len(tid_lists[item]) >= min_support_count
    ]

    head = 0
    while head < len(queue):
        current_itemset, current_tid_list = queue[head]
        head += 1

        # Considerar extensões do current_itemset
        # Para evitar duplicatas e trabalho redundante, só estendemos com itens
        # que são lexicograficamente maiores que o "último" item no current_itemset.
        last_item_in_current = max(current_itemset) if current_itemset else "" # type: ignore

        for item_to_add in sorted_items:
            if item_to_add not in current_itemset and item_to_add > last_item_in_current:
                # Interseccionar TID-lists
                # A TID-list do item_to_add é tid_lists[item_to_add]
                new_tid_list = current_tid_list.intersection(tid_lists[item_to_add])

                if len(new_tid_list) >= min_support_count:
                    new_itemset = current_itemset.union({item_to_add})
                    if new_itemset not in collected_frequent_itemsets:
                         collected_frequent_itemsets[new_itemset] = new_tid_list
                         queue.append((new_itemset, new_tid_list))


    # 4. Filtrar para obter apenas os itemsets maximais
    frequent_itemsets_list = list(collected_frequent_itemsets.keys())
    maximal_frequent_itemsets = []

    for i in range(len(frequent_itemsets_list)):
        itemset_i = frequent_itemsets_list[i]
        is_maximal = True
        for j in range(len(frequent_itemsets_list)):
            if i == j:
                continue
            itemset_j = frequent_itemsets_list[j]
            # Se itemset_i é um subconjunto próprio de itemset_j, então itemset_i não é maximal
            if itemset_i.issubset(itemset_j) and itemset_i != itemset_j:
                is_maximal = False
                break
        if is_maximal:
            maximal_frequent_itemsets.append(itemset_i)

    return maximal_frequent_itemsets


# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Base de dados de exemplo (lista de transações)
    # Cada transação é um conjunto de itens
    transactions_data = [
        {"A", "B", "C", "D"},
        {"A", "C", "D", "E"},
        {"A", "B", "D"},
        {"B", "E"},
        {"A", "B", "C", "D", "E"},
        {"B", "C", "E"},
        {"A", "D", "E"},
        {"A", "B", "C"},
    ]

    # Suporte mínimo (ex: 30% das transações)
    min_support_threshold = 0.3

    print(f"Base de Dados ({len(transactions_data)} transações):")
    for i, t in enumerate(transactions_data):
        print(f"  T{i+1}: {sorted(list(t))}") # type: ignore
    print(f"Suporte Mínimo: {min_support_threshold * 100}%")
    print("-" * 30)

    maximal_itemsets = max_eclat(transactions_data, min_support_threshold)

    print("\nItemsets Frequentes Maximais Encontrados:")
    if maximal_itemsets:
        for idx, itemset in enumerate(maximal_itemsets):
            # Calcula o suporte real para exibição
            count = 0
            for t in transactions_data:
                if itemset.issubset(t):
                    count +=1
            support_percentage = (count / len(transactions_data)) * 100
            print(f"  {idx+1}. {sorted(list(itemset))} (Suporte: {support_percentage:.2f}%, Contagem: {count})") # type: ignore
    else:
        print("  Nenhum itemset frequente maximal encontrado com o suporte mínimo especificado.")

    print("-" * 30)
    # Exemplo 2: Suporte mais alto para menos resultados
    min_support_threshold_2 = 0.5
    print(f"\nExemplo com Suporte Mínimo: {min_support_threshold_2 * 100}%")
    maximal_itemsets_2 = max_eclat(transactions_data, min_support_threshold_2)
    print("\nItemsets Frequentes Maximais Encontrados:")
    if maximal_itemsets_2:
        for idx, itemset in enumerate(maximal_itemsets_2):
            count = 0
            for t in transactions_data:
                if itemset.issubset(t):
                    count +=1
            support_percentage = (count / len(transactions_data)) * 100
            print(f"  {idx+1}. {sorted(list(itemset))} (Suporte: {support_percentage:.2f}%, Contagem: {count})") # type: ignore
    else:
        print("  Nenhum itemset frequente maximal encontrado com o suporte mínimo especificado.")

    # Exemplo 3: Base de dados diferente
    transactions_data_2 = [
        {'leite', 'pão', 'manteiga'},
        {'pão', 'manteiga', 'café'},
        {'leite', 'pão', 'café'},
        {'leite', 'manteiga'},
        {'pão', 'café'},
        {'leite', 'pão', 'manteiga', 'café'}
    ]
    min_support_threshold_3 = 0.4 # 40%
    print(f"\nBase de Dados 2 ({len(transactions_data_2)} transações):")
    for i, t in enumerate(transactions_data_2):
        print(f"  T{i+1}: {sorted(list(t))}") # type: ignore
    print(f"Suporte Mínimo: {min_support_threshold_3 * 100}%")
    print("-" * 30)
    maximal_itemsets_3 = max_eclat(transactions_data_2, min_support_threshold_3)
    print("\nItemsets Frequentes Maximais Encontrados:")
    if maximal_itemsets_3:
        for idx, itemset in enumerate(maximal_itemsets_3):
            count = 0
            for t in transactions_data_2:
                if itemset.issubset(t):
                    count +=1
            support_percentage = (count / len(transactions_data_2)) * 100
            print(f"  {idx+1}. {sorted(list(itemset))} (Suporte: {support_percentage:.2f}%, Contagem: {count})") # type: ignore
    else:
        print("  Nenhum itemset frequente maximal encontrado com o suporte mínimo especificado.")
