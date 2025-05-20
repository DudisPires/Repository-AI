import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Dados do problema
num_tarefas = 40
num_maquinas = 5

# Tabela de tarefas: (tempo_processamento, prioridades)
tarefas = {
    1: (25, [2, 3]),
    2: (17, []),
    3: (20, [4, 5]),
    4: (12, []),
    5: (28, [6]),
    6: (16, []),
    7: (22, []),
    8: (15, [9]),
    9: (18, []),
    10: (30, [11]),
    11: (19, []),
    12: (23, [13]),
    13: (11, []),
    14: (27, []),
    15: (14, [16]),
    16: (21, []),
    17: (17, [18]),
    18: (24, []),
    19: (26, [20]),
    20: (19, []),
    21: (13, [22]),
    22: (10, []),
    23: (15, [24]),
    24: (28, []),
    25: (22, [26]),
    26: (18, []),
    27: (21, [28]),
    28: (30, []),
    29: (23, [30]),
    30: (17, []),
    31: (25, [32]),
    32: (20, []),
    33: (22, [34]),
    34: (16, []),
    35: (18, [36]),
    36: (12, []),
    37: (26, [38]),
    38: (14, []),
    39: (27, [40]),
    40: (11, [])
}

# Pré-processamento: criar grafo de precedência e ordem topológica
def construir_grafo_precedencia():
    grafo = defaultdict(list)
    for tarefa, (_, prioridades) in tarefas.items():
        for p in prioridades:
            grafo[tarefa].append(p)
    return grafo

def ordenacao_topologica():
    grafo = construir_grafo_precedencia()
    visitados = set()
    ordem = []
    
    def dfs(tarefa):
        if tarefa not in visitados:
            visitados.add(tarefa)
            for vizinho in grafo[tarefa]:
                dfs(vizinho)
            ordem.append(tarefa)
    
    for t in range(1, num_tarefas + 1):
        dfs(t)
    
    return ordem

ordem_topologica = ordenacao_topologica()

# Representação da solução: lista de listas, cada sublista é uma máquina com tarefas ordenadas
def gerar_solucao_inicial():
    solucao = [[] for _ in range(num_maquinas)]
    tarefas_disponiveis = ordem_topologica.copy()
    
    while tarefas_disponiveis:
        maq = random.randint(0, num_maquinas - 1)
        # Escolher uma tarefa que pode ser alocada (suas precedências já foram alocadas)
        for t in tarefas_disponiveis:
            pode_alocar = True
            for p in tarefas[t][1]:
                if any(p in m for m in solucao):
                    continue
                else:
                    pode_alocar = False
                    break
            if pode_alocar:
                solucao[maq].append(t)
                tarefas_disponiveis.remove(t)
                break
        else:
            # Se não encontrou tarefa para alocar, força alocação (pode violar restrições)
            t = random.choice(tarefas_disponiveis)
            solucao[maq].append(t)
            tarefas_disponiveis.remove(t)
    
    return solucao

def calcular_makespan(solucao):
    tempos = [0] * num_maquinas
    tempos_conclusao = {t: 0 for t in range(1, num_tarefas + 1)}
    
    # Primeiro passada: calcular tempos de conclusão respeitando precedências
    for maq in range(num_maquinas):
        tempo_atual = 0
        for t in solucao[maq]:
            # Verificar se todas as precedências foram concluídas
            tempo_inicio = tempo_atual
            for p in tarefas[t][1]:
                tempo_inicio = max(tempo_inicio, tempos_conclusao[p])
            
            tempos_conclusao[t] = tempo_inicio + tarefas[t][0]
            tempo_atual = tempos_conclusao[t]
    
    # Segunda passada: recalcular tempos das máquinas considerando dependências entre máquinas
    tempos = [0] * num_maquinas
    for maq in range(num_maquinas):
        tempo_atual = 0
        for t in solucao[maq]:
            # Verificar precedências em outras máquinas
            tempo_inicio = tempo_atual
            for p in tarefas[t][1]:
                tempo_inicio = max(tempo_inicio, tempos_conclusao[p])
            
            tempos_conclusao[t] = tempo_inicio + tarefas[t][0]
            tempo_atual = tempos_conclusao[t]
        tempos[maq] = tempo_atual
    
    return max(tempos), tempos_conclusao

def fitness(solucao):
    makespan, _ = calcular_makespan(solucao)
    return makespan

def crossover(pai1, pai2):
    filho1 = [[] for _ in range(num_maquinas)]
    filho2 = [[] for _ in range(num_maquinas)]
    
    # Crossover em um ponto
    ponto_corte = random.randint(1, num_tarefas - 1)
    
    # Achatar as soluções
    flat_pai1 = [t for maq in pai1 for t in maq]
    flat_pai2 = [t for maq in pai2 for t in maq]
    
    # Criar filhos
    filho1_flat = flat_pai1[:ponto_corte] + [t for t in flat_pai2 if t not in flat_pai1[:ponto_corte]]
    filho2_flat = flat_pai2[:ponto_corte] + [t for t in flat_pai1 if t not in flat_pai2[:ponto_corte]]
    
    # Distribuir nas máquinas mantendo a ordem
    for t in filho1_flat:
        maq = random.randint(0, num_maquinas - 1)
        filho1[maq].append(t)
    
    for t in filho2_flat:
        maq = random.randint(0, num_maquinas - 1)
        filho2[maq].append(t)
    
    return filho1, filho2

def mutacao(solucao):
    # Escolher duas tarefas em máquinas diferentes e trocá-las
    maq1, maq2 = random.sample(range(num_maquinas), 2)
    if solucao[maq1] and solucao[maq2]:
        idx1 = random.randint(0, len(solucao[maq1]) - 1)
        idx2 = random.randint(0, len(solucao[maq2]) - 1)
        solucao[maq1][idx1], solucao[maq2][idx2] = solucao[maq2][idx2], solucao[maq1][idx1]
    return solucao

def busca_local(solucao):
    melhor_solucao = [maq.copy() for maq in solucao]
    melhor_fitness = fitness(melhor_solucao)
    
    for _ in range(10):  # Número de tentativas de melhoria
        nova_solucao = [maq.copy() for maq in solucao]
        maq1, maq2 = random.sample(range(num_maquinas), 2)
        if nova_solucao[maq1] and nova_solucao[maq2]:
            idx1 = random.randint(0, len(nova_solucao[maq1]) - 1)
            idx2 = random.randint(0, len(nova_solucao[maq2]) - 1)
            nova_solucao[maq1][idx1], nova_solucao[maq2][idx2] = nova_solucao[maq2][idx2], nova_solucao[maq1][idx1]
            
            novo_fitness = fitness(nova_solucao)
            if novo_fitness < melhor_fitness:
                melhor_solucao = nova_solucao
                melhor_fitness = novo_fitness
    
    return melhor_solucao

def algoritmo_memetico(tamanho_populacao=50, geracoes=100, prob_mutacao=0.1, prob_busca_local=0.2):
    start_time = time.time()
    historico_fitness = []
    
    # Inicializar população
    populacao = [gerar_solucao_inicial() for _ in range(tamanho_populacao)]
    fitness_pop = [fitness(ind) for ind in populacao]
    
    melhor_idx = np.argmin(fitness_pop)
    melhor_solucao = [maq.copy() for maq in populacao[melhor_idx]]
    melhor_fitness = fitness_pop[melhor_idx]
    historico_fitness.append(melhor_fitness)
    
    for geracao in range(geracoes):
        # Seleção por torneio
        nova_populacao = []
        for _ in range(tamanho_populacao // 2):
            # Torneio binário
            candidatos = random.sample(range(tamanho_populacao), 2)
            pai1 = populacao[min(candidatos, key=lambda x: fitness_pop[x])]
            
            candidatos = random.sample(range(tamanho_populacao), 2)
            pai2 = populacao[min(candidatos, key=lambda x: fitness_pop[x])]
            
            # Crossover
            filho1, filho2 = crossover(pai1, pai2)
            
            # Mutação
            if random.random() < prob_mutacao:
                filho1 = mutacao(filho1)
            if random.random() < prob_mutacao:
                filho2 = mutacao(filho2)
            
            nova_populacao.extend([filho1, filho2])
        
        # Aplicar busca local em alguns indivíduos
        for i in range(len(nova_populacao)):
            if random.random() < prob_busca_local:
                nova_populacao[i] = busca_local(nova_populacao[i])
        
        # Avaliar nova população
        nova_fitness = [fitness(ind) for ind in nova_populacao]
        
        # Elitismo: manter o melhor da geração anterior
        pior_idx = np.argmax(nova_fitness)
        if nova_fitness[pior_idx] > melhor_fitness:
            nova_populacao[pior_idx] = melhor_solucao
            nova_fitness[pior_idx] = melhor_fitness
        
        # Atualizar população
        populacao = nova_populacao
        fitness_pop = nova_fitness
        
        # Atualizar melhor solução
        melhor_idx = np.argmin(fitness_pop)
        if fitness_pop[melhor_idx] < melhor_fitness:
            melhor_solucao = [maq.copy() for maq in populacao[melhor_idx]]
            melhor_fitness = fitness_pop[melhor_idx]
        
        historico_fitness.append(melhor_fitness)
        
        if geracao % 10 == 0:
            print(f"Geração {geracao}: Makespan = {melhor_fitness}")
    
    tempo_execucao = time.time() - start_time
    
    # Resultados finais
    makespan_final, tempos_conclusao = calcular_makespan(melhor_solucao)
    
    print("\n--- Resultados Finais ---")
    print(f"Makespan: {makespan_final}")
    print(f"Tempo de execução: {tempo_execucao:.2f} segundos")
    
    # Imprimir alocação de tarefas
    print("\nAlocação de Tarefas por Máquina:")
    for i, maq in enumerate(melhor_solucao):
        print(f"Máquina {i+1}: {maq}")
        print(f"Tempo da máquina {i+1}: {max(tempos_conclusao[t] for t in maq) if maq else 0}")
    
    # Plotar evolução do fitness
    plt.figure(figsize=(10, 5))
    plt.plot(historico_fitness)
    plt.title("Evolução do Makespan ao Longo das Gerações")
    plt.xlabel("Geração")
    plt.ylabel("Makespan")
    plt.grid(True)
    plt.show()
    plt.savefig("evolucao_makespan.png")

    return melhor_solucao, makespan_final, tempo_execucao

# Executar o algoritmo
solucao_otima, makespan, tempo = algoritmo_memetico(tamanho_populacao=50, geracoes=100)