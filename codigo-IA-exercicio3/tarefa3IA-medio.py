import random
import time
import matplotlib.pyplot as plt

tarefa_tempos = [
    25, 17, 20, 12, 28, 16, 22, 15, 18, 30,
    19, 23, 11, 27, 14, 21, 17, 24, 26, 19,
    13, 10, 15, 28, 22, 18, 21, 30, 23, 17,
    25, 20, 22, 16, 18, 12, 26, 14, 27, 11
]

capacidades_maquinas = [18, 22, 12, 15, 28, 10]
num_maquinas = len(capacidades_maquinas)
num_tarefas = len(tarefa_tempos)
populacao_tamanho = 50
geracoes = 100
taxa_crossover = 0.8
taxa_mutacao = 0.1

def tempo_execucao(tarefa_id, maquina_id):
    return tarefa_tempos[tarefa_id] / capacidades_maquinas[maquina_id]

def avaliar(solucao):
    tempos_maquinas = [0] * num_maquinas
    for tarefa_id, maquina_id in enumerate(solucao):
        tempos_maquinas[maquina_id] += tempo_execucao(tarefa_id, maquina_id)
    return max(tempos_maquinas)

def gerar_populacao():
    return [
        [random.randint(0, num_maquinas - 1) for _ in range(num_tarefas)]
        for _ in range(populacao_tamanho)
    ]

def selecao_torneio(populacao, k=3):
    selecionados = random.sample(populacao, k)
    return min(selecionados, key=avaliar)

def crossover(pai1, pai2):
    ponto = random.randint(1, num_tarefas - 1)
    filho1 = pai1[:ponto] + pai2[ponto:]
    filho2 = pai2[:ponto] + pai1[ponto:]
    return filho1, filho2


def mutar(individuo):
    novo = individuo[:]
    if random.random() < taxa_mutacao:
        pos = random.randint(0, num_tarefas - 1)
        novo[pos] = random.randint(0, num_maquinas - 1)
    return novo

def busca_local(individuo):
    melhor = individuo[:]
    melhor_valor = avaliar(melhor)
    for i in range(num_tarefas):
        for m in range(num_maquinas):
            if melhor[i] != m:
                vizinho = melhor[:]
                vizinho[i] = m
                valor_vizinho = avaliar(vizinho)
                if valor_vizinho < melhor_valor:
                    melhor = vizinho
                    melhor_valor = valor_vizinho
    return melhor

def algoritmo_memetico():
    inicio = time.time()
    populacao = gerar_populacao()
    melhor_solucao = min(populacao, key=avaliar)
    melhor_makespan = avaliar(melhor_solucao)
    historico = [melhor_makespan]

    for geracao in range(geracoes):
        nova_populacao = []

        while len(nova_populacao) < populacao_tamanho:
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)

            if random.random() < taxa_crossover:
                filho1, filho2 = crossover(pai1, pai2)
            else:
                filho1, filho2 = pai1[:], pai2[:]

            filho1 = mutar(filho1)
            filho2 = mutar(filho2)

            filho1 = busca_local(filho1)
            filho2 = busca_local(filho2)

            nova_populacao.extend([filho1, filho2])

        populacao = nova_populacao[:populacao_tamanho]
        atual = min(populacao, key=avaliar)
        atual_makespan = avaliar(atual)

        if atual_makespan < melhor_makespan:
            melhor_solucao = atual
            melhor_makespan = atual_makespan

        historico.append(melhor_makespan)

    fim = time.time()
    tempo_execucao = fim - inicio
    return melhor_solucao, melhor_makespan, tempo_execucao, historico

solucao, makespan, tempo_total, historico = algoritmo_memetico()

print("Atribuição de tarefas às máquinas:")
for i, maquina in enumerate(solucao):
    print(f"Tarefa {i+1} -> Máquina {maquina+1}")

print(f"\nMakespan final: {makespan:.2f}")
print(f"Tempo de execução: {tempo_total:.2f} segundos")

alocacao_por_maquina = [[] for _ in range(num_maquinas)]
tempos_maquinas = [0] * num_maquinas

for tarefa_id, maquina_id in enumerate(solucao):
    alocacao_por_maquina[maquina_id].append(tarefa_id + 1)
    tempos_maquinas[maquina_id] += tempo_execucao(tarefa_id, maquina_id)

print("\nAlocação de Tarefas por Máquina:")
for maquina_id, tarefas in enumerate(alocacao_por_maquina):
    print(f"Máquina {maquina_id + 1}: {tarefas}")
    print(f"Tempo da máquina {maquina_id + 1}: {round(tempos_maquinas[maquina_id], 2)}\n")

plt.plot(historico)
plt.title("Evolução do Makespan")
plt.xlabel("Geração")
plt.ylabel("Makespan")
plt.grid(True)
plt.savefig("evolucao_makespan.png")
plt.show()  # remover se o ambiente utilizado for nao grafico 
