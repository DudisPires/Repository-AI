import random
import time
import matplotlib.pyplot as plt


tempos_tarefas = [
    12, 5, 9, 7, 4, 11, 8, 6, 10, 3,
    7, 9, 5, 6, 4, 8, 12, 3, 7, 10
]

num_tarefas = len(tempos_tarefas)
num_maquinas = 5


populacao_tamanho = 50
geracoes = 100
taxa_crossover = 0.8
taxa_mutacao = 0.1


def avaliar(solucao):
    carga_maquinas = [0] * num_maquinas
    for tarefa_id, maquina_id in enumerate(solucao):
        carga_maquinas[maquina_id] += tempos_tarefas[tarefa_id]
    return max(carga_maquinas)


def gerar_populacao():
    return [
        [random.randint(0, num_maquinas - 1) for _ in range(num_tarefas)]
        for _ in range(populacao_tamanho)
    ]


def selecao_torneio(populacao, k=3):
    candidatos = random.sample(populacao, k)
    return min(candidatos, key=avaliar)


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

    for _ in range(geracoes):
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

print("\n------------------------------------------------------------")
print("ATRIBUIÇÃO FINAL DE TAREFAS ÀS MÁQUINAS")
print("------------------------------------------------------------\n")

alocacao_por_maquina = [[] for _ in range(num_maquinas)]
carga_maquinas = [0] * num_maquinas

for tarefa_id, maquina_id in enumerate(solucao):
    alocacao_por_maquina[maquina_id].append(tarefa_id + 1)
    carga_maquinas[maquina_id] += tempos_tarefas[tarefa_id]

for i in range(num_maquinas):
    print(f"Máquina {i+1}: Tarefas {alocacao_por_maquina[i]}, Tempo total: {carga_maquinas[i]}")

print(f"VALOR FINAL DO MAKESPAN: {makespan}")
print(f"TEMPO DE EXECUÇÃO DO ALGORITMO: {tempo_total:.2f} segundos\n\n")

plt.figure()
plt.plot(historico, marker='o')
plt.title("Evolução do Makespan por Geração")
plt.xlabel("Geração")
plt.ylabel("Makespan")
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_makespan.png")
plt.show()
