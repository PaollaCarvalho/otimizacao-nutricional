import numpy as np
import pandas as pd
from multiprocessing import Pool
import json
import time
##CONFIG##
n = 100
rng = np.random.default_rng(seed=42)
resultados = {}
#################
# mínimos (tem que somar <= 1)
minimos = np.array([0.2, 0.2, 0.2])  # carb, prot, lip
restante = 1 - minimos.sum()

# gera aleatório pro restante
valores = rng.dirichlet([1,1,1], size=n) * restante

# soma com mínimos
valores_final = valores + minimos
valores_final = np.round(valores_final, 2)
df = pd.DataFrame(valores_final, columns=["carboidratos", "proteinas", "lipideos"])
df["ID"] = range(1, n+1)
clientes = df[["ID", "carboidratos", "proteinas", "lipideos"]]
#################
alimentos=pd.read_csv(r"data\taco_reduzido.csv",sep=",")
#################
arroz_tabela=alimentos[alimentos["id"].isin([1,3,5])]
feijao_tabela=alimentos[alimentos["id"].isin([561,567])]
alimentos=alimentos[~alimentos["id"].isin([1,3,5,561,567,27])]  

alim_vals = alimentos[["carboidrato(g)", "proteina(g)", "lipideos(g)"]].values.astype(np.float32)
arroz_vals = arroz_tabela[["carboidrato(g)", "proteina(g)", "lipideos(g)"]].values.astype(np.float32)
feijao_vals = feijao_tabela[["carboidrato(g)", "proteina(g)", "lipideos(g)"]].values.astype(np.float32)

alim_ids = alimentos["id"].values
arroz_ids = arroz_tabela["id"].values
feijao_ids = feijao_tabela["id"].values
#################
def valido(kcal, alvo, tol=0.1):
    return np.all(np.abs(kcal - alvo) / alvo <= tol, axis=1)

def avaliar(individuos, alvo):
    total = (
        alim_vals[individuos[:, 2]] +
        alim_vals[individuos[:, 3]] +
        alim_vals[individuos[:, 4]] +
        arroz_vals[individuos[:, 0]] +
        feijao_vals[individuos[:, 1]]
    )

    kcal = total.copy()
    kcal[:, 0] *= 4
    kcal[:, 1] *= 4
    kcal[:, 2] *= 9

    kcal /= kcal.sum(axis=1, keepdims=True)

    erro = np.abs(kcal - alvo).sum(axis=1)

    return kcal, -erro

def inicializar(pop_size):
    individuos = np.empty((pop_size, 5), dtype=np.int32)

    individuos[:, 0] = np.random.randint(0, len(arroz_vals), pop_size)
    individuos[:, 1] = np.random.randint(0, len(feijao_vals), pop_size)
    individuos[:, 2:] = [np.random.choice(len(alim_vals), 3, replace=False) for _ in range(pop_size)]

    return individuos

def selecao(pop, fit, k=3):
    idx = np.random.randint(0, len(pop), (len(pop), k))
    melhores = idx[np.arange(len(pop)), np.argmax(fit[idx], axis=1)]
    return pop[melhores]

def crossover(pais):
    filhos = pais.copy()

    mask = np.random.rand(len(pais)) < 0.5

    filhos[mask, 2:] = pais[mask][:, [3, 4, 2]]  # troca alimentos

    return filhos

def mutacao(pop, taxa=0.1):
    # arroz (posição 0)
    mask_arroz = np.random.rand(len(pop)) < taxa
    pop[mask_arroz, 0] = np.random.randint(0, len(arroz_vals), mask_arroz.sum())

    # feijão (posição 1)
    mask_feijao = np.random.rand(len(pop)) < taxa
    pop[mask_feijao, 1] = np.random.randint(0, len(feijao_vals), mask_feijao.sum())

    for ind in pop:
        for i in range(2, 5):
            if np.random.rand() < taxa:
                usados = set(ind[2:])
                usados.remove(ind[i])

                possiveis = list(set(range(len(alim_vals))) - usados)
                ind[i] = np.random.choice(possiveis)

    return pop

def corrigir(individuos):
    for ind in individuos:
        alimentos = ind[2:]

        if len(set(alimentos)) < 3:  # tem duplicata
            usados = set()
            novos = []

            for a in alimentos:
                if a not in usados:
                    novos.append(a)
                    usados.add(a)
                else:
                    # sorteia um novo que não esteja repetido
                    possiveis = list(set(range(len(alim_vals))) - usados)
                    novo = np.random.choice(possiveis)
                    novos.append(novo)
                    usados.add(novo)

            ind[2:] = np.sort(novos)

    return individuos

def genetico_busca(alvo, total_avaliacoes, pop_size=100):
    pop = inicializar(pop_size)

    avaliacoes = 0
    epoca = 0
    encontrados_ids = []
    encontrados_kcal = []
    hist_diversidade = []
    encontrados_set = set() 
    hist_unicos = []
    hist_validos = []
    while avaliacoes < total_avaliacoes:
        kcal, fit = avaliar(pop, alvo)
        avaliacoes += len(pop)

        # 🔥 FILTRO IGUAL AO MONTE CARLO
        mask = valido(kcal, alvo)

        #################
        
        diversidade=len(np.unique(pop, axis=0)) / len(pop)
        hist_diversidade.append(diversidade)
        hist_validos.append(np.mean(mask))
        # ids=np.vstack(encontrados_ids) if encontrados_ids else np.empty((0, 5), dtype=np.int32)
        # encontrados=len(np.unique(ids, axis=0))
        # print(f"Época {epoca}: Avaliações={avaliacoes}, Diversidade={diversidade:.2f}, Taxa Válidos={np.mean(mask):.2f}, Encontrados={len(encontrados_set)}")
        ################
        if np.any(mask):
            validos_ids = pop[mask]
            for ind in validos_ids:
                encontrados_set.add(tuple(ind))
            hist_unicos.append(len(encontrados_set))
            encontrados_ids.append(pop[mask])
            encontrados_kcal.append(kcal[mask])
        else:
            hist_unicos.append(len(encontrados_set))
        # evolução
        pais = selecao(pop, fit)
        filhos = crossover(pais)
        pop = mutacao(filhos)
        pop = corrigir(pop)
        epoca += 1
    if encontrados_ids:
        return np.vstack(encontrados_ids), np.vstack(encontrados_kcal),np.vstack(hist_diversidade), np.vstack(hist_validos),hist_unicos
    return np.empty((0, 5), dtype=np.int32), np.empty((0,3)),np.vstack(hist_diversidade), np.vstack(hist_validos),hist_unicos
##########################
alvos = clientes[["carboidratos", "proteinas", "lipideos"]].values
ids_clientes = clientes["ID"].values

args = [
    (ids_clientes[i], alvos[i], 100000)
    for i in range(len(alvos))
]
##########################
def worker(args):
    cliente_id, alvo, total_avaliacoes= args

    ids, kcal,diversidade,taxa_validos,qnt_encontrados = genetico_busca(
        alvo=alvo,
        total_avaliacoes=total_avaliacoes,
        pop_size=100
    )
    ids = np.unique(ids, axis=0)
    return cliente_id, ids, kcal,diversidade,taxa_validos,qnt_encontrados
####################
if __name__ == "__main__":
    tempo=time.time() 
    with Pool() as p:
        resultados_lista = p.map(worker, args)    
    print("Tempo total:", time.time() - tempo)
####################
    for cliente_id, ids, kcal, diversidade, taxa_validos, qnt_encontrados in resultados_lista:
        resultados[int(cliente_id)] = {
            "ids": ids.tolist(),
            "kcal": kcal.tolist(),
            "qtd_solucoes": len(ids),
            "diversidade": diversidade.tolist(),
            "taxa_validos": taxa_validos.tolist(),
            "qnt_encontrados": qnt_encontrados
        }

    with open("clientes_genetico.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=4)
        print(f"Cliente {cliente_id}:")
        print("ids", ids)
        print("qtd_solucoes", len(ids))
        print("-" * 40)
        print(arroz_tabela.iloc[ids[0][0],2])
        print(feijao_tabela.iloc[ids[0][1],2])
        print(alimentos.iloc[ids[0][2],2])
        print(alimentos.iloc[ids[0][3],2])
        print(alimentos.iloc[ids[0][4],2])
        