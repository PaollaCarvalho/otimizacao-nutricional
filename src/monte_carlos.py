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

def gerar_marmitas_ultra(N, rng):
    idx_m = np.array([rng.choice(alim_vals.shape[0], size=3, replace=False) for _ in range(N)])
    idx_a = rng.integers(0, arroz_vals.shape[0], size=N)
    idx_f = rng.integers(0, feijao_vals.shape[0], size=N)

    total = (
        alim_vals[idx_m[:, 0]] +
        alim_vals[idx_m[:, 1]] +
        alim_vals[idx_m[:, 2]] +
        arroz_vals[idx_a] +
        feijao_vals[idx_f]
    )

    kcal = total.copy()
    kcal[:, 0] *= 4
    kcal[:, 1] *= 4
    kcal[:, 2] *= 9

    kcal /= kcal.sum(axis=1, keepdims=True)

    ids = np.empty((N, 5), dtype=np.int32)
    ids[:, 0] = arroz_ids[idx_a]
    ids[:, 1] = feijao_ids[idx_f]
    ids[:, 2:] = alim_ids[idx_m]

    return kcal, ids
#################
def filtrar(kcal, alvo, tol=0.1):
    alvo = np.where(alvo == 0, 1e-8, alvo)
    return np.all(np.abs(kcal - alvo) / alvo <= tol, axis=1)
#################
def monte_carlo_cliente(alvo, total_avaliacoes, batch=100, seed=42):
    rng = np.random.default_rng(seed)

    encontrados_ids = []
    encontrados_kcal = []
    hist_diversidade = []
    encontrados_set = set() 
    hist_unicos = []
    hist_validos = []
    avaliacoes = 0

    while avaliacoes < total_avaliacoes:
        kcal, ids = gerar_marmitas_ultra(batch, rng)

        mask = filtrar(kcal, alvo)

        diversidade=len(np.unique(ids, axis=0)) / len(ids)
        hist_diversidade.append(diversidade)
        hist_validos.append(np.mean(mask))
        if np.any(mask):
            encontrados_ids.append(ids[mask])
            encontrados_kcal.append(kcal[mask])
            validos_ids = ids[mask]
            for ind in validos_ids:
                encontrados_set.add(tuple(sorted(ind)))
            hist_unicos.append(len(encontrados_set))
        else:
            hist_unicos.append(len(encontrados_set))
        avaliacoes += batch
    if encontrados_ids:
        return np.vstack(encontrados_ids), np.vstack(encontrados_kcal),np.vstack(hist_diversidade), np.vstack(hist_validos),hist_unicos
    return np.empty((0, 5), dtype=np.int32), np.empty((0,3)),np.vstack(hist_diversidade), np.vstack(hist_validos),hist_unicos
###################
alvos = clientes[["carboidratos", "proteinas", "lipideos"]].values
ids_clientes = clientes["ID"].values


args = [
    (ids_clientes[i], alvos[i], 100000, ids_clientes[i])
    for i in range(len(alvos))
]
###################
def worker(args):
    cliente_id, alvo, total_avaliacoes, seed = args

    ids, kcal,diversidade,taxa_validos,qnt_encontrados = monte_carlo_cliente(
        alvo=alvo,
        total_avaliacoes=total_avaliacoes,
        seed=seed
    )
    ids = np.sort(ids, axis=1)
    ids = np.unique(ids, axis=0)
    return cliente_id, ids, kcal,diversidade,taxa_validos,qnt_encontrados
####################
if __name__ == "__main__":
    tempo = time.time()
    with Pool() as p:
        resultados_lista = p.map(worker, args)  
    print("Tempo total:", time.time() - tempo)  
####################
    for cliente_id, ids, kcal,diversidade,taxa_validos, qnt_encontrados in resultados_lista:
        resultados[int(cliente_id)] = {
            "ids": ids.tolist(),
            "kcal": kcal.tolist(),
            "qtd_solucoes": len(ids),
            "diversidade": diversidade.tolist(),
            "taxa_validos": taxa_validos.tolist(),
            "qnt_encontrados": qnt_encontrados
        }
        # print(f"Cliente {cliente_id}:")
        # print("ids", ids)
        # print("qtd_solucoes", len(ids))
        # print("-" * 40)
        # print("qnt_encontrados:",qnt_encontrados)
    with open("clientes_mc.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=4)