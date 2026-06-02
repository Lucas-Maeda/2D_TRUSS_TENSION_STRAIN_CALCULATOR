#author:Lucas Maeda (https://www.linkedin.com/in/lucas-maeda-rodrigues-0780a821b/)
#Main Reference: Elementos Finitos: A base da tecnologia CAE - Avelino Alves Filho 

import numpy as np

# --- 0. FUNÇÕES DE UTILIDADE NUMÉRICA ---

def limpar_matriz(matriz, tolerancia=1e-10):
    """Substitui elementos muito próximos de zero por zero."""
    matriz_limpa = matriz.copy()
    matriz_limpa[np.abs(matriz_limpa) < tolerancia] = 0
    return matriz_limpa

# --- 1. FUNÇÕES DE SUPORTE PARA MATRIZ K E GDLs ---

def criar_K_elementares(n_barras):
    lista_id_node = []
    lista_K = []
    lista_propriedades = [] 

    for nbar in range(n_barras):
        print(f"\n--- Barra {nbar + 1} ---")
        try:
            L = float(input(f"Comprimento L da Barra {nbar + 1}: "))
            A = float(input(f"Área da Seção A da Barra {nbar + 1}: "))
            E = float(input(f"Módulo de Elasticidade E da Barra {nbar + 1}: "))
        except ValueError:
            print("Entrada inválida. Usando L=1.0, A=1.0, E=1.0.")
            L, A, E = 1.0, 1.0, 1.0

        id_no1 = int(input("Id do 1º nó: "))
        id_no2 = int(input("Id do 2º nó: "))
        ang = float(input("Ângulo da barra (em graus, em relação ao X global): "))

        rad_ang = ang * np.pi / 180
        lmbda = np.cos(rad_ang)
        muh = np.sin(rad_ang)

        lista_propriedades.append({'L': L, 'A': A, 'E': E, 'lmbda': lmbda, 'muh': muh})

        matrix_K_elementar = np.array([
            [lmbda**2, lmbda*muh, -lmbda**2, -lmbda*muh],
            [lmbda*muh, muh**2, -lmbda*muh, -muh**2],
            [-lmbda**2, -lmbda*muh, lmbda**2, lmbda*muh],
            [-lmbda*muh, -muh**2, lmbda*muh, muh**2]
        ])

        K = (A * E / L) * matrix_K_elementar
        K = limpar_matriz(K)

        lista_id_node.extend([id_no1, id_no2])
        lista_K.append(K)

    return lista_id_node, lista_K, lista_propriedades

def montar_matriz_K_global(lista_K, lista_id_node, no_map):
    N = len(no_map) * 2
    matriz_global = np.zeros((N, N))

    for i, K_barra in enumerate(lista_K):
        idx_base_no1 = no_map[lista_id_node[2 * i]] * 2
        idx_base_no2 = no_map[lista_id_node[2 * i + 1]] * 2
        indices_globais = [idx_base_no1, idx_base_no1 + 1, idx_base_no2, idx_base_no2 + 1]

        for linha in range(4):
            for coluna in range(4):
                matriz_global[indices_globais[linha], indices_globais[coluna]] += K_barra[linha, coluna]

    return limpar_matriz(matriz_global)

# --- 2. FUNÇÕES PARA MATRIZ DE FORÇAS F ---

def input_forcas_externas():
    forcas_aplicadas = []
    try:
        n_forcas_totais = int(input("\nNúmero total de forças externas a serem aplicadas nos nós: "))
    except ValueError: return []

    print("\n--- Entrada de Forças Externas ---")
    for i in range(n_forcas_totais):
        try:
            no_id = int(input(f"Força {i+1} - ID do Nó: "))
            magnitude = float(input("Magnitude: "))
            angulo = float(input("Ângulo (graus): "))
            forcas_aplicadas.append((no_id, magnitude, angulo))
        except ValueError: continue
    return forcas_aplicadas

def coletar_e_montar_matriz_F(forcas_aplicadas, nos_unicos):
    no_map = {no_id: i for i, no_id in enumerate(nos_unicos)}
    N = len(nos_unicos) * 2
    matriz_F = np.zeros((N, 1))

    for no_id, magnitude, angulo in forcas_aplicadas:
        if no_id in no_map:
            rad_angulo = angulo * np.pi / 180
            i = no_map[no_id]
            matriz_F[i * 2, 0] += magnitude * np.cos(rad_angulo)
            matriz_F[i * 2 + 1, 0] += magnitude * np.sin(rad_angulo)
    return matriz_F

# --- 3. FUNÇÕES DE RESTRIÇÃO E SOLUÇÃO ---

def cortar_matriz_K_Global(matriz_global, gdl_restritos):
    gdl_restritos = sorted(list(set(gdl_restritos)), reverse=True)
    matriz_reduzida = matriz_global.copy()
    for idx in gdl_restritos:
        matriz_reduzida = np.delete(matriz_reduzida, idx, axis=0)
        matriz_reduzida = np.delete(matriz_reduzida, idx, axis=1)
    return matriz_reduzida

def cortar_matriz_F_Global(matriz_f_global, gdl_restritos):
    gdl_restritos = sorted(list(set(gdl_restritos)), reverse=True)
    matriz_f_reduzida = matriz_f_global.copy()
    for idx in gdl_restritos:
        matriz_f_reduzida = np.delete(matriz_f_reduzida, idx, axis=0)
    return matriz_f_reduzida

def calcular_deformacoes_elementos(D_global, lista_id_node, lista_propriedades, no_map):
    deformacoes = []
    for i in range(len(lista_propriedades)):
        prop = lista_propriedades[i]
        idx_base_no1 = no_map[lista_id_node[2*i]] * 2
        idx_base_no2 = no_map[lista_id_node[2*i+1]] * 2
        indices = [idx_base_no1, idx_base_no1 + 1, idx_base_no2, idx_base_no2 + 1]
        d_e = D_global[indices].reshape((4, 1))
        T = np.array([[-prop['lmbda'], -prop['muh'], prop['lmbda'], prop['muh']]])
        delta_e = T @ d_e
        deformacoes.append(limpar_matriz(delta_e)[0, 0])
    return deformacoes

def calcular_tensoes_elementos(deformacoes_elementos, lista_propriedades):
    tensoes = []
    for i in range(len(deformacoes_elementos)):
        delta_e = deformacoes_elementos[i]
        prop = lista_propriedades[i]
        sigma_e = (prop['E'] / prop['L']) * delta_e if prop['L'] != 0 else 0.0
        tensoes.append(limpar_matriz(np.array([sigma_e]))[0])
    return tensoes

# --- 4. FUNÇÃO PRINCIPAL ---

def main():
    try:
        n_barras = int(input("Número de barras: "))
    except ValueError: return

    lista_id_node, lista_K, lista_propriedades = criar_K_elementares(n_barras)
    nos_unicos = sorted(list(set(lista_id_node)))
    no_map = {no_id: i for i, no_id in enumerate(nos_unicos)}
    N_total = len(nos_unicos) * 2

    forcas_aplicadas = input_forcas_externas()
    matriz_F_global = limpar_matriz(coletar_e_montar_matriz_F(forcas_aplicadas, nos_unicos))
    matriz_global = montar_matriz_K_global(lista_K, lista_id_node, no_map)

    # Definição de Restrições
    gdl_restritos = []
    try:
        n_engastes = int(input("\nNúmero de nós com Engaste (Restrição X e Y): "))
        for n in range(n_engastes):
            engaste = int(input(f"ID do nó engastado {n+1}: "))
            if engaste in no_map:
                gdl_restritos.extend([no_map[engaste] * 2, no_map[engaste] * 2 + 1])
        
        quant_outras = int(input("Número de GDLs adicionais restritos (Apoios simples): "))
        for n in range(quant_outras):
            no_restrito = int(input(f"ID do Nó da Restrição {n+1}: "))
            direcao = int(input("Direção (1 para X, 2 para Y): "))
            if no_restrito in no_map:
                gdl_restritos.append(no_map[no_restrito] * 2 + (direcao - 1))
    except ValueError: pass

    gdl_restritos_unicos = sorted(list(set(gdl_restritos)))
    matriz_K_reduzida = cortar_matriz_K_Global(matriz_global, gdl_restritos_unicos)
    matriz_f_reduzida = cortar_matriz_F_Global(matriz_F_global, gdl_restritos_unicos)

    # Solução
    if matriz_K_reduzida.size > 0 and np.linalg.det(matriz_K_reduzida) != 0:
        matriz_defor_reduzida = np.linalg.solve(matriz_K_reduzida, matriz_f_reduzida)
        matriz_defor_global = np.zeros((N_total, 1))
        gdl_livres_indices = [i for i in range(N_total) if i not in gdl_restritos_unicos]
        for idx_red, idx_global in enumerate(gdl_livres_indices):
            matriz_defor_global[idx_global] = matriz_defor_reduzida[idx_red]

        deformacoes_elementos = calcular_deformacoes_elementos(matriz_defor_global, lista_id_node, lista_propriedades, no_map)
        tensoes_elementos = calcular_tensoes_elementos(deformacoes_elementos, lista_propriedades)

        # --- CÁLCULO DAS REAÇÕES NOS ENGASTES/APOIOS ---
        # Reações = K_global * Deslocamento_global - Forças_Externas
        forcas_totais = matriz_global @ matriz_defor_global
        reacoes_apoio = limpar_matriz(forcas_totais - matriz_F_global)
    else:
        matriz_defor_reduzida = "Sistema Singular ou Sem GDLs Livres"
        matriz_defor_global = np.zeros((N_total, 1))
        deformacoes_elementos = tensoes_elementos = reacoes_apoio = "Não Calculado"

    # --- Resultados Mantidos e Incrementados ---
    print("\n" + "="*50)
    print("RESULTADOS DA ANÁLISE ESTRUTURAL")
    print("="*50)

    print("Matriz de Rigidez Global (K):")
    print(matriz_global)

    print("\nMatriz de Deslocamentos (D_reduzida):")
    print(matriz_defor_reduzida)

    print("\nMatriz de Deslocamentos Global Reconstruída (D):")
    print(matriz_defor_global)

    print("\n--- Reações de Apoio (Forças nos Engastes/Restrições) ---")
    if isinstance(reacoes_apoio, np.ndarray):
        for gdl in gdl_restritos_unicos:
            no_id = nos_unicos[gdl // 2]
            eixo = "X" if gdl % 2 == 0 else "Y"
            valor = reacoes_apoio[gdl, 0]
            print(f"Nó {no_id} - Reação no eixo {eixo}: {valor:.6e}")
    else:
        print(reacoes_apoio)

    print("\n--- Deformações e Tensões dos Elementos ---")
    if isinstance(deformacoes_elementos, list):
        for i in range(len(deformacoes_elementos)):
            print(f"Barra {i + 1} (Nós {lista_id_node[2*i]} -> {lista_id_node[2*i+1]}):")
            print(f"  Deformação Axial = {deformacoes_elementos[i]:.6e}")
            print(f"  Tensão Axial = {tensoes_elementos[i]:.6e}")
            print("-" * 25)

    print("="*50 + "\n")

if __name__ == "__main__":
    main()