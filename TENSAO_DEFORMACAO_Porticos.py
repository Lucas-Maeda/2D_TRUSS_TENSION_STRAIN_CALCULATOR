import numpy as np

# --- 0. FUNÇÕES DE UTILIDADE NUMÉRICA ---

def limpar_matriz(matriz, tolerancia=1e-10):
    matriz_limpa = matriz.copy()
    matriz_limpa[np.abs(matriz_limpa) < tolerancia] = 0
    return matriz_limpa

# --- 1. FUNÇÕES DE SUPORTE PARA MATRIZ K ---

def criar_K_elementares_portico(n_barras):
    lista_id_node = []
    lista_K_global = []
    lista_K_local = [] # Adicionado para o item (c,d,e)
    lista_propriedades = [] 

    for nbar in range(n_barras):
        print(f"\n--- Elemento {nbar + 1} ---")
        L = float(input(f"Comprimento L: "))
        A = float(input(f"Área A: "))
        E = float(input(f"Módulo de Elasticidade E: "))
        I = float(input(f"Inércia da seção I: "))
        
        id_no1 = int(input("Id do 1º nó: "))
        id_no2 = int(input("Id do 2º nó: "))
        ang = float(input("Ângulo (graus, ref. X global): "))

        rad_ang = np.radians(ang)
        c = np.cos(rad_ang)
        s = np.sin(rad_ang)

        k_axial = (E * A / L)
        k_flex1 = (12 * E * I / L**3)
        k_flex2 = (6 * E * I / L**2)
        k_flex3 = (4 * E * I / L)
        k_flex4 = (2 * E * I / L)

        kl = np.array([
            [k_axial,  0,       0,      -k_axial, 0,       0],
            [0,        k_flex1, k_flex2, 0,      -k_flex1, k_flex2],
            [0,        k_flex2, k_flex3, 0,      -k_flex2, k_flex4],
            [-k_axial, 0,       0,       k_axial, 0,       0],
            [0,       -k_flex1,-k_flex2, 0,       k_flex1, -k_flex2],
            [0,        k_flex2, k_flex4, 0,      -k_flex2, k_flex3]
        ])

        T = np.array([
            [c,  s,  0,  0,  0,  0],
            [-s, c,  0,  0,  0,  0],
            [0,  0,  1,  0,  0,  0],
            [0,  0,  0,  c,  s,  0],
            [0,  0,  0, -s,  c,  0],
            [0,  0,  0,  0,  0,  1]
        ])

        K_g_elem = T.T @ kl @ T
        
        # ITEM (a): Demonstração das matrizes elementares
        print(f"> Matriz de Rigidez Local do Elemento {nbar+1} (K_local):\n{limpar_matriz(kl)}")
        
        lista_propriedades.append({'L': L, 'A': A, 'E': E, 'I': I, 'cos': c, 'sin': s, 'T': T, 'nodes': [id_no1, id_no2]})
        lista_K_global.append(limpar_matriz(K_g_elem))
        lista_K_local.append(kl)
        lista_id_node.extend([id_no1, id_no2])

    return lista_id_node, lista_K_global, lista_K_local, lista_propriedades

def montar_matriz_K_global(lista_K, lista_id_node, no_map):
    N = len(no_map) * 3
    matriz_global = np.zeros((N, N))
    for i, K_barra in enumerate(lista_K):
        n1 = no_map[lista_id_node[2 * i]]
        n2 = no_map[lista_id_node[2 * i + 1]]
        idx = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]
        for r in range(6):
            for c in range(6):
                matriz_global[idx[r], idx[c]] += K_barra[r, c]
    return limpar_matriz(matriz_global)

def coletar_e_montar_matriz_F(nos_unicos):
    no_map = {no_id: i for i, no_id in enumerate(nos_unicos)}
    N = len(nos_unicos) * 3
    matriz_F = np.zeros((N, 1))
    n_forcas = int(input("\nNúmero de nós com cargas aplicadas: "))
    for _ in range(n_forcas):
        no_id = int(input("ID do Nó: "))
        if no_id in no_map:
            idx = no_map[no_id] * 3
            fx = float(input("Força em X: "))
            fy = float(input("Força em Y: "))
            mz = float(input("Momento (Z): "))
            matriz_F[idx, 0] += fx
            matriz_F[idx+1, 0] += fy
            matriz_F[idx+2, 0] += mz
    return matriz_F

# --- 3. SOLUÇÃO E CÁLCULO DE ESFORÇOS ---

def main():
    print("--- ANÁLISE DE PÓRTICOS PLANOS (MEF) ---")
    n_barras = int(input("Número de elementos: "))
    
    lista_id_node, lista_K_g, lista_K_l, lista_prop = criar_K_elementares_portico(n_barras)
    nos_unicos = sorted(list(set(lista_id_node)))
    no_map = {no_id: i for i, no_id in enumerate(nos_unicos)}
    N_total = len(nos_unicos) * 3

    matriz_F_global = coletar_e_montar_matriz_F(nos_unicos)
    matriz_global = montar_matriz_K_global(lista_K_g, lista_id_node, no_map)

    gdl_restritos = []
    n_apoios = int(input("\nNúmero de nós com restrição: "))
    for _ in range(n_apoios):
        no_id = int(input("ID do nó: "))
        if input("Restringir X? (1=S, 0=N) ") == '1': gdl_restritos.append(no_map[no_id]*3)
        if input("Restringir Y? (1=S, 0=N) ") == '1': gdl_restritos.append(no_map[no_id]*3 + 1)
        if input("Restringir Rotação? (1=S, 0=N) ") == '1': gdl_restritos.append(no_map[no_id]*3 + 2)

    gdl_livres = [i for i in range(N_total) if i not in gdl_restritos]
    K_red = matriz_global[np.ix_(gdl_livres, gdl_livres)]
    F_red = matriz_F_global[gdl_livres]
    desloc_red = np.linalg.solve(K_red, F_red)
    
    D_global = np.zeros((N_total, 1))
    for i, idx in enumerate(gdl_livres):
        D_global[idx] = desloc_red[i]

    print("\n" + "="*50)
    print("ITEM (B): SOLUÇÃO DO SISTEMA {U}")
    print("="*50)
    for no_id in nos_unicos:
        idx = no_map[no_id] * 3
        print(f"Nó {no_id}: U={D_global[idx,0]:.4e}, V={D_global[idx+1,0]:.4e}, Rot={D_global[idx+2,0]:.4e}")

    print("\n" + "="*50)
    print("ITENS (C, D, E): ESFORÇOS INTERNOS NAS BARRAS")
    print("="*50)
    
    for i, p in enumerate(lista_prop):
        n1, n2 = p['nodes']
        idx1, idx2 = no_map[n1]*3, no_map[n2]*3
        
        # Deslocamentos globais do elemento
        u_g = np.array([
            D_global[idx1,0], D_global[idx1+1,0], D_global[idx1+2,0],
            D_global[idx2,0], D_global[idx2+1,0], D_global[idx2+2,0]
        ])
        
        # Transformação para o sistema local: u_local = T * u_global
        u_l = p['T'] @ u_g
        
        # Esforços locais: f_local = k_local * u_local
        f_l = lista_K_l[i] @ u_l
        f_l = limpar_matriz(f_l)

        print(f"\nELEMENTO {i+1} (Nós {n1}-{n2}):")
        print(f"  Normal (N) [Nó Inicial]: {f_l[0]:.2f} N")
        print(f"  Cortante (V) [Nó Inicial]: {f_l[1]:.2f} N")
        print(f"  Momento (M) [Nó Inicial]: {f_l[2]:.2f} N.m")
        print(f"  Normal (N) [Nó Final]: {f_l[3]:.2f} N")
        print(f"  Cortante (V) [Nó Final]: {f_l[4]:.2f} N")
        print(f"  Momento (M) [Nó Final]: {f_l[5]:.2f} N.m")

if __name__ == "__main__":
    main()