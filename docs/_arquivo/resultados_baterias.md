# Resultados — bateria oficial × bateria smooth + Elem (2026-10-10)

## O que mudou

**Gabarito smooth.** Mesmo raw (`data/raw/mesh_ans_138x276/`, 4000 amostras), outro gabarito de B:
- **Antiga (unificado):** B = curl(A) exato por elemento; no nó, média simples dos elementos
  incidentes; na grade, interpolação baricêntrica desse B nodal.
- **Smooth:** B suavizado do pós-processador do FEMM, portado do xfemm e lido só do `.ans`
  (bate com o FEMM a ~1e-15). É linear por elemento e descontínuo entre elementos. No nó, o
  valor vem do lado do material de maior prioridade (ferro > ímã > cobre > ar). Na grade, é o
  valor suavizado no centro do pixel. Nas interfaces ele salta mais e é mais fiel ao FEMM.

**FNO_BipartiteGNN_Elem.** É a Bipartite com os papéis dos grafos trocados:
- os **elementos** formam o grafo principal, iterado pela GNN, com saída Bx,By por elemento;
- os **vértices** formam o grafo auxiliar, estático e só com posição.

A saída por elemento evita o problema do nó de interface, que precisa escolher um lado. Na
versão smooth, o alvo é o B suavizado no centróide do elemento. Para comparar nos nós, cada
nó recebe a média dos elementos incidentes do material de maior prioridade, a mesma regra do
gabarito.

Fora isso, as duas baterias usam a mesma configuração: hiperparâmetros, chunking 125×32 e
`split_seed=12`, com o mesmo treino de 37 chunks.

## Erro de |B| nos nós

Mesmas 64 amostras de teste. Cada bateria é medida contra o próprio gabarito. Cada célula traz
L1 por superfície / L2 por superfície / L1 ponto a ponto, em % do B_ref: 0,609 T na antiga e
0,654 T na smooth.

| Arch | Loss | Antiga | Smooth |
|---|---|---|---|
| FNO2d | mae | 4,75 / 9,30 / 22,70 | 9,28 / 28,72 / 25,84 |
| FNO_GNN | mae | 5,39 / 8,50 / 10,80 | 8,28 / 12,45 / 13,83 |
| GNN_PostBase | mae | 4,26 / 7,93 / 15,56 | 7,48 / 24,85 / 13,51 |
| FNO_BipartiteGNN | mse | 6,96 / 9,41 / 9,01 | 7,16 / 9,93 / 9,66 |
| FNO_BipartiteGNN | mae | **3,77 / 5,61 / 7,51** | 4,27 / 6,55 / 7,81 |
| FNO_BipartiteGNN_Elem | mse | — | 6,44 / 9,13 / 8,94 |
| FNO_BipartiteGNN_Elem | mae | — | **3,41 / 5,52 / 6,56** |
| *piso: grade interpolada nos nós* | — | 2,09 / 6,62 / 19,85 | 7,52 / 28,44 / 23,95 |

## Leitura

- **Melhor modelo:**
  - Antiga: Bipartite mae. No teste completo de 2816 amostras ela dá 3,80 / 5,68 / 7,64.
  - Smooth: Elem mae, que vence a Bipartite smooth nas três métricas.
- **Quem interpola a grade nos nós sofre com o gabarito smooth.** O piso de L2 sobe de 6,6%
  para 28,4%, e o FNO2d e o GNN_PostBase ficam presos a ele.
- **Os modelos que preveem no grafo quase não sentem a troca:** FNO_GNN, Bipartite e Elem.
- **Comparar entre as baterias é só indicativo:** os gabaritos e os B_ref são diferentes. Um
  modelo antigo medido contra o gabarito smooth sobe para L2 ~21%.

## Ressalvas

- Vários runs da smooth pararam cedo pelo critério GL, como a Elem mse com best na época 49.
- O L2 ponto a ponto ainda não é calculado.
- O erro na grade contra a malha ainda não usa interseção célula ∩ triângulo. Fica pendente
  (ver CLAUDE.md, "Convenção de erro").

**Fontes:** `pos_bateria/eval_smooth.json` (bateria smooth), `eval_elem_smooth.json`
(`mesh_ans_138x276_smooth_elem_proto`) e `RELATORIO.md` (bateria oficial).
