# Fase 0 — resultados (1 chunks de teste)

B_ref (malha): dual = 0.5932 T, elem = 0.5878 T  | nós = 1026683, elementos = 2031046

## T0a — piso de representação (y_hw → nós vs node_y)

| caso | L1 dual (%B_ref) | L2 dual (%B_ref) | L1 elem (%B_ref) | L2 elem (%B_ref) |
|---|---|---|---|---|
| legacy | 4.001 | 10.129 | 4.037 | 9.473 |
| cell_centered | 2.150 | 7.038 | 2.170 | 6.426 |

## T0b — FNO2d nos nós

| caso | L1 dual (%B_ref) | L2 dual (%B_ref) | L1 elem (%B_ref) | L2 elem (%B_ref) |
|---|---|---|---|---|
| FNO2d mse/legacy | 7.326 | 12.865 | 7.393 | 12.148 |
| FNO2d mse/cell_centered | 6.332 | 10.730 | 6.391 | 10.071 |
| FNO2d mae/legacy | 5.814 | 11.985 | 5.867 | 11.230 |
| FNO2d mae/cell_centered | 4.756 | 9.656 | 4.800 | 8.968 |

## T0c — estágio FNO na grade (ε_grid, peso r·dr·dθ)

| arch / loss | decod. | B_ref (T) | L1 (%B_ref) | L2 (%B_ref) |
|---|---|---|---|---|
| FNO_GNN / mse | y_hw | 0.5856 | 41.487 | 58.278 |
| FNO_GNN / mse | node_y | 0.5856 | 54.801 | 80.204 |
| FNO_GNN / mae | y_hw | 0.5856 | 29.737 | 44.955 |
| FNO_GNN / mae | node_y | 0.5856 | 30.251 | 57.367 |
| FNO_BipartiteGNN / mse | y_hw | 0.5856 | 60.388 | 82.466 |
| FNO_BipartiteGNN / mse | node_y | 0.5856 | 56.988 | 77.990 |
| FNO_BipartiteGNN / mae | y_hw | 0.5856 | 58.995 | 81.112 |
| FNO_BipartiteGNN / mae | node_y | 0.5856 | 55.171 | 76.339 |

### T0c (extra) — mesmo estágio FNO interpolado nos nós (legacy)

| caso | L1 dual (%B_ref) | L2 dual (%B_ref) | L1 elem (%B_ref) | L2 elem (%B_ref) |
|---|---|---|---|---|
| FNO_GNN/mse/y_hw | 40.857 | 56.441 | 41.232 | 55.937 |
| FNO_GNN/mse/node_y | 53.242 | 77.022 | 53.730 | 76.559 |
| FNO_GNN/mae/y_hw | 28.254 | 42.545 | 28.513 | 42.023 |
| FNO_GNN/mae/node_y | 26.759 | 53.110 | 27.005 | 52.553 |
| FNO_BipartiteGNN/mse/y_hw | 60.948 | 83.052 | 61.507 | 82.930 |
| FNO_BipartiteGNN/mse/node_y | 57.686 | 78.688 | 58.215 | 78.513 |
| FNO_BipartiteGNN/mae/y_hw | 59.485 | 81.592 | 60.030 | 81.465 |
| FNO_BipartiteGNN/mae/node_y | 55.765 | 76.890 | 56.276 | 76.708 |

## Checagens

- máx |node_dual_area − Σ A_e/3| / máx(dual): 7.07e-08
- máx |r_base,c_base (v1) − (bipartite)|: 0.00e+00
- máx |node_y (v1) − node_y (bipartite)|: 0.00e+00
