"""
src/postprocess/
----------------
Pós-processamento oficial (só gabarito SMOOTH): métricas, inferência a partir de
runs salvas, amostras do raw e plots. Nada aqui treina.

Executável: scripts/postprocess.py (sem config dedicada).
Saída: resultados/<dataset>/{dados,plots}/ — só dados, sem código; data/logs/
fica apenas com o que vem direto do treino.
Só se migra para cá código efetivamente usado; o resto é descartado na limpeza.

Regra de dependência: pode importar de src.configs, src.neural_op e src.data_gen;
nenhum desses importa de src.postprocess (evita ciclos).

Módulos:
  metrics    — erro de |B| (Accum, pesos de área, piso da grade)
  fields     — operações sobre o campo (|B|, raio do entreferro)
  inference  — reconstrução da run a partir do log + predição
  samples    — amostras do raw (designs, subsets, parse smooth)
  train_table — tabela de treinamento a partir dos logs das runs (P3)
  plots/     — figuras (primitivas, treino, erro, campo)
"""
