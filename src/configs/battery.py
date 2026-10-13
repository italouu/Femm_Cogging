"""
battery.py — definição de uma bateria de treinos (2026-10-13).

Uma bateria é uma sequência de execuções; cada execução é uma função que devolve um
`NnCfg` escrito POR INTEIRO no arquivo da bateria (src/configs/batteries/<nome>.py).
Nada é herdado de outra run nem de preset: o que está escrito na função é o que treina.

A função (e não um `NnCfg` já instanciado) é necessária porque `NnCfg.__post_init__`
lê o dataset (dimensões, normalizador) e `GNN_PostBaseConfig.__post_init__` lê o
config.json da base — a base só existe depois que a run anterior da bateria termina.
A função só é chamada imediatamente antes do treino, com instâncias novas.

Dependência entre runs (ex: GNN_PostBase precisa dos pesos de um FNO2d da mesma
bateria) é declarada em `needs`: a função recebe `runs = {nome: run_dir}` contendo
SÓ as runs declaradas em `needs`, já concluídas na mesma repetição.

Execução: python -m scripts.run_battery <nome>   (ver scripts/run_battery.py)
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BatteryRun:
    name:  str        # único na bateria; chave do manifesto e de `needs`
    build: Callable   # (runs: dict[str, str]) -> NnCfg   (NnCfg.repeat fica no default 0 —
                      #  é o runner que grava a repetição)
    needs: tuple = () # nomes de runs ANTERIORES desta bateria das quais esta depende


@dataclass
class BatteryCfg:
    name:      str                           # data/logs/_batteries/<name>/manifest.json
    runs:      list = field(default_factory=list)
    n_repeats: int  = 1                      # a sequência inteira é repetida n vezes
                                             # (sem controle de semente); NnCfg.repeat = índice

    def __post_init__(self):
        seen = set()
        for r in self.runs:
            if r.name in seen:
                raise ValueError(f"bateria '{self.name}': nome de run repetido '{r.name}'")
            for dep in r.needs:
                if dep not in seen:
                    raise ValueError(
                        f"bateria '{self.name}': run '{r.name}' depende de '{dep}', que não "
                        f"existe ou não vem ANTES dela na sequência")
            seen.add(r.name)
        if self.n_repeats < 1:
            raise ValueError(f"bateria '{self.name}': n_repeats={self.n_repeats} < 1")
