"""
fix_stack_length.py
-------------------
Correção pontual (2026-10-21) do valid_designs.csv de um raw gerado antes do stack length
fixo: substitui SÓ a coluna `stack_length [mm]` por motor_constants.STACK_LENGTH_MM. Fora do
pipeline de geração.

Contexto: até 2026-10-21 o sampler 'legacy' sorteava o stack length em U[5, 10] mm; agora ele
é fixo (STACK_LENGTH_MM), com o sorteio mantido e o valor descartado
(BLDC_Process.generate_samples) -- as geometrias do raw continuam as mesmas. Os `.ans` não
dependem do stack length (problema planar; [Depth] só escala integrais que o pipeline não lê
do `.ans`), então só o CSV precisa ser corrigido.

Conferências antes de gravar (aborta se alguma falhar):
  1. demais colunas idênticas às do original (mesmos textos, mesma ordem de linhas/colunas);
  2. CSV resultante == o que generate_samples + add_phase_currents + export_params produzem
     com datagen_seed / n_samples / current_peak do DatagenConfig (célula a célula, texto);
  3. nº de linhas == nº de sample_XXXXXX.ans.gz na pasta.
Backup do original em valid_designs.csv.bak (nunca sobrescreve um .bak existente); escrita
atômica (temporário + rename). Idempotente: num CSV já corrigido, não grava nada.

Execução (a partir da raiz do projeto):
    python -m scripts.fix_stack_length [--raw-dir data/raw/<dataset>] [--dry-run]
"""
import argparse
import csv
import io
import os
import re
import sys
import tempfile
from pathlib import Path

from src.configs.datagen import DatagenConfig
from src.data_gen.motor_constants import STACK_LENGTH_MM
from src.data_gen.motor_model import BLDC_Process

COL = 'stack_length [mm]'
CSV_NAME = 'valid_designs.csv'
_SAMPLE_RE = re.compile(r'^sample_\d{6}\.ans\.gz$')


def _read(path: Path):
    """(linhas de texto [header + dados], terminador de linha) -- células como texto cru."""
    raw = path.read_bytes().decode('utf-8')
    term = '\r\n' if '\r\n' in raw else '\n'
    return list(csv.reader(io.StringIO(raw, newline=''))), term


def _expected_rows(dg: DatagenConfig):
    """CSV que o gerador atual escreveria (export_params num diretório temporário)."""
    params = BLDC_Process.generate_samples(num_samples=dg.n_samples, seed=dg.datagen_seed)
    if dg.current_peak > 0:
        BLDC_Process.add_phase_currents(params, i_peak=dg.current_peak, seed=dg.datagen_seed)
    with tempfile.TemporaryDirectory() as tmp:
        BLDC_Process.export_params(params, filename=CSV_NAME, data_path=tmp)
        return _read(Path(tmp) / CSV_NAME)[0]


def _fail(msg):
    sys.exit(f"ABORTADO (nada gravado): {msg}")


def main():
    dg = DatagenConfig()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    ap.add_argument('--raw-dir', default=f'data/raw/{dg.dataset}')
    ap.add_argument('--dry-run', action='store_true', help='só confere, não grava')
    a = ap.parse_args()

    raw_dir = Path(a.raw_dir)
    csv_path = raw_dir / CSV_NAME
    if not csv_path.exists():
        _fail(f"{csv_path} não existe")
    orig, term = _read(csv_path)
    header, rows = orig[0], orig[1:]
    if COL not in header:
        _fail(f"coluna {COL!r} ausente em {csv_path}")
    j = header.index(COL)

    expected = _expected_rows(dg)
    if expected[0] != header:
        _fail(f"colunas do CSV {header} != as do gerador {expected[0]}")
    new_val = {r[j] for r in expected[1:]}
    if len(new_val) != 1 or float(next(iter(new_val))) != STACK_LENGTH_MM:
        _fail(f"gerador não produz stack_length constante = {STACK_LENGTH_MM}: {sorted(new_val)[:5]}")
    new_val = next(iter(new_val))          # texto exatamente como export_params grava

    fixed = [header] + [r[:j] + [new_val] + r[j + 1:] for r in rows]

    # 1. demais colunas idênticas ao original
    bad = [i for i, (r, f) in enumerate(zip(rows, fixed[1:]))
           if len(r) != len(header) or r[:j] + r[j + 1:] != f[:j] + f[j + 1:]]
    if bad:
        _fail(f"conferência 1: {len(bad)} linha(s) com outras colunas alteradas (ex.: {bad[:5]})")
    # 2. igual ao gerador atual (célula a célula, como texto)
    if len(fixed) != len(expected):
        _fail(f"conferência 2: {len(fixed) - 1} linhas no CSV x {len(expected) - 1} do gerador "
              f"(n_samples={dg.n_samples})")
    diff = [(i, header[c]) for i, (r, e) in enumerate(zip(fixed[1:], expected[1:]))
            for c in range(len(header)) if r[c] != e[c]]
    if diff:
        _fail(f"conferência 2: {len(diff)} célula(s) diferem do gerador (ex.: {diff[:5]}) — "
              f"o raw não foi gerado com seed={dg.datagen_seed}/n={dg.n_samples}/"
              f"I_pico={dg.current_peak}, ou o formato numérico mudou")
    # 3. nº de linhas == nº de amostras
    n_ans = sum(1 for p in raw_dir.iterdir() if _SAMPLE_RE.match(p.name))
    if n_ans != len(rows):
        _fail(f"conferência 3: {len(rows)} linhas no CSV x {n_ans} sample_*.ans.gz em {raw_dir}")

    print(f"conferências 1-3 ok ({len(rows)} linhas, {len(header)} colunas, {n_ans} .ans.gz)")
    if fixed == orig:
        print(f"{csv_path}: {COL} já é {new_val} em todas as linhas — nada a fazer")
        return
    old_vals = sorted({float(r[j]) for r in rows})
    print(f"{COL}: {len(old_vals)} valor(es) distintos em [{old_vals[0]}, {old_vals[-1]}] -> {new_val}")
    if a.dry_run:
        print("--dry-run: nada gravado")
        return

    bak = csv_path.with_name(CSV_NAME + '.bak')
    if not bak.exists():
        bak.write_bytes(csv_path.read_bytes())
        print(f"backup: {bak}")
    else:
        print(f"backup já existe, mantido: {bak}")
    buf = io.StringIO(newline='')
    csv.writer(buf, lineterminator=term).writerows(fixed)
    tmp = csv_path.with_name(CSV_NAME + '.tmp')
    tmp.write_bytes(buf.getvalue().encode('utf-8'))
    os.replace(tmp, csv_path)
    print(f"gravado: {csv_path}")


if __name__ == '__main__':
    main()
