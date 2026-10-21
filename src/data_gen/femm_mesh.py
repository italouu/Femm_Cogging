"""
femm_mesh.py
------------
Geração do raw: desenha o setor de 120° (BLDC_FEMM_Model_Sym120_Annular), malha, resolve
(mi_analyze) e salva o `.ans` inteiro comprimido em gzip -- ver save_ans_gzip_sample.
Chamado por scripts/generate_data_femm_mesh_v2.py (Windows, pyfemm).
"""

import gzip
import shutil
from pathlib import Path

import femm

from src.data_gen.motor_constants import stack_length_mm
from src.data_gen.motor_model import BLDC_FEMM_Model_Sym120_Annular


def save_ans_gzip_sample(motor_params: dict, tmp_dir: Path, out_path: Path) -> dict:
    """Gera 1 amostra (desenha + malha + mi_analyze) e salva o `.ans`
    BRUTO -- texto completo escrito pelo FEMM, sem nenhuma extração ou
    derivação -- comprimido em gzip direto em `out_path`
    (`data/raw/<dataset>/sample_XXXXXX.ans.gz`, escrita atômica).

    Guarda o `.ans` inteiro (malha, conectividade e solução) em vez de
    derivar arrays na hora: qualquer derivação futura (ex: B suavizado,
    B exato por elemento) parte só do arquivo, sem FEMM e sem resimular.

    SEM `mi_loadsolution()`/chamada de pós-processamento nenhuma -- não
    precisa, o arquivo que `mi_analyze()` já escreveu em disco é
    exatamente o que se quer guardar.

    Compressão: gzip padrão da stdlib, cópia em bytes crus (`'rb'`/`'wb'`,
    sem tradução de encoding/newline) -- formato portável, decompressível
    em qualquer SO. Quem for parsear o conteúdo depois (Linux, ao que tudo
    indica) deve abrir com `encoding='utf-8'` explícito, nunca depender do
    encoding padrão do SO (nesta sessão, prints no console Windows já
    quebraram por causa disso -- cp1252 não é UTF-8).
    """
    tmp_dir = Path(tmp_dir)
    out_path = Path(out_path)
    fem_file = str(tmp_dir / "model.fem")
    ans_file = tmp_dir / "model.ans"

    phase = float(motor_params['rotor_phase']['value']) if 'rotor_phase' in motor_params else 0.0
    model = BLDC_FEMM_Model_Sym120_Annular(motor_params=motor_params, phase=phase)

    femm.openfemm('bHide')
    femm.main_resize(1000, 1000)
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, stack_length_mm(motor_params), 200)
    model.draw_motor()
    femm.mi_zoomnatural()
    femm.mi_saveas(fem_file)
    femm.mi_createmesh()
    femm.mi_analyze()
    femm.closefemm()

    # escrita atômica: salva em .tmp e renomeia
    stem = out_path.stem.removesuffix('.ans')   # sample_XXXXXX (out_path = sample_XXXXXX.ans.gz)
    tmp_gz = out_path.parent / f"{stem}.tmp.ans.gz"
    with open(ans_file, 'rb') as f_in, gzip.open(tmp_gz, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    tmp_gz.replace(out_path)

    return {
        'ans_size_bytes': ans_file.stat().st_size,
        'gz_size_bytes': out_path.stat().st_size,
    }
