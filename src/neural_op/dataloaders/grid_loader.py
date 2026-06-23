import random
import sys
import threading
import queue as _queue
import torch
from torch.utils.data import IterableDataset, DataLoader


# [REMOVIDO] GridChunkDataset — carregava todos os chunks em RAM simultaneamente
#            via torch.cat no __init__, inviável para datasets grandes.
#            Substituída por ChunkStreamDataset (IterableDataset com buffer manual).
#
# class GridChunkDataset(Dataset):
#     """
#     Carrega todos os quad_chunk_*.pt em memória no __init__ e os concatena
#     num único tensor contíguo [N, C, H, W].
#     Cada item retorna (x[i], y[i]) com shape [C, H, W] via indexação direta.
#
#     Estimativa de RAM: N_amostras × C × H × W × 4 bytes × 2 (x e y).
#     Para 1500 amostras [2,80,240]: ~460 MB.
#     """
#     def __init__(self, chunk_paths):
#         xs, ys = [], []
#         n_chunks = len(chunk_paths)
#         for ci, p in enumerate(chunk_paths):
#             print(f"\r  carregando chunks  [{ci+1}/{n_chunks}]", end='', flush=True)
#             d = torch.load(p, map_location='cpu')
#             xs.append(d['x_hw'])   # [B, C, H, W]
#             ys.append(d['y_hw'])   # [B, C, H, W]
#         print()
#
#         self.x = torch.cat(xs, dim=0)   # [N, C, H, W]
#         self.y = torch.cat(ys, dim=0)   # [N, C, H, W]
#
#     def __len__(self):
#         return self.x.shape[0]
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]


# ── Streaming dataset ─────────────────────────────────────────────────────────

class ChunkStreamDataset(IterableDataset):
    """
    Lê quad_chunk_*.pt do disco um a um — nunca mais de 1 chunk em memória
    por vez — e emite amostras individuais (x_hw, y_hw) através de um buffer
    de shuffle manual.

    Fluxo por iteração:
      1. Embaralha a ordem dos chunks (diferente a cada época).
      2. Para cada chunk:
           a. torch.load → extrai x_hw e y_hw.
           b. del dict original → grafos (edge_index, node_x, …) liberados
              imediatamente.
           c. Itera sobre amostras em ordem aleatória; cada amostra é clonada
              antes de entrar no buffer — storage independente do chunk,
              permitindo que del x_hw/y_hw libere o storage ao final.
           d. A cada inserção: se buffer cheio, sorteia 1 posição, emite a
              amostra nessa posição e substitui pela última (O(1)).
           e. del x_hw, y_hw → storage do chunk liberado.
      3. Flush: embaralha e emite o buffer restante.

    Com num_workers > 0: cada worker recebe uma fatia não-sobreponente dos
    chunks (stride = num_workers), com semente RNG independente.

    Memória máxima em RAM por worker:
      • Durante torch.load: tamanho do arquivo de chunk (temporário)
      • Persistente: buffer_size × C × H × W × 4B × 2  (x e y)
        ex: 1024 × [2,80,240] float32 ≈ 300 MB
    """

    def __init__(self, chunk_paths, buffer_size, prefetch_chunks=2):
        self.chunk_paths     = list(chunk_paths)
        self.buffer_size     = buffer_size
        self.prefetch_chunks = prefetch_chunks

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.chunk_paths)

        # Instância local de RNG — não altera o estado global de random
        if worker_info is not None:
            rng   = random.Random(worker_info.seed % (2 ** 32))
            paths = paths[worker_info.id :: worker_info.num_workers]
        else:
            rng = random.Random()   # seed automático — diferente a cada época

        rng.shuffle(paths)

        # ── Thread de background: pré-carrega chunks sem bloquear o pipeline ─
        # torch.load libera o GIL durante a leitura de disco e a cópia de
        # storage; o thread principal continua entregando amostras ao DataLoader
        # enquanto o próximo chunk é desserializado.
        #
        # Sem .clone() por amostra: x_hw[i] é uma view — o storage do chunk
        # permanece vivo enquanto o buffer mantiver alguma view dele (refcount
        # Python), sendo liberado automaticamente quando a última view sai.
        #
        # stop_event: sinaliza ao thread para encerrar quando o gerador for
        # abandonado pelo DataLoader (fim de época com persistent_workers).
        # Sem isso, o thread fica orphan fazendo q.put() bloqueante, segurando
        # tensores na fila — cada época acumularia um novo vazamento de RAM.
        stop = threading.Event()
        q    = _queue.Queue(maxsize=self.prefetch_chunks)

        def _loader():
            for path in paths:
                if stop.is_set():
                    break
                try:
                    d = torch.load(path, map_location='cpu')
                except Exception as e:
                    print(f"\n  [WARNING] chunk corrompido ignorado: {path}\n  {e}", flush=True)
                    continue
                x_hw = d['x_hw']    # [B, C, H, W]
                y_hw = d['y_hw']    # [B, C, H, W]
                del d               # edge_index, node_x, etc. liberados aqui
                # put com timeout para checar stop_event se a fila estiver cheia
                while not stop.is_set():
                    try:
                        q.put((x_hw, y_hw), timeout=0.05)
                        break
                    except _queue.Full:
                        pass
            if not stop.is_set():
                q.put(None)         # sentinela de fim

        t = threading.Thread(target=_loader, daemon=True)
        t.start()

        buffer = []

        try:
            while True:
                # get com timeout — evita travar se o thread morrer sem sentinela
                while True:
                    try:
                        item = q.get(timeout=0.1)
                        break
                    except _queue.Empty:
                        if not t.is_alive():
                            item = None
                            break

                if item is None:
                    break
                x_hw, y_hw = item
                n    = x_hw.shape[0]
                perm = torch.randperm(n).tolist()

                for i in perm:
                    buffer.append((x_hw[i], y_hw[i]))  # views — sem clone

                    if len(buffer) >= self.buffer_size:
                        idx = rng.randrange(len(buffer))
                        yield buffer[idx]
                        buffer[idx] = buffer[-1]        # substitui pela última — O(1)
                        buffer.pop()

                # x_hw/y_hw saem de escopo; storage liberado quando não houver
                # mais views no buffer

            # ── Flush do buffer restante ──────────────────────────────────────
            rng.shuffle(buffer)
            yield from buffer

        finally:
            # Executado mesmo se GeneratorExit for lançado (gerador abandonado).
            # Sinaliza o thread e drena a fila para desbloqueá-lo do q.put().
            stop.set()
            while t.is_alive():
                try:
                    q.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)


# ── Helpers de device ────────────────────────────────────────────────────────

def _to_device(obj, device, non_blocking=False):
    """Move tensores para device recursivamente — suporta tensor, dict, list, tuple."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device, non_blocking) for v in obj)
    return obj


def _record_stream(obj, stream):
    """Chama record_stream em todos os tensores recursivamente."""
    if isinstance(obj, torch.Tensor):
        obj.record_stream(stream)
    elif isinstance(obj, dict):
        for v in obj.values():
            _record_stream(v, stream)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _record_stream(v, stream)


# ── GPU prefetcher ────────────────────────────────────────────────────────────

class CUDAPrefetcher:
    """
    Mantém o próximo batch sendo transferido para GPU em um CUDA stream
    separado enquanto a GPU processa o batch atual (overlap CPU↔GPU).

    Em modo CPU (device sem CUDA) comporta-se como iterador normal, sem
    overhead de stream.

    Agnóstico ao tipo de batch — suporta tuple (x, y), dict, ou qualquer
    estrutura aninhada de tensores via _to_device / _record_stream.

    Uso:
        loader   = DataLoader(dataset, ...)
        prefetch = CUDAPrefetcher(loader, device)
        for batch in prefetch:   # batch já está no device
            ...
    """

    def __init__(self, loader, device):
        self.loader     = loader
        self.device     = torch.device(device)
        self._stream    = (torch.cuda.Stream()
                           if self.device.type == 'cuda' else None)
        self._iter      = None
        self.next_batch = None

    # __iter__ é chamado no início de cada época pelo loop `for batch in loader`
    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        if self._stream is not None:
            # Aguarda a transferência assíncrona do batch atual terminar
            torch.cuda.current_stream().wait_stream(self._stream)
            # Impede que o allocator reutilize o buffer antes do kernel terminar
            _record_stream(batch, torch.cuda.current_stream())
        self._preload()         # dispara transferência do próximo batch
        return batch

    def _preload(self):
        try:
            raw = next(self._iter)
        except StopIteration:
            self.next_batch = None
            return
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self.next_batch = _to_device(raw, self.device, non_blocking=True)
        else:
            self.next_batch = _to_device(raw, self.device)

    def __len__(self):
        return len(self.loader)


# ── Qtree streaming dataset ───────────────────────────────────────────────────

class ChunkQtreeDataset(IterableDataset):
    """
    Versão de ChunkStreamDataset que preserva dados de grafo.

    Lê quad_chunk_*.pt do disco um a um e emite dicts individuais com
    grade (x_hw, y_hw) + grafo (node_x, node_y, edge_index, edge_attr).

    edge_index emitido com índices locais à amostra (0..L[i]-1).
    qtree_collate é responsável por re-offsetar ao montar o batch.

    Estratégia de memória:
      - x_hw, y_hw, node_x, node_y, edge_attr são views do chunk — storage
        liberado quando a última view sair do buffer.
      - edge_index de cada amostra é tensor independente (subtração cria cópia).
    """

    def __init__(self, chunk_paths, buffer_size, prefetch_chunks=2):
        self.chunk_paths     = list(chunk_paths)
        self.buffer_size     = buffer_size
        self.prefetch_chunks = prefetch_chunks

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.chunk_paths)

        if worker_info is not None:
            rng   = random.Random(worker_info.seed % (2 ** 32))
            paths = paths[worker_info.id :: worker_info.num_workers]
        else:
            rng = random.Random()

        rng.shuffle(paths)

        stop = threading.Event()
        q    = _queue.Queue(maxsize=self.prefetch_chunks)

        def _loader():
            for path in paths:
                if stop.is_set():
                    break
                try:
                    d = torch.load(path, map_location='cpu')
                except Exception as e:
                    print(f"\n  [WARNING] chunk corrompido ignorado: {path}\n  {e}", flush=True)
                    continue
                payload = {
                    'x_hw':       d['x_hw'],        # [B, 2, H, W]
                    'y_hw':       d['y_hw'],         # [B, 2, H, W]
                    'node_x':     d['node_x'],       # [S_tot, 5]
                    'node_y':     d['node_y'],       # [S_tot, 5]
                    'edge_index': d['edge_index'],   # [2, E_tot]  índices globais no chunk
                    'edge_attr':  d['edge_attr'],    # [E_tot, 4]
                    'L':          d['L'],            # [B]
                    'E_L':        d['E_L'],          # [B]
                }
                del d
                while not stop.is_set():
                    try:
                        q.put(payload, timeout=0.05)
                        break
                    except _queue.Full:
                        pass
            if not stop.is_set():
                q.put(None)

        t = threading.Thread(target=_loader, daemon=True)
        t.start()

        buffer = []

        try:
            while True:
                while True:
                    try:
                        item = q.get(timeout=0.1)
                        break
                    except _queue.Empty:
                        if not t.is_alive():
                            item = None
                            break

                if item is None:
                    break

                L, E_L = item['L'], item['E_L']
                B      = int(L.shape[0])

                # Offsets acumulados de nós e arestas dentro do chunk
                n_off = torch.cat([torch.zeros(1, dtype=torch.long), L.cumsum(0)])
                e_off = torch.cat([torch.zeros(1, dtype=torch.long), E_L.cumsum(0)])

                perm = torch.randperm(B).tolist()

                for i in perm:
                    ns, ne = int(n_off[i]), int(n_off[i + 1])
                    es, ee = int(e_off[i]), int(e_off[i + 1])

                    sample = {
                        'x_hw':       item['x_hw'][i],                      # view [2, H, W]
                        'y_hw':       item['y_hw'][i],                      # view [2, H, W]
                        'node_x':     item['node_x'][ns:ne],                # view [S_i, 5]
                        'node_y':     item['node_y'][ns:ne],                # view [S_i, 5]
                        'edge_index': item['edge_index'][:, es:ee] - ns,    # cópia [2, E_i]
                        'edge_attr':  item['edge_attr'][es:ee],             # view [E_i, 4]
                        'L':          int(L[i]),
                        'E_L':        int(E_L[i]),
                    }

                    buffer.append(sample)

                    if len(buffer) >= self.buffer_size:
                        idx = rng.randrange(len(buffer))
                        yield buffer[idx]
                        buffer[idx] = buffer[-1]
                        buffer.pop()

            # ── Flush do buffer restante ──────────────────────────────────────
            rng.shuffle(buffer)
            yield from buffer

        finally:
            stop.set()
            while t.is_alive():
                try:
                    q.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)


# ── build_loaders ─────────────────────────────────────────────────────────────

# [REMOVIDO] assinatura antiga de build_loaders — recebia train_portion e usava
#            random_split sobre GridChunkDataset (split ao nível de amostra).
#            Substituída pela versão abaixo que faz split ao nível de chunk e
#            recebe os parâmetros de streaming.
#
# def build_loaders(chunk_paths, batch_size, train_portion=0.70, seed=None):
#     ds = GridChunkDataset(chunk_paths)
#     n       = len(ds)
#     n_train = int(n * train_portion)
#     n_test  = n - n_train
#     generator = torch.Generator().manual_seed(seed) if seed is not None else None
#     train_ds, test_ds = torch.utils.data.random_split(
#         ds, [n_train, n_test], generator=generator
#     )
#     pin = torch.cuda.is_available()
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
#     test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
#     print(f"Dataset: {n} amostras  ({n_train} treino / {n_test} teste)")
#     return train_loader, test_loader


def qtree_collate(samples):
    """
    Collate para ChunkQtreeDataset.

    Cada amostra chega com edge_index de índices locais (0..L[i]-1).
    Esta função re-offseta os índices para o espaço global do batch.

    Saída : dict com grade [B, 2, H, W] e grafo com índices globais no batch.
    """
    L   = torch.tensor([s['L']   for s in samples], dtype=torch.long)   # [B]
    E_L = torch.tensor([s['E_L'] for s in samples], dtype=torch.long)   # [B]

    # Offset acumulado de nós para re-indexar edge_index no batch
    n_offsets = torch.cat([torch.zeros(1, dtype=torch.long), L.cumsum(0)[:-1]])  # [B]

    edge_index = torch.cat(
        [s['edge_index'] + off for s, off in zip(samples, n_offsets)],
        dim=1,
    )   # [2, E_tot]

    return {
        'x_hw':       torch.stack([s['x_hw']      for s in samples]),   # [B, 2, H, W]
        'y_hw':       torch.stack([s['y_hw']      for s in samples]),   # [B, 2, H, W]
        'node_x':     torch.cat(  [s['node_x']    for s in samples]),   # [S_tot, 5]
        'node_y':     torch.cat(  [s['node_y']    for s in samples]),   # [S_tot, 5]
        'edge_index': edge_index,                                         # [2, E_tot]
        'edge_attr':  torch.cat(  [s['edge_attr'] for s in samples]),   # [E_tot, 4]
        'L':          L,                                                  # [B]
        'E_L':        E_L,                                               # [B]
    }


def split_chunk_paths(chunk_paths, train_split, seed=None):
    """
    Divide chunk_paths em treino/teste ao nível de chunk — mesmo shuffle
    determinístico (via seed) usado por build_loaders, exposto separadamente
    para que scripts/eval.py possa reproduzir o split de uma run sem duplicar
    a lógica.

    Returns
    -------
    train_paths, test_paths : list[str]
    """
    paths = list(chunk_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    n_train = int(len(paths) * train_split)
    n_train = max(n_train, 1) if len(paths) >= 2 else len(paths)  # 1 chunk → tudo para treino, sem test
    return paths[:n_train], paths[n_train:]


def build_loaders(chunk_paths, batch_size, train_split,
                  buffer_size, num_workers, prefetch_factor,
                  seed=None, mode='grid'):
    """
    Divide chunk_paths em treino/teste ao nível de chunk, cria um dataset
    para cada split e retorna os DataLoaders.

    Nenhum chunk é lido aqui — apenas as listas de caminhos são particionadas.

    Parameters
    ----------
    chunk_paths    : list[str]       caminhos dos quad_chunk_*.pt
    batch_size     : int             amostras por mini-lote
    train_split    : float           fração de chunks para treino
    buffer_size    : int             amostras no reservatório de shuffle
    num_workers    : int             workers paralelos de I/O
    prefetch_factor: int             batches pré-buscados por worker
    seed           : int|None        semente para o split reproduzível
    mode           : 'grid'|'qtree'  'grid' usa ChunkStreamDataset + collate padrão;
                                     'qtree' usa ChunkQtreeDataset + qtree_collate

    Returns
    -------
    train_loader, test_loader
    """
    if mode == 'grid':
        cls, collate_fn = ChunkStreamDataset, None
    else:
        cls, collate_fn = ChunkQtreeDataset, qtree_collate

    # No Windows, shared memory entre processos (necessária com num_workers > 0)
    # falha para tensores grandes (erro 1455 = ERROR_COMMITMENT_LIMIT).
    # Os datasets já têm thread de background para prefetch de chunks, então
    # num_workers=0 não prejudica o throughput de I/O.
    if sys.platform == 'win32' and num_workers > 0:
        print(f"  [Windows] num_workers={num_workers} → forçado para 0 "
              f"(shared memory insuficiente para chunks {'qtree' if mode == 'qtree' else 'grid'} grandes)")
        num_workers = 0

    paths = list(chunk_paths)
    if not paths:
        raise FileNotFoundError(
            "Nenhum chunk encontrado. Verifique se os arquivos data_chunk_*.pt estão em "
            f"data/torch/data_chunks/<dataset>/ e se o campo 'dataset' em NnCfg aponta "
            "para o subdiretório correto. Execute build_data_chunks.py antes do treino."
        )

    # [REMOVIDO] split inline — movido para split_chunk_paths() para que
    # scripts/eval.py possa reproduzir o mesmo split sem duplicar a lógica.
    #
    # rng = random.Random(seed)
    # rng.shuffle(paths)
    # n_train     = int(len(paths) * train_split)
    # n_train     = max(n_train, 1) if len(paths) >= 2 else len(paths)
    # train_paths = paths[:n_train]
    # test_paths  = paths[n_train:]
    train_paths, test_paths = split_chunk_paths(paths, train_split, seed)

    train_ds = cls(train_paths, buffer_size, prefetch_chunks=2)
    test_ds  = cls(test_paths,  buffer_size, prefetch_chunks=2)

    pin = torch.cuda.is_available()

    # prefetch_factor e persistent_workers só são válidos com num_workers > 0
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=pin, collate_fn=collate_fn)
    if num_workers > 0:
        loader_kw['prefetch_factor']    = prefetch_factor
        loader_kw['persistent_workers'] = True

    train_loader = DataLoader(train_ds, **loader_kw)
    test_loader  = DataLoader(test_ds,  **loader_kw)

    print(f"Chunks: {len(paths)} total  "
          f"({len(train_paths)} treino / {len(test_paths)} teste)  "
          f"| buffer {buffer_size} amostras  "
          f"| workers {num_workers}  "
          f"| mode '{mode}'", flush=True)
    return train_loader, test_loader
