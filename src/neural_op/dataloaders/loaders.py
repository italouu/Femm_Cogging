import random
import sys
import threading
import queue as _queue
import torch
from torch.utils.data import IterableDataset, DataLoader


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

    normalizer : Normalizer|None — se fornecido, aplica encode_batch() logo
        após mover o batch pro device (ver src/neural_op/normalization.py).
        Transparente para step_fn: a loss passa a ser calculada em espaço
        normalizado sem nenhuma mudança no step_fn de cada arch.

    Uso:
        loader   = DataLoader(dataset, ...)
        prefetch = CUDAPrefetcher(loader, device, normalizer=normalizer)
        for batch in prefetch:   # batch já está no device (e normalizado)
            ...
    """

    def __init__(self, loader, device, normalizer=None):
        self.loader     = loader
        self.device     = torch.device(device)
        self.normalizer = normalizer
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
                if self.normalizer is not None:
                    self.next_batch = self.normalizer.encode_batch(self.next_batch)
        else:
            self.next_batch = _to_device(raw, self.device)
            if self.normalizer is not None:
                self.next_batch = self.normalizer.encode_batch(self.next_batch)

    def __len__(self):
        return len(self.loader)


# ── Streaming dataset do grafo de vértices (mode='vertex') ──────────────────

class ChunkVertexDataset(IterableDataset):
    """
    Versão de ChunkStreamDataset que preserva dados de grafo (mode='vertex':
    grafo de vértices da malha do FEMM -- FNO_GNN, GNN_PostBase).

    Lê quad_chunk_*.pt do disco um a um e emite dicts individuais com
    grade (x_hw, y_hw) + grafo (node_x, node_y, edge_index, edge_attr).

    edge_index emitido com índices locais à amostra (0..L[i]-1).
    vertex_collate é responsável por re-offsetar ao montar o batch.

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


# ── Grafo duplo (mode='bipartite') ────────────────────────────────────────

class ChunkBipartiteDataset(IterableDataset):
    """
    Versão de ChunkVertexDataset para o grafo duplo do mode='bipartite'
    (vértices + elementos + arestas cruzadas, ver src/data_gen/parsers/bipartite_structure.py
    e src/neural_op/archs/fno_bipartite_gnn.py).

    edge_index (vértice-vértice) e cross_edge_index (elemento->vértice) são
    emitidos com índices LOCAIS à amostra (0..L[i]-1 / 0..elem_L[i]-1) --
    bipartite_collate re-offseta os dois espaços ao montar o batch.
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
                    'x_hw':             d['x_hw'],              # [B, 2, H, W]
                    'y_hw':             d['y_hw'],               # [B, 1, H, W]
                    'node_x':           d['node_x'],             # [S_tot, 2]
                    'node_y':           d['node_y'],             # [S_tot, 1]
                    'edge_index':       d['edge_index'],         # [2, E_tot]  índices globais no chunk
                    'edge_attr':        d['edge_attr'],          # [E_tot, 3]
                    'elem_x':           d['elem_x'],             # [M_tot, 5]
                    'cross_edge_index': d['cross_edge_index'],   # [2, C_tot]  índices globais no chunk
                    'cross_edge_attr':  d['cross_edge_attr'],    # [C_tot, 1]
                    'L':                d['L'],                  # [B]
                    'elem_L':           d['elem_L'],             # [B]
                    'E_L':              d['E_L'],                # [B]
                    'C_L':              d['C_L'],                # [B]
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

                L, elem_L, E_L, C_L = item['L'], item['elem_L'], item['E_L'], item['C_L']
                B = int(L.shape[0])

                # Offsets acumulados de nós/elementos/arestas dentro do chunk
                n_off = torch.cat([torch.zeros(1, dtype=torch.long), L.cumsum(0)])
                m_off = torch.cat([torch.zeros(1, dtype=torch.long), elem_L.cumsum(0)])
                e_off = torch.cat([torch.zeros(1, dtype=torch.long), E_L.cumsum(0)])
                c_off = torch.cat([torch.zeros(1, dtype=torch.long), C_L.cumsum(0)])

                perm = torch.randperm(B).tolist()

                for i in perm:
                    ns, ne = int(n_off[i]), int(n_off[i + 1])
                    ms, me = int(m_off[i]), int(m_off[i + 1])
                    es, ee = int(e_off[i]), int(e_off[i + 1])
                    cs, ce = int(c_off[i]), int(c_off[i + 1])

                    cei = item['cross_edge_index'][:, cs:ce].clone()
                    cei[0] -= ms   # linha 0: índice do elemento
                    cei[1] -= ns   # linha 1: índice do vértice

                    sample = {
                        'x_hw':             item['x_hw'][i],                      # view [2, H, W]
                        'y_hw':             item['y_hw'][i],                      # view [1, H, W]
                        'node_x':           item['node_x'][ns:ne],                # view [S_i, 2]
                        'node_y':           item['node_y'][ns:ne],                # view [S_i, 1]
                        'edge_index':       item['edge_index'][:, es:ee] - ns,    # cópia [2, E_i]
                        'edge_attr':        item['edge_attr'][es:ee],             # view [E_i, 3]
                        'elem_x':           item['elem_x'][ms:me],                # view [M_i, 5]
                        'cross_edge_index': cei,                                   # cópia [2, C_i]
                        'cross_edge_attr':  item['cross_edge_attr'][cs:ce],       # view [C_i, 1]
                        'L':                int(L[i]),
                        'elem_L':           int(elem_L[i]),
                        'E_L':              int(E_L[i]),
                        'C_L':              int(C_L[i]),
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


def bipartite_collate(samples):
    """
    Collate para ChunkBipartiteDataset.

    Re-offseta DOIS espaços de índice: edge_index (vértice-vértice, por nó)
    e cross_edge_index (elemento->vértice -- linha 0 por elemento, linha 1
    por nó, mesmo offset de nó usado em edge_index).

    Saída : dict com grade [B, 2/1, H, W], grafo de vértices e grafo de
    elementos, todos com índices globais no batch.
    """
    L      = torch.tensor([s['L']      for s in samples], dtype=torch.long)   # [B]
    elem_L = torch.tensor([s['elem_L'] for s in samples], dtype=torch.long)   # [B]
    E_L    = torch.tensor([s['E_L']    for s in samples], dtype=torch.long)   # [B]
    C_L    = torch.tensor([s['C_L']    for s in samples], dtype=torch.long)   # [B]

    n_offsets = torch.cat([torch.zeros(1, dtype=torch.long), L.cumsum(0)[:-1]])       # [B]
    m_offsets = torch.cat([torch.zeros(1, dtype=torch.long), elem_L.cumsum(0)[:-1]])  # [B]

    edge_index = torch.cat(
        [s['edge_index'] + off for s, off in zip(samples, n_offsets)],
        dim=1,
    )   # [2, E_tot]

    cross_edge_index = torch.cat(
        [torch.stack([s['cross_edge_index'][0] + m_off, s['cross_edge_index'][1] + n_off])
         for s, m_off, n_off in zip(samples, m_offsets, n_offsets)],
        dim=1,
    )   # [2, C_tot]

    return {
        'x_hw':             torch.stack([s['x_hw']             for s in samples]),   # [B, 2, H, W]
        'y_hw':             torch.stack([s['y_hw']             for s in samples]),   # [B, 1, H, W]
        'node_x':           torch.cat(  [s['node_x']           for s in samples]),   # [S_tot, 2]
        'node_y':           torch.cat(  [s['node_y']           for s in samples]),   # [S_tot, 1]
        'edge_index':       edge_index,                                                # [2, E_tot]
        'edge_attr':        torch.cat(  [s['edge_attr']        for s in samples]),   # [E_tot, 3]
        'elem_x':           torch.cat(  [s['elem_x']           for s in samples]),   # [M_tot, 5]
        'cross_edge_index': cross_edge_index,                                          # [2, C_tot]
        'cross_edge_attr':  torch.cat(  [s['cross_edge_attr']  for s in samples]),   # [C_tot, 1]
        'L':                L,                                                         # [B]
        'elem_L':           elem_L,                                                    # [B]
        'E_L':              E_L,                                                       # [B]
        'C_L':              C_L,                                                       # [B]
    }


# ── build_loaders ─────────────────────────────────────────────────────────────


def vertex_collate(samples):
    """
    Collate para ChunkVertexDataset.

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


def split_chunk_paths(chunk_paths, train_split, val_split, seed=None):
    """
    Divide chunk_paths em treino/validação/teste ao nível de chunk — mesmo shuffle
    determinístico (via seed) usado por build_loaders, exposto separadamente
    para que o pós-processamento (src/postprocess/samples.py) e o Normalizer
    reproduzam o split de uma run sem duplicar a lógica.

    n_train = round(n·train_split), n_val = round(n·val_split), teste = complemento.
      treino    : ajuste dos pesos e stats da normalização;
      validação : parada antecipada (GL/paciência) e escolha do best.pth;
      teste     : só avaliação final (pós-processamento), nunca visto no treino.

    Returns
    -------
    train_paths, val_paths, test_paths : list[str]
    """
    paths = list(chunk_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    n_train = round(len(paths) * train_split)
    n_val   = round(len(paths) * val_split)
    if n_train < 1 or n_val < 1 or n_train + n_val >= len(paths):
        raise ValueError(f"split {train_split}/{val_split} de {len(paths)} chunks deixa treino, "
                         f"validação ou teste vazio ({n_train}/{n_val}/{len(paths) - n_train - n_val})")
    train_paths = paths[:n_train]
    val_paths   = paths[n_train:n_train + n_val]
    test_paths  = paths[n_train + n_val:]
    return train_paths, val_paths, test_paths


def build_loaders(chunk_paths, batch_size, train_split, val_split,
                  buffer_size, num_workers, prefetch_factor,
                  seed=None, mode='grid'):
    """
    Divide chunk_paths em treino/validação/teste ao nível de chunk (split_chunk_paths)
    e retorna os DataLoaders de treino e validação. O teste não tem loader aqui: só o
    pós-processamento o lê.

    Nenhum chunk é lido aqui — apenas as listas de caminhos são particionadas.

    Parameters
    ----------
    chunk_paths    : list[str]       caminhos dos data_chunk_*.pt
    batch_size     : int             amostras por mini-lote
    train_split    : float           fração de chunks para treino
    val_split      : float           fração de chunks para validação (teste = complemento)
    buffer_size    : int             amostras no reservatório de shuffle
    num_workers    : int             workers paralelos de I/O
    prefetch_factor: int             batches pré-buscados por worker
    seed           : int|None        semente para o split reproduzível
    mode           : str             'grid' usa ChunkStreamDataset + collate padrão;
                                     'vertex' usa ChunkVertexDataset + vertex_collate;
                                     'bipartite' usa ChunkBipartiteDataset +
                                     bipartite_collate

    Returns
    -------
    train_loader, val_loader, train_paths, val_paths, test_paths
    """
    if mode == 'grid':
        cls, collate_fn = ChunkStreamDataset, None
    elif mode == 'bipartite':
        cls, collate_fn = ChunkBipartiteDataset, bipartite_collate
    elif mode == 'vertex':
        cls, collate_fn = ChunkVertexDataset, vertex_collate
    else:
        raise ValueError(f"mode de loader desconhecido: {mode!r} "
                         f"(esperado 'grid', 'vertex' ou 'bipartite')")

    # No Windows, shared memory entre processos (necessária com num_workers > 0)
    # falha para tensores grandes (erro 1455 = ERROR_COMMITMENT_LIMIT).
    # Os datasets já têm thread de background para prefetch de chunks, então
    # num_workers=0 não prejudica o throughput de I/O.
    if sys.platform == 'win32' and num_workers > 0:
        print(f"  [Windows] num_workers={num_workers} → forçado para 0 "
              f"(shared memory insuficiente para chunks '{mode}' grandes)")
        num_workers = 0

    paths = list(chunk_paths)
    if not paths:
        raise FileNotFoundError(
            "Nenhum chunk encontrado. Verifique se os arquivos data_chunk_*.pt estão em "
            "data/torch/data_chunks/<dataset>/ e se o campo 'dataset' em NnCfg aponta "
            "para o subdiretório correto. Gere os datasets antes do treino: "
            "python -m scripts.build_datasets"
        )

    train_paths, val_paths, test_paths = split_chunk_paths(paths, train_split, val_split, seed)

    train_ds = cls(train_paths, buffer_size, prefetch_chunks=2)
    val_ds   = cls(val_paths,   buffer_size, prefetch_chunks=2)

    pin = torch.cuda.is_available()

    # prefetch_factor e persistent_workers só são válidos com num_workers > 0
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=pin, collate_fn=collate_fn)
    if num_workers > 0:
        loader_kw['prefetch_factor']    = prefetch_factor
        loader_kw['persistent_workers'] = True

    train_loader = DataLoader(train_ds, **loader_kw)
    val_loader   = DataLoader(val_ds,   **loader_kw)

    print(f"Chunks: {len(paths)} total  "
          f"({len(train_paths)} treino / {len(val_paths)} validação / {len(test_paths)} teste)  "
          f"| buffer {buffer_size} amostras  "
          f"| workers {num_workers}  "
          f"| mode '{mode}'", flush=True)
    return train_loader, val_loader, train_paths, val_paths, test_paths
