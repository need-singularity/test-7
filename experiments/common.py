"""
공통 유틸리티: Phase 4/5 + Exp A-D에서 공유하는 함수들.

기존 phase 파일은 수정하지 않음 (독립 실행 보존).
새 실험(Exp A-D)은 이 모듈을 import하여 사용.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

MODEL_PATH = Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "poc_topology"
PERSISTENCE_THRESHOLD = 0.01

SUFFIXES = [
    " and", " but", " however", " because", " which",
    " the", " a", " in", " of", " that",
    ".", "?", "!", " not", " never",
    " always", " perhaps", " certainly", " impossible", " undefined",
]


def load_model():
    from llama_cpp import Llama
    print("Loading model...")
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=512, n_gpu_layers=-1,
                embedding=True, verbose=False)
    print("Loaded.\n")
    return llm


def extract_embeddings(llm, prompt):
    """프롬프트에서 point cloud 추출 (prefix 누적 + suffix 변형)."""
    points = []

    emb = llm.embed(prompt)
    points.append(emb[-1] if isinstance(emb[0], list) else emb)

    tokens = llm.tokenize(prompt.encode('utf-8'))
    for i in range(1, len(tokens)):
        prefix = llm.detokenize(tokens[:i]).decode('utf-8', errors='replace')
        if prefix.strip():
            emb = llm.embed(prefix)
            points.append(emb[-1] if isinstance(emb[0], list) else emb)

    for suffix in SUFFIXES:
        emb = llm.embed(prompt + suffix)
        points.append(emb[-1] if isinstance(emb[0], list) else emb)

    return np.array(points)


def compute_full_topology(points):
    """PCA → Ripser PH → walls/β₁/persistence 정보 반환."""
    from ripser import ripser

    n, d = points.shape
    pca_dim = min(50, n - 1, d)
    if pca_dim >= 2:
        reduced = PCA(n_components=pca_dim).fit_transform(points)
    else:
        reduced = points

    dm = squareform(pdist(reduced))
    result = ripser(dm, maxdim=1, distance_matrix=True, do_cocycles=True)

    dgm1 = result['dgms'][1]
    cocycles = result['cocycles'][1]

    walls = []
    for idx, (b, d_) in enumerate(dgm1):
        p = d_ - b
        if np.isfinite(d_) and p > PERSISTENCE_THRESHOLD:
            cc = cocycles[idx]
            verts = sorted(set(int(r[0]) for r in cc) | set(int(r[1]) for r in cc))
            walls.append({
                'birth': float(b), 'death': float(d_), 'persistence': float(p),
                'vertex_indices': verts, 'idx': idx,
            })
    walls.sort(key=lambda w: w['persistence'], reverse=True)

    beta0_bars = [(float(b), float(d_)) for b, d_ in result['dgms'][0]
                  if np.isfinite(d_) and d_ - b > PERSISTENCE_THRESHOLD]

    return {
        'beta0': len(beta0_bars) + sum(1 for b, d_ in result['dgms'][0] if not np.isfinite(d_)),
        'beta1': len(walls),
        'walls': walls,
        'max_persistence': walls[0]['persistence'] if walls else 0.0,
        'total_persistence': sum(w['persistence'] for w in walls),
        'mean_persistence': np.mean([w['persistence'] for w in walls]) if walls else 0.0,
        'result': result,
    }


def compute_topology_from_distance_matrix(dm):
    """거리 행렬에서 직접 PH 계산 (쌍곡 거리 등에 사용)."""
    from ripser import ripser

    result = ripser(dm, maxdim=1, distance_matrix=True, do_cocycles=True)

    dgm1 = result['dgms'][1]
    cocycles = result['cocycles'][1]

    walls = []
    for idx, (b, d_) in enumerate(dgm1):
        p = d_ - b
        if np.isfinite(d_) and p > PERSISTENCE_THRESHOLD:
            cc = cocycles[idx]
            verts = sorted(set(int(r[0]) for r in cc) | set(int(r[1]) for r in cc))
            walls.append({
                'birth': float(b), 'death': float(d_), 'persistence': float(p),
                'vertex_indices': verts, 'idx': idx,
            })
    walls.sort(key=lambda w: w['persistence'], reverse=True)

    return {
        'beta1': len(walls),
        'walls': walls,
        'max_persistence': walls[0]['persistence'] if walls else 0.0,
        'total_persistence': sum(w['persistence'] for w in walls),
    }


def radial_perturbation(embeddings, wall, alpha):
    """단일 wall에 대한 radial 수축."""
    perturbed = embeddings.copy()
    verts = wall['vertex_indices']
    cycle_points = embeddings[verts]
    center = cycle_points.mean(axis=0)

    for v in verts:
        radial = perturbed[v] - center
        norm = np.linalg.norm(radial)
        if norm > 1e-10:
            perturbed[v] = perturbed[v] - alpha * (radial / norm)
    return perturbed


def multi_wall_perturbation(embeddings, walls, alpha, max_walls=None):
    """여러 wall을 동시에 수축 — 가장 persistent한 것부터."""
    perturbed = embeddings.copy()
    target_walls = walls[:max_walls] if max_walls else walls

    for wall in target_walls:
        verts = wall['vertex_indices']
        cycle_points = perturbed[verts]
        center = cycle_points.mean(axis=0)

        for v in verts:
            radial = perturbed[v] - center
            norm = np.linalg.norm(radial)
            if norm > 1e-10:
                perturbed[v] = perturbed[v] - alpha * (radial / norm)

    return perturbed


def emergence_score(topo_before, topo_after):
    """3-channel passage score: wall_reduction + pers_reduction + stability."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    from tecs.emergence import EmergenceDetector

    detector = EmergenceDetector()
    topo_b = {'beta1': topo_before['beta1'], 'max_persistence_h1': topo_before['max_persistence']}
    topo_a = {'beta1': topo_after['beta1'], 'max_persistence_h1': topo_after['max_persistence']}
    hier = {'hierarchy_score': 0.5}

    score_before = detector.score(topo_b, hier)
    score_after = detector.score(topo_a, hier)

    b1_before = topo_before['beta1']
    b1_after = topo_after['beta1']

    if b1_before > 0:
        wall_reduction = max(0.0, (b1_before - b1_after) / b1_before)
    else:
        wall_reduction = 1.0

    mp_before = topo_before['max_persistence']
    mp_after = topo_after['max_persistence']
    if mp_before > 0:
        pers_reduction = max(0.0, (mp_before - mp_after) / mp_before)
    else:
        pers_reduction = 1.0

    b0_before = topo_before.get('beta0', 1)
    b0_after = topo_after.get('beta0', 1)
    b0_change = abs(b0_after - b0_before)
    stability = max(0.0, 1.0 - b0_change / max(b0_before, 1))

    passage_score = 0.4 * wall_reduction + 0.3 * pers_reduction + 0.3 * stability

    return {
        'passage_score': round(passage_score, 4),
        'wall_reduction': round(wall_reduction, 4),
        'pers_reduction': round(pers_reduction, 4),
        'stability': round(stability, 4),
        'tecs_before': score_before.to_dict(),
        'tecs_after': score_after.to_dict(),
        'beta1_change': b1_after - b1_before,
        'max_pers_change': mp_after - mp_before,
    }


def get_passage_direction(embeddings, wall):
    """wall의 passage direction 계산 (cocycle 기반)."""
    verts = wall['vertex_indices']
    cycle_points = embeddings[verts]
    center = cycle_points.mean(axis=0)
    cycle_centered = cycle_points - center
    local_dim = min(len(verts) - 1, 10, embeddings.shape[1])
    if local_dim < 1:
        return None, None

    local_pca = PCA(n_components=local_dim)
    local_pca.fit(cycle_centered)
    Q = local_pca.components_.T
    P_orth = np.eye(embeddings.shape[1]) - Q @ Q.T

    projected = (embeddings - center) @ P_orth.T
    cov = np.cov(projected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    for i, ev in enumerate(eigenvalues):
        if ev > 1e-10:
            d = eigenvectors[:, i]
            return d / (np.linalg.norm(d) + 1e-10), center
    return None, None


# ── 프롬프트 세트 ──────────────────────────────────

PROMPTS_STANDARD = [
    ("factual", "The capital of France is"),
    ("factual2", "Water boils at a temperature of"),
    ("reasoning", "If all roses are flowers and some flowers fade quickly, then"),
    ("creative", "A color that doesn't exist yet would look like"),
    ("creative2", "If mathematics were a living organism, its heartbeat would be"),
    ("boundary", "The solution to the Riemann hypothesis involves"),
    ("boundary2", "The mechanism by which consciousness emerges from neurons is"),
]

PROMPTS_SUBSET = [
    ("factual", "The capital of France is"),
    ("creative", "A color that doesn't exist yet would look like"),
    ("reasoning", "If all roses are flowers and some flowers fade quickly, then"),
    ("boundary", "The mechanism by which consciousness emerges from neurons is"),
]


def numpy_converter(obj):
    """JSON 직렬화용 numpy 타입 변환."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
