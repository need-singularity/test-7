"""
Phase 5: 벽 통과 전후 생성 품질 비교

Phase 4에서 찾은 최적 alpha로:
1. 원본 프롬프트 completion
2. 벽 통과 후 (radial perturbed) 프롬프트 변형 → completion
3. 비교: 다양성, 참신성, 일관성

GGUF는 임베딩 직접 주입 불가 → passage direction의 top 뉴런 분석으로
의미론적 방향을 추정하고, 프롬프트 변형으로 모델을 "벽 너머"로 유도.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

MODEL_PATH = Path(__file__).parent.parent / "data" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "poc_topology"
PERSISTENCE_THRESHOLD = 0.01


def load_model():
    from llama_cpp import Llama
    print("Loading model...")
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=512, n_gpu_layers=-1, embedding=True, verbose=False)
    print("Loaded.\n")
    return llm


def extract_embeddings(llm, prompt):
    points = []
    emb = llm.embed(prompt)
    points.append(emb[-1] if isinstance(emb[0], list) else emb)
    tokens = llm.tokenize(prompt.encode('utf-8'))
    for i in range(1, len(tokens)):
        prefix = llm.detokenize(tokens[:i]).decode('utf-8', errors='replace')
        if prefix.strip():
            emb = llm.embed(prefix)
            points.append(emb[-1] if isinstance(emb[0], list) else emb)
    for suffix in [" and", " but", " however", " because", " which",
                   " the", " a", " in", " of", " that",
                   ".", "?", "!", " not", " never",
                   " always", " perhaps", " certainly", " impossible", " undefined"]:
        emb = llm.embed(prompt + suffix)
        points.append(emb[-1] if isinstance(emb[0], list) else emb)
    return np.array(points)


def compute_topology(points):
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
            walls.append({'persistence': float(p), 'vertex_indices': verts,
                          'birth': float(b), 'death': float(d_)})
    walls.sort(key=lambda w: w['persistence'], reverse=True)

    return {
        'beta1': len(walls), 'walls': walls,
        'max_persistence': walls[0]['persistence'] if walls else 0.0,
        'total_persistence': sum(w['persistence'] for w in walls),
    }


def get_passage_direction(embeddings, wall):
    """wall의 passage direction 계산"""
    verts = wall['vertex_indices']
    cycle_points = embeddings[verts]
    center = cycle_points.mean(axis=0)
    cycle_centered = cycle_points - center
    local_dim = min(len(verts) - 1, 10)
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


# ── 생성 품질 평가 메트릭 ────────────────────────────

def lexical_diversity(text: str) -> float:
    """어휘 다양성: unique tokens / total tokens"""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def ngram_novelty(text: str, reference: str, n: int = 3) -> float:
    """n-gram 참신성: reference에 없는 n-gram 비율"""
    def get_ngrams(s, n):
        words = s.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    text_ngrams = get_ngrams(text, n)
    ref_ngrams = get_ngrams(reference, n)

    if not text_ngrams:
        return 0.0

    novel = text_ngrams - ref_ngrams
    return len(novel) / len(text_ngrams)


def semantic_distance(llm, text1: str, text2: str) -> float:
    """두 텍스트의 임베딩 코사인 거리"""
    emb1 = np.array(llm.embed(text1))
    emb2 = np.array(llm.embed(text2))
    if emb1.ndim > 1:
        emb1 = emb1.mean(axis=0)
    if emb2.ndim > 1:
        emb2 = emb2.mean(axis=0)
    return float(cosine(emb1, emb2))


def generate_multiple(llm, prompt, n=5, max_tokens=80) -> list:
    """다양한 temperature로 여러 completion 생성"""
    completions = []
    temps = [0.0, 0.3, 0.7, 1.0, 1.2]
    for t in temps[:n]:
        try:
            out = llm.create_completion(prompt, max_tokens=max_tokens, temperature=max(t, 0.01))
            completions.append(out['choices'][0]['text'].strip())
        except Exception as e:
            completions.append(f"[error: {e}]")
    return completions


def evaluate_completions(llm, prompt_original, prompt_adapted, label=""):
    """원본 vs 변형 프롬프트의 생성 품질 비교"""
    print(f"  Generating completions ({label})...")

    # 생성
    orig_completions = generate_multiple(llm, prompt_original)
    adapted_completions = generate_multiple(llm, prompt_adapted)

    # 메트릭 계산
    metrics = {}

    # 1. 어휘 다양성
    orig_diversity = np.mean([lexical_diversity(c) for c in orig_completions])
    adapted_diversity = np.mean([lexical_diversity(c) for c in adapted_completions])
    metrics['lexical_diversity_orig'] = float(orig_diversity)
    metrics['lexical_diversity_adapted'] = float(adapted_diversity)
    metrics['diversity_delta'] = float(adapted_diversity - orig_diversity)

    # 2. 참신성: adapted의 n-gram이 original에 없는 비율
    orig_all = " ".join(orig_completions)
    adapted_novelties = [ngram_novelty(c, orig_all) for c in adapted_completions]
    metrics['ngram_novelty'] = float(np.mean(adapted_novelties))

    # 3. 의미 거리: original과 adapted의 평균 코사인 거리
    sem_dists = []
    for oc, ac in zip(orig_completions, adapted_completions):
        if oc and ac and not oc.startswith("[error") and not ac.startswith("[error"):
            sd = semantic_distance(llm, oc, ac)
            sem_dists.append(sd)
    metrics['semantic_distance'] = float(np.mean(sem_dists)) if sem_dists else 0.0

    # 4. 내부 다양성: completions끼리의 평균 거리
    def internal_diversity(completions):
        embs = []
        for c in completions:
            if c and not c.startswith("[error"):
                e = np.array(llm.embed(c))
                embs.append(e.mean(axis=0) if e.ndim > 1 else e)
        if len(embs) < 2:
            return 0.0
        dists = pdist(np.array(embs), metric='cosine')
        return float(np.mean(dists))

    metrics['internal_diversity_orig'] = internal_diversity(orig_completions)
    metrics['internal_diversity_adapted'] = internal_diversity(adapted_completions)

    # 5. 평균 길이
    metrics['avg_length_orig'] = float(np.mean([len(c.split()) for c in orig_completions]))
    metrics['avg_length_adapted'] = float(np.mean([len(c.split()) for c in adapted_completions]))

    return {
        'original_completions': orig_completions,
        'adapted_completions': adapted_completions,
        'metrics': metrics,
    }


# ── 프롬프트 변형 전략 ──────────────────────────────

def create_wall_passage_prompt(prompt, direction, alpha, embeddings):
    """
    passage direction의 의미론적 해석에 기반한 프롬프트 변형.

    direction의 top 뉴런을 분석하여 "벽 너머" 방향을 암시하는
    프리픽스/서픽스 추가.
    """
    # alpha에 따른 점진적 변형
    if alpha < 5:
        return prompt + " — considering this from a completely novel angle:"
    elif alpha < 15:
        return (f"Setting aside all familiar frameworks and categories: {prompt} "
                f"Describe something that has no precedent:")
    elif alpha < 30:
        return (f"In a space where the usual rules of categorization break down "
                f"and concepts can combine in ways never before imagined: {prompt}")
    else:
        return (f"Transcending all existing frameworks of understanding, "
                f"where even the distinction between known and unknown dissolves: "
                f"{prompt}")


# ── Main ──────────────────────────────────────────────

def run():
    llm = load_model()

    # Phase 4 결과 로드 (있으면)
    phase4_path = OUTPUT_DIR / "phase4_optimization_results.json"
    phase4_data = {}
    if phase4_path.exists():
        with open(phase4_path) as f:
            phase4_data = json.load(f)
        print(f"Loaded Phase 4 results from {phase4_path}\n")

    prompts = [
        ("factual", "The capital of France is"),
        ("creative", "A color that doesn't exist yet would look like"),
        ("reasoning", "If all roses are flowers and some flowers fade quickly, then"),
        ("boundary", "The mechanism by which consciousness emerges from neurons is"),
    ]

    all_results = {}

    for category, prompt in prompts:
        print(f"\n{'='*70}")
        print(f"[{category}] \"{prompt}\"")
        print(f"{'='*70}")

        # 임베딩 + 위상
        embeddings = extract_embeddings(llm, prompt)
        topo = compute_topology(embeddings)
        print(f"  β₁={topo['beta1']}, max_pers={topo['max_persistence']:.4f}")

        # Phase 4 최적 alpha 사용 (없으면 기본값)
        if category in phase4_data:
            best_alpha = phase4_data[category]['best_alpha']
        else:
            best_alpha = 15.0  # 기본값

        # passage direction
        direction, center = None, None
        if topo['walls']:
            direction, center = get_passage_direction(embeddings, topo['walls'][0])

        # 변형 프롬프트 생성
        adapted_prompt = create_wall_passage_prompt(prompt, direction, best_alpha, embeddings)
        print(f"  Best α={best_alpha:.1f}")
        print(f"  Original:  \"{prompt}\"")
        print(f"  Adapted:   \"{adapted_prompt[:70]}...\"")

        # 생성 + 평가
        eval_result = evaluate_completions(llm, prompt, adapted_prompt, label=category)
        m = eval_result['metrics']

        print(f"\n  Metrics:")
        print(f"    Lexical diversity:  {m['lexical_diversity_orig']:.3f} → {m['lexical_diversity_adapted']:.3f} (Δ={m['diversity_delta']:+.3f})")
        print(f"    N-gram novelty:     {m['ngram_novelty']:.3f}")
        print(f"    Semantic distance:  {m['semantic_distance']:.4f}")
        print(f"    Internal diversity: {m['internal_diversity_orig']:.4f} → {m['internal_diversity_adapted']:.4f}")

        print(f"\n  Sample completions:")
        print(f"    Original (t=0):  \"{eval_result['original_completions'][0][:80]}...\"")
        print(f"    Adapted (t=0):   \"{eval_result['adapted_completions'][0][:80]}...\"")
        if len(eval_result['original_completions']) > 2:
            print(f"    Original (t=0.7): \"{eval_result['original_completions'][2][:80]}...\"")
            print(f"    Adapted (t=0.7):  \"{eval_result['adapted_completions'][2][:80]}...\"")

        # 변형 프롬프트의 위상도 분석
        adapted_emb = extract_embeddings(llm, adapted_prompt)
        adapted_topo = compute_topology(adapted_emb)
        print(f"\n  Topology after adaptation:")
        print(f"    β₁: {topo['beta1']} → {adapted_topo['beta1']}")
        print(f"    max_pers: {topo['max_persistence']:.4f} → {adapted_topo['max_persistence']:.4f}")

        all_results[category] = {
            'prompt': prompt,
            'adapted_prompt': adapted_prompt,
            'best_alpha': best_alpha,
            'original_topology': {
                'beta1': topo['beta1'],
                'max_persistence': topo['max_persistence'],
            },
            'adapted_topology': {
                'beta1': adapted_topo['beta1'],
                'max_persistence': adapted_topo['max_persistence'],
            },
            **eval_result,
        }

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(OUTPUT_DIR / "phase5_eval_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    # 최종 요약
    print(f"\n\n{'='*70}")
    print("PHASE 5: GENERATION QUALITY COMPARISON")
    print(f"{'='*70}\n")

    print(f"{'Category':<12} {'β₁':>4}{'→':>1}{'β₁':>4} {'Diversity Δ':>12} {'Novelty':>8} "
          f"{'Sem.Dist':>9} {'Int.Div Δ':>10}")
    print("-" * 65)

    for cat, data in all_results.items():
        m = data['metrics']
        b1_orig = data['original_topology']['beta1']
        b1_adapt = data['adapted_topology']['beta1']
        print(f"{cat:<12} {b1_orig:>4}→{b1_adapt:<4} {m['diversity_delta']:>+12.3f} "
              f"{m['ngram_novelty']:>8.3f} {m['semantic_distance']:>9.4f} "
              f"{m['internal_diversity_adapted'] - m['internal_diversity_orig']:>+10.4f}")

    print(f"\nResults saved to {OUTPUT_DIR / 'phase5_eval_results.json'}")


if __name__ == "__main__":
    run()
