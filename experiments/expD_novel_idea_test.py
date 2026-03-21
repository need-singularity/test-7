"""
Experiment D: 벽 통과 후 진짜 새로운 아이디어가 나오는가?

핵심 질문: 원본 모델에서 N번 생성해도 안 나오는 것이
벽 통과 프롬프트에서 나오는가?

방법:
  1. 원본 프롬프트로 50회 생성 (다양한 temperature)
  2. 벽 통과 프롬프트로 50회 생성
  3. 비교: 원본 집합에 없는 "새로운 개념"이 등장하는가?
  4. cosine distance로 의미적 참신성 검증
  5. 인간 평가용 HTML 리포트 생성
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent))
from common import load_model, OUTPUT_DIR, numpy_converter

N_SAMPLES = 50


def generate_batch(llm, prompt, n=N_SAMPLES, max_tokens=100):
    """다양한 temperature로 N번 생성"""
    results = []
    temperatures = np.linspace(0.01, 1.5, n)

    for i, temp in enumerate(temperatures):
        try:
            out = llm.create_completion(prompt, max_tokens=max_tokens,
                                        temperature=float(temp), top_p=0.95)
            text = out['choices'][0]['text'].strip()
            results.append({'text': text, 'temperature': float(temp)})
        except Exception as e:
            results.append({'text': f'[error: {e}]', 'temperature': float(temp)})

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n} generated")

    return results


def extract_concepts(texts):
    """텍스트들에서 핵심 개념/구절 추출 (단순 n-gram 기반)"""
    all_bigrams = Counter()
    all_trigrams = Counter()

    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - 1):
            all_bigrams[tuple(words[i:i+2])] += 1
        for i in range(len(words) - 2):
            all_trigrams[tuple(words[i:i+3])] += 1

    return all_bigrams, all_trigrams


def find_novel_concepts(original_texts, adapted_texts):
    """adapted에만 있고 original에는 없는 개념 찾기"""
    orig_bi, orig_tri = extract_concepts(original_texts)
    adapt_bi, adapt_tri = extract_concepts(adapted_texts)

    novel_trigrams = {}
    for tri, count in adapt_tri.items():
        if count >= 2 and tri not in orig_tri:
            novel_trigrams[' '.join(tri)] = count

    novel_bigrams = {}
    for bi, count in adapt_bi.items():
        if count >= 2 and bi not in orig_bi:
            novel_bigrams[' '.join(bi)] = count

    return novel_trigrams, novel_bigrams


def compute_uniqueness(texts):
    """고유한 첫 문장 비율"""
    first_sentences = set()
    for t in texts:
        first = t.split('.')[0].strip().lower()
        if first:
            first_sentences.add(first)
    return len(first_sentences) / max(len(texts), 1)


def compute_semantic_distances(llm, orig_texts, adapted_texts, n_sample=10):
    """원본 corpus와 adapted 생성물 간 cosine distance 측정.

    adapted 각 텍스트를 모델에 다시 embed하여 original corpus의
    평균 embedding과의 거리를 계산.
    """
    # 원본 corpus 평균 embedding
    orig_embs = []
    for t in orig_texts[:n_sample]:
        if t and not t.startswith('[error'):
            emb = np.array(llm.embed(t))
            orig_embs.append(emb.mean(axis=0) if emb.ndim > 1 else emb)

    if not orig_embs:
        return 0.0, []

    orig_centroid = np.mean(orig_embs, axis=0)

    # adapted 각각의 거리
    distances = []
    for t in adapted_texts[:n_sample]:
        if t and not t.startswith('[error'):
            emb = np.array(llm.embed(t))
            emb = emb.mean(axis=0) if emb.ndim > 1 else emb
            d = float(cosine(emb, orig_centroid))
            distances.append(d)

    mean_dist = float(np.mean(distances)) if distances else 0.0
    return mean_dist, distances


def wall_passage_prompts(base_prompt):
    """벽 통과 프롬프트 변형 세트"""
    return [
        base_prompt + " — considering this from a completely novel angle:",
        f"Setting aside all familiar frameworks and categories: {base_prompt}\nDescribe something that has no precedent:",
        f"In a space where the usual rules of categorization break down and concepts can combine in ways never before imagined: {base_prompt}",
        f"Transcending all existing frameworks of understanding, where even the distinction between known and unknown dissolves: {base_prompt}",
    ]


# ── 테스트 시나리오 ──────────────────────────────────

SCENARIOS = [
    {
        'name': 'creativity_color',
        'category': 'Creativity Test',
        'prompt': "A color that doesn't exist yet would look like",
        'what_to_look_for': '원본: "무지개 혼합", "보라+파랑" 반복 vs 벽 통과: 새로운 감각/개념',
    },
    {
        'name': 'knowledge_boundary',
        'category': 'Knowledge Boundary Test',
        'prompt': "The solution to the Riemann hypothesis involves",
        'what_to_look_for': '원본: "제타 함수", "비자명 영점" 교과서 반복 vs 벽 통과: 새로운 접근법',
    },
    {
        'name': 'impossible_concept',
        'category': 'Creativity Test',
        'prompt': "An emotion that no human has ever felt would be described as",
        'what_to_look_for': '원본: 기존 감정 조합 vs 벽 통과: 완전히 새로운 감정 개념',
    },
    {
        'name': 'paradigm_break',
        'category': 'Paradigm Break Test',
        'prompt': "A form of mathematics that contradicts all known axioms would work by",
        'what_to_look_for': '원본: "비유클리드", "괴델" 반복 vs 벽 통과: 새로운 수학적 구조',
    },
    {
        'name': 'consciousness_mechanism',
        'category': 'Knowledge Boundary Test',
        'prompt': "The mechanism by which consciousness emerges from neurons is",
        'what_to_look_for': '원본: "통합정보이론", "글로벌 워크스페이스" 반복 vs 벽 통과: 새로운 메커니즘',
    },
]


def judge_verdict(novel_trigrams, orig_uniqueness, adapted_uniqueness, mean_sem_dist):
    """강화된 판정 기준 (cosine distance 추가)."""
    if novel_trigrams > 50 and adapted_uniqueness > orig_uniqueness and mean_sem_dist > 0.15:
        return "YES"
    elif novel_trigrams > 50 and adapted_uniqueness > orig_uniqueness:
        return "YES-"  # trigram은 충족하나 의미 거리 부족
    elif novel_trigrams > 20:
        return "partial"
    else:
        return "no"


# ── HTML 리포트 생성 ──────────────────────────────────

def generate_html_report(all_results, output_path):
    """인간 평가용 HTML 리포트 생성."""
    html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Exp-D: Novel Idea Generation Report</title>
<style>
body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
h2 { color: #79c0ff; margin-top: 40px; }
h3 { color: #d2a8ff; }
.scenario { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; }
.verdict { font-size: 1.5em; font-weight: bold; padding: 5px 15px; border-radius: 4px; display: inline-block; }
.verdict-yes { background: #238636; color: #fff; }
.verdict-partial { background: #9e6a03; color: #fff; }
.verdict-no { background: #da3633; color: #fff; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }
.metric { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; text-align: center; }
.metric-value { font-size: 1.8em; font-weight: bold; color: #58a6ff; }
.metric-label { color: #8b949e; font-size: 0.85em; }
.comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0; }
.column { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 15px; }
.column h4 { margin-top: 0; }
.orig-col h4 { color: #8b949e; }
.adapt-col h4 { color: #58a6ff; }
.sample { background: #161b22; border-left: 3px solid #30363d; padding: 10px; margin: 8px 0; font-size: 0.9em; line-height: 1.5; }
.adapt-col .sample { border-left-color: #58a6ff; }
.novel-concepts { background: #0d1117; border: 1px solid #238636; border-radius: 6px; padding: 15px; margin: 15px 0; }
.novel-tag { display: inline-block; background: #238636; color: #fff; padding: 2px 8px; border-radius: 12px; margin: 3px; font-size: 0.85em; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; }
th, td { padding: 10px 15px; border: 1px solid #30363d; text-align: center; }
th { background: #161b22; color: #58a6ff; }
.temp-label { color: #8b949e; font-size: 0.8em; }
</style>
</head>
<body>
<h1>Exp-D: Novel Idea Generation Test</h1>
<p>벽 통과 프롬프트가 원본 모델이 절대 생성하지 못하는 새로운 개념을 만들어내는가?</p>

<h2>Summary</h2>
<table>
<tr><th>Scenario</th><th>Category</th><th>Orig Uniqueness</th><th>Adapt Uniqueness</th><th>Novel Trigrams</th><th>Semantic Dist</th><th>Verdict</th></tr>
"""

    for name, data in all_results.items():
        verdict = data.get('verdict', 'no')
        v_class = 'verdict-yes' if 'YES' in verdict else ('verdict-partial' if verdict == 'partial' else 'verdict-no')
        html += f"""<tr>
<td>{name}</td><td>{data.get('category', '')}</td>
<td>{data['orig_uniqueness']:.3f}</td><td>{data['adapted_uniqueness']:.3f}</td>
<td>{data['n_novel_trigrams']}</td><td>{data.get('mean_semantic_distance', 0):.3f}</td>
<td><span class="verdict {v_class}">{verdict}</span></td>
</tr>"""

    html += "</table>"

    # 각 시나리오 상세
    for name, data in all_results.items():
        verdict = data.get('verdict', 'no')
        v_class = 'verdict-yes' if 'YES' in verdict else ('verdict-partial' if verdict == 'partial' else 'verdict-no')

        html += f"""
<div class="scenario">
<h2>{name}</h2>
<p><strong>Prompt:</strong> "{data['prompt']}"</p>
<p><strong>Adapted:</strong> "{data['adapted_prompt'][:100]}..."</p>
<span class="verdict {v_class}">{verdict}</span>

<div class="metrics">
<div class="metric"><div class="metric-value">{data['n_novel_trigrams']}</div><div class="metric-label">Novel Trigrams</div></div>
<div class="metric"><div class="metric-value">{data['orig_uniqueness']:.2f}</div><div class="metric-label">Orig Uniqueness</div></div>
<div class="metric"><div class="metric-value">{data['adapted_uniqueness']:.2f}</div><div class="metric-label">Adapt Uniqueness</div></div>
<div class="metric"><div class="metric-value">{data.get('mean_semantic_distance', 0):.3f}</div><div class="metric-label">Semantic Distance</div></div>
</div>
"""

        # Novel concepts
        top_novel = data.get('top_novel_trigrams', {})
        if top_novel:
            html += '<div class="novel-concepts"><h3>Novel Concepts (adapted only)</h3>'
            for phrase, count in sorted(top_novel.items(), key=lambda x: x[1], reverse=True)[:15]:
                html += f'<span class="novel-tag">{phrase} (x{count})</span>'
            html += '</div>'

        # Sample comparison
        orig_completions = data.get('original_completions', [])
        adapted_completions = data.get('adapted_completions', [])

        html += '<div class="comparison">'
        html += '<div class="column orig-col"><h4>Original Outputs</h4>'
        for i, r in enumerate(orig_completions[:8]):
            text = r['text'][:200] if isinstance(r, dict) else str(r)[:200]
            temp = r.get('temperature', '?') if isinstance(r, dict) else '?'
            html += f'<div class="sample"><span class="temp-label">t={temp:.2f}</span><br>{text}</div>'
        html += '</div>'

        html += '<div class="column adapt-col"><h4>Adapted Outputs</h4>'
        for i, r in enumerate(adapted_completions[:8]):
            text = r['text'][:200] if isinstance(r, dict) else str(r)[:200]
            temp = r.get('temperature', '?') if isinstance(r, dict) else '?'
            html += f'<div class="sample"><span class="temp-label">t={temp:.2f}</span><br>{text}</div>'
        html += '</div></div></div>'

    html += """
<footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #30363d; color: #8b949e;">
<p>Generated by Exp-D Novel Idea Test | Topological Wall Passage for LLMs</p>
</footer>
</body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nHTML report: {output_path}")


# ── Main ──────────────────────────────────────────────

def run():
    llm = load_model()
    all_results = {}

    for scenario in SCENARIOS:
        name = scenario['name']
        prompt = scenario['prompt']

        print(f"\n{'='*70}")
        print(f"[{name}] \"{prompt}\"")
        print(f"Looking for: {scenario['what_to_look_for']}")
        print(f"{'='*70}")

        # 1. 원본 50회 생성
        print(f"\n  === ORIGINAL (x{N_SAMPLES}) ===")
        t0 = time.time()
        orig_results = generate_batch(llm, prompt)
        t_orig = time.time() - t0
        orig_texts = [r['text'] for r in orig_results if not r['text'].startswith('[error')]

        # 2. 벽 통과 프롬프트 (Level 3 사용)
        adapted_prompt = wall_passage_prompts(prompt)[2]
        print(f"\n  === ADAPTED (x{N_SAMPLES}) ===")
        print(f"  Prompt: \"{adapted_prompt[:70]}...\"")
        t0 = time.time()
        adapted_results = generate_batch(llm, adapted_prompt)
        t_adapted = time.time() - t0
        adapted_texts = [r['text'] for r in adapted_results if not r['text'].startswith('[error')]

        # 3. 분석
        print(f"\n  === ANALYSIS ===")

        orig_uniqueness = compute_uniqueness(orig_texts)
        adapted_uniqueness = compute_uniqueness(adapted_texts)
        print(f"  Uniqueness (first sentence): orig={orig_uniqueness:.3f}, adapted={adapted_uniqueness:.3f}")

        novel_tri, novel_bi = find_novel_concepts(orig_texts, adapted_texts)
        print(f"  Novel trigrams (in adapted, not in original): {len(novel_tri)}")
        if novel_tri:
            top_novel = sorted(novel_tri.items(), key=lambda x: x[1], reverse=True)[:10]
            for phrase, count in top_novel:
                print(f"    \"{phrase}\" (x{count})")

        # 4. Semantic distance (cosine)
        print(f"\n  === SEMANTIC DISTANCE ===")
        mean_sem_dist, sem_dists = compute_semantic_distances(llm, orig_texts, adapted_texts)
        print(f"  Mean cosine distance (adapted vs orig centroid): {mean_sem_dist:.4f}")

        # 원본 패턴 분석
        orig_starts = Counter()
        for t in orig_texts:
            start = ' '.join(t.split()[:5]).lower()
            orig_starts[start] += 1

        adapted_starts = Counter()
        for t in adapted_texts:
            start = ' '.join(t.split()[:5]).lower()
            adapted_starts[start] += 1

        print(f"\n  Top original starts:")
        for start, count in orig_starts.most_common(5):
            print(f"    \"{start}\" (x{count})")

        print(f"  Top adapted starts:")
        for start, count in adapted_starts.most_common(5):
            print(f"    \"{start}\" (x{count})")

        # 샘플 비교
        print(f"\n  -- Sample Comparison --")
        if orig_texts:
            print(f"  [ORIGINAL best (t=0.01)]:")
            print(f"    \"{orig_texts[0][:150]}\"")
            mid_idx = len(orig_texts) // 2
            print(f"  [ORIGINAL mid (t=0.75)]:")
            print(f"    \"{orig_texts[mid_idx][:150]}\"")
            print(f"  [ORIGINAL wild (t=1.5)]:")
            print(f"    \"{orig_texts[-1][:150]}\"")

        if adapted_texts:
            print(f"\n  [ADAPTED best (t=0.01)]:")
            print(f"    \"{adapted_texts[0][:150]}\"")
            mid_idx = len(adapted_texts) // 2
            print(f"  [ADAPTED mid (t=0.75)]:")
            print(f"    \"{adapted_texts[mid_idx][:150]}\"")
            print(f"  [ADAPTED wild (t=1.5)]:")
            print(f"    \"{adapted_texts[-1][:150]}\"")

        # 판정
        verdict = judge_verdict(len(novel_tri), orig_uniqueness, adapted_uniqueness, mean_sem_dist)
        print(f"\n  ★ VERDICT: {verdict}")

        all_results[name] = {
            'prompt': prompt,
            'adapted_prompt': adapted_prompt,
            'category': scenario['category'],
            'n_samples': N_SAMPLES,
            'orig_uniqueness': orig_uniqueness,
            'adapted_uniqueness': adapted_uniqueness,
            'n_novel_trigrams': len(novel_tri),
            'n_novel_bigrams': len(novel_bi),
            'mean_semantic_distance': mean_sem_dist,
            'semantic_distances': sem_dists,
            'top_novel_trigrams': dict(sorted(novel_tri.items(), key=lambda x: x[1], reverse=True)[:20]),
            'original_completions': orig_results,
            'adapted_completions': adapted_results,
            'time_original': t_orig,
            'time_adapted': t_adapted,
            'verdict': verdict,
        }

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "expD_novel_ideas.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=numpy_converter)

    # HTML 리포트 생성
    generate_html_report(all_results, OUTPUT_DIR / "expD_report.html")

    # 최종 요약
    print(f"\n\n{'='*70}")
    print("EXPERIMENT D: NOVEL IDEA GENERATION SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Scenario':<25} {'Orig Uniq':>10} {'Adapt Uniq':>11} {'Novel 3g':>9} {'Sem.Dist':>9} {'Verdict':>10}")
    print("-" * 80)

    for name, data in all_results.items():
        print(f"{name:<25} {data['orig_uniqueness']:>10.3f} {data['adapted_uniqueness']:>11.3f} "
              f"{data['n_novel_trigrams']:>9} {data['mean_semantic_distance']:>9.3f} {data['verdict']:>10}")

    print(f"\nFull results: {OUTPUT_DIR / 'expD_novel_ideas.json'}")
    print(f"HTML report:  {OUTPUT_DIR / 'expD_report.html'}")


if __name__ == "__main__":
    run()
