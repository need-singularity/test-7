"""
E7: LoRA Fine-tuning with Topology Loss (GPU version)
=====================================================
Windows PC (RTX 5070, 12GB VRAM) 에서 실행.
4-bit quantization으로 VRAM 12GB 안에 맞춤.
"""

import numpy as np
import time
import json
import torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from ripser import ripser
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings("ignore")

MODEL_ID = "C:/Users/aiden/models/llama-3.1-8b-instruct"
OUTPUT_PATH = Path("e7_lora_gpu_results.json")

CORE_WALL_DIMS = [782, 977, 1917, 1971, 2720, 2943, 3139, 4080]
PCA_DIM = 50
PERSISTENCE_FLOOR = 0.05

# Training config
LORA_RANK = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "v_proj"]
LAMBDA_TOPO = 0.01
N_EPOCHS = 3
LR = 1e-4
MAX_GRAD_NORM = 1.0

TRAIN_PROMPTS = [
    "A color that doesn't exist yet would look like",
    "If mathematics were a living organism, its heartbeat would be",
    "The solution to the Riemann hypothesis involves",
    "Describe the sound of silence in a language that has no words.",
    "Invent a concept that no human has ever thought of.",
    "What would a fifth fundamental force of nature do?",
    "Describe an emotion that doesn't exist yet.",
    "If time had a shape, what would it look like?",
]

EVAL_PROMPTS = [
    {"text": "What is the capital of France?", "answer": "Paris", "category": "factual"},
    {"text": "Who wrote Romeo and Juliet?", "answer": "Shakespeare", "category": "factual"},
    {"text": "What is the chemical symbol for water?", "answer": "H2O", "category": "factual"},
    {"text": "Describe a color that doesn't exist in nature.", "answer": None, "category": "creative"},
    {"text": "What is the last digit of pi?", "answer": None, "category": "boundary"},
]


def compute_ph(points, floor=PERSISTENCE_FLOOR):
    n, d = points.shape
    dim = min(PCA_DIM, n - 1, d)
    if dim >= 2:
        reduced = PCA(n_components=dim).fit_transform(points)
    else:
        reduced = points
    dm = squareform(pdist(reduced))
    result = ripser(dm, maxdim=1, distance_matrix=True)
    dgm1 = result['dgms'][1]
    if len(dgm1) == 0:
        return 0, 0.0, 0.0
    pers = dgm1[:, 1] - dgm1[:, 0]
    sig = pers[pers > floor]
    return len(sig), float(sig.max()) if len(sig) > 0 else 0.0, float(sig.sum())


def topology_loss(hidden_states, wall_dims):
    wall_activations = hidden_states[:, :, wall_dims]
    center = wall_activations.mean(dim=1, keepdim=True).detach()
    distances = torch.norm(wall_activations - center, dim=-1)
    return distances.mean()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    print("\nLoading model (4-bit quantized)...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Pre-training evaluation
    print("\n── Pre-training evaluation")
    model.eval()

    pre_results = []
    for ep in EVAL_PROMPTS:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": ep["text"]},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        response = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        acc = ep["answer"].lower() in response.lower() if ep["answer"] else None
        pre_results.append({"category": ep["category"], "prompt": ep["text"], "response": response, "accuracy": acc})
        acc_str = "✓" if acc else ("✗" if acc is not None else "—")
        print(f"  {acc_str} [{ep['category']}] {response[:80]}...")

    pre_factual = sum(1 for r in pre_results if r["accuracy"] is True)
    pre_total = sum(1 for r in pre_results if r["accuracy"] is not None)
    print(f"  Pre-training accuracy: {pre_factual}/{pre_total}")

    # ── Measure pre-training β₁
    print("\n── Pre-training β₁")
    pre_b1s = []
    for prompt_text in TRAIN_PROMPTS[:3]:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[-1][0].float().cpu().numpy()
        b1, mp, tp = compute_ph(hs)
        pre_b1s.append({"prompt": prompt_text[:40], "beta1": b1, "max_pers": mp})
        print(f"  β₁={b1}, mp={mp:.4f} — {prompt_text[:40]}...")

    # ── Setup LoRA
    print("\n── Setting up LoRA")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=list(range(28, 32)),
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.3f}%)")

    # ── Training loop
    print(f"\n── Training: {N_EPOCHS} epochs, λ_topo={LAMBDA_TOPO}, lr={LR}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )

    model.train()
    training_log = []

    for epoch in range(N_EPOCHS):
        epoch_losses = []
        for i, prompt_text in enumerate(TRAIN_PROMPTS):
            messages = [
                {"role": "system", "content": "You are a creative and knowledgeable assistant."},
                {"role": "user", "content": prompt_text},
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            outputs = model(**inputs, output_hidden_states=True, labels=inputs["input_ids"])

            l_ce = outputs.loss
            last_hidden = outputs.hidden_states[-1]
            l_topo = topology_loss(last_hidden, CORE_WALL_DIMS)
            l_total = l_ce + LAMBDA_TOPO * l_topo

            optimizer.zero_grad()
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            epoch_losses.append({
                "l_total": l_total.item(),
                "l_ce": l_ce.item(),
                "l_topo": l_topo.item(),
            })

        avg_ce = np.mean([l["l_ce"] for l in epoch_losses])
        avg_topo = np.mean([l["l_topo"] for l in epoch_losses])
        avg_total = np.mean([l["l_total"] for l in epoch_losses])
        print(f"  Epoch {epoch+1}/{N_EPOCHS}: L_total={avg_total:.4f} (CE={avg_ce:.4f} + λ·topo={LAMBDA_TOPO*avg_topo:.4f})")
        training_log.append({"epoch": epoch+1, "l_ce": avg_ce, "l_topo": avg_topo, "l_total": avg_total})

        if device == "cuda":
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"    VRAM peak: {mem:.1f}GB")

    # ── Post-training evaluation
    print("\n── Post-training evaluation")
    model.eval()

    post_results = []
    for ep in EVAL_PROMPTS:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": ep["text"]},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        response = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        acc = ep["answer"].lower() in response.lower() if ep["answer"] else None
        post_results.append({"category": ep["category"], "prompt": ep["text"], "response": response, "accuracy": acc})
        acc_str = "✓" if acc else ("✗" if acc is not None else "—")
        print(f"  {acc_str} [{ep['category']}] {response[:80]}...")

    post_factual = sum(1 for r in post_results if r["accuracy"] is True)
    post_total = sum(1 for r in post_results if r["accuracy"] is not None)
    print(f"  Post-training accuracy: {post_factual}/{post_total}")

    # ── Post-training β₁
    print("\n── Post-training β₁")
    post_b1s = []
    for prompt_text in TRAIN_PROMPTS[:3]:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[-1][0].float().cpu().numpy()
        b1, mp, tp = compute_ph(hs)
        post_b1s.append({"prompt": prompt_text[:40], "beta1": b1, "max_pers": mp})
        print(f"  β₁={b1}, mp={mp:.4f} — {prompt_text[:40]}...")

    # ── Summary
    print("\n\n" + "=" * 70)
    print("E7 SUMMARY — LoRA Fine-tuning with Topology Loss (GPU)")
    print("=" * 70)

    print(f"\n  Training: {N_EPOCHS} epochs, {len(TRAIN_PROMPTS)} prompts")
    print(f"  LoRA: rank={LORA_RANK}, layers 28-31, trainable={trainable:,} ({trainable/total*100:.3f}%)")
    print(f"  λ_topo={LAMBDA_TOPO}, lr={LR}, grad_clip={MAX_GRAD_NORM}")

    print(f"\n  Loss trajectory:")
    for log in training_log:
        print(f"    Epoch {log['epoch']}: CE={log['l_ce']:.4f}, topo={log['l_topo']:.4f}")

    print(f"\n  Factual accuracy: {pre_factual}/{pre_total} → {post_factual}/{post_total}")

    print(f"\n  β₁ comparison:")
    print(f"  {'Prompt':>40}  {'Pre β₁':>7}  {'Post β₁':>8}  {'Δ':>4}")
    print(f"  {'-'*62}")
    for pre, post in zip(pre_b1s, post_b1s):
        delta = post['beta1'] - pre['beta1']
        print(f"  {pre['prompt']:>40}  {pre['beta1']:>7}  {post['beta1']:>8}  {delta:>+4}")

    print(f"\n  Generation comparison (creative):")
    pre_creative = [r for r in pre_results if r["category"] == "creative"]
    post_creative = [r for r in post_results if r["category"] == "creative"]
    if pre_creative and post_creative:
        print(f"    Pre:  {pre_creative[0]['response'][:100]}...")
        print(f"    Post: {post_creative[0]['response'][:100]}...")

    # Save
    results = {
        "config": {
            "device": device,
            "quantization": "4bit-nf4",
            "lora_rank": LORA_RANK,
            "lambda_topo": LAMBDA_TOPO,
            "lr": LR,
            "n_epochs": N_EPOCHS,
            "target_layers": list(range(28, 32)),
            "wall_dims": CORE_WALL_DIMS,
            "trainable_params": trainable,
        },
        "training_log": training_log,
        "pre_eval": pre_results,
        "post_eval": post_results,
        "pre_beta1": pre_b1s,
        "post_beta1": post_b1s,
    }

    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=convert, ensure_ascii=False)
    print(f"\n  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
