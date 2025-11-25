"""
Lightweight hyperparameter sweep for MEMIT_ARE on UnKE-style evaluation.

For each hyperparameter combination we:
1) write a temporary hparams JSON under hparams/MEMIT_ARE/.
2) run `experiments.evaluate_uns` with that file.
3) copy the produced output JSON to a unique name.
4) score it with ROUGE/BLEU/BERTScore (similar to summarize_uns).

Usage example:
python -m experiments.sweep_uns \
  --base_hparams Gemma3-1B-it.json \
  --model_name google/gemma-3-1b-it \
  --ds_name unke \
  --dataset_size_limit 50 \
  --device 0
"""

import argparse
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from itertools import product

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


def load_base_hparams(base_path: Path) -> dict:
    with base_path.open() as f:
        return json.load(f)


def write_tmp_hparams(base_hparams: dict, sweep_cfg: dict, fname: Path) -> Path:
    hp = dict(base_hparams)
    hp.update(sweep_cfg)
    fname.parent.mkdir(parents=True, exist_ok=True)
    with fname.open("w") as f:
        json.dump(hp, f, indent=4)
    return fname


def run_edit(alg_name: str, model_name: str, hparams_fname: str, ds_name: str, dataset_size_limit: int, num_edits: int):
    cmd = [
        "python",
        "-m",
        "experiments.evaluate_uns",
        f"--alg_name={alg_name}",
        f"--model_name={model_name}",
        f"--hparams_fname={hparams_fname}",
        f"--ds_name={ds_name}",
        f"--dataset_size_limit={dataset_size_limit}",
        f"--num_edits={num_edits}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def score_file(path: Path, device: int, bert_model: str):
    with path.open() as f:
        data = json.load(f)

    rouge = Rouge()
    bleu_scores = []
    rouge_ls = []
    answers = []
    preds = []
    for row in data:
        if row.get("original_prediction", "") == "":
            continue
        bleu_scores.append(sentence_bleu([row["answer"]], row["original_prediction"]))
        rouge_ls.append(rouge.get_scores(row["original_prediction"], row["answer"])[0]["rouge-l"]["r"])
        answers.append(row["answer"])
        preds.append(row["original_prediction"])

    metrics = {
        "BLEU": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        "ROUGE-L": sum(rouge_ls) / len(rouge_ls) if rouge_ls else 0.0,
    }

    if answers and preds:
        model = SentenceTransformer(bert_model, device=f"cuda:{device}")
        emb_a = model.encode(answers, convert_to_tensor=True, show_progress_bar=False)
        emb_p = model.encode(preds, convert_to_tensor=True, show_progress_bar=False)
        cos = util.cos_sim(emb_a, emb_p).diagonal().mean().item()
        metrics["BERTScore"] = cos
    else:
        metrics["BERTScore"] = 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_hparams", default="Gemma3-1B-it.json")
    parser.add_argument("--alg_name", default="MEMIT_ARE")
    parser.add_argument("--model_name", default="google/gemma-3-1b-it")
    parser.add_argument("--ds_name", default="unke")
    parser.add_argument("--dataset_size_limit", type=int, default=50)
    parser.add_argument("--num_edits", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    base_path = ROOT / "hparams" / args.alg_name / args.base_hparams
    base_hp = load_base_hparams(base_path)

    sweep_space = {
        "layers": [[6], [7], [6, 7]],
        "clamp_norm_factor": [1.5, 2.0],
        "v_lr": [0.05, 0.1],
        "v_num_grad_steps": [6, 10],
        "kl_factor": [0.2, 0.5],
        "window_size": [10, 20],
        "overlap": [5],
    }

    combos = list(product(*sweep_space.values()))
    key_order = list(sweep_space.keys())
    results = []

    tmp_dir = ROOT / "hparams" / args.alg_name / "sweeps"
    model_stub = args.model_name.replace("/", "_")
    out_dir = ROOT / "output" / "sweeps" / model_stub
    out_dir.mkdir(parents=True, exist_ok=True)

    for combo in tqdm(combos, desc="Sweep configs"):
        cfg = dict(zip(key_order, combo))
        run_id = uuid.uuid4().hex[:8]
        tmp_fname = tmp_dir / f"sweep_{run_id}.json"
        write_tmp_hparams(base_hp, cfg, tmp_fname)

        # Run edit/eval (pass path relative to hparams/<alg_name>)
        rel_hp = tmp_fname.relative_to(ROOT / "hparams" / args.alg_name)
        run_edit(args.alg_name, args.model_name, str(rel_hp), args.ds_name, args.dataset_size_limit, args.num_edits)

        # Move output to unique name
        out_path = ROOT / f"output/{args.alg_name}_{base_hp['model_name']}_{args.ds_name}_result.json"
        dest = out_dir / f"{run_id}.json"
        shutil.copy(out_path, dest)

        metrics = score_file(dest, args.device, args.bert_model)
        results.append((metrics, cfg, dest.name))
        print(f"Run {run_id}: {metrics} cfg={cfg}")

    # Sort by BERTScore then ROUGE-L
    results.sort(key=lambda x: (x[0].get("BERTScore", 0), x[0].get("ROUGE-L", 0)), reverse=True)
    print("\n=== Top runs ===")
    for m, cfg, fname in results[:5]:
        print(f"{fname} | BERTScore={m.get('BERTScore', 0):.4f} ROUGE-L={m.get('ROUGE-L', 0):.4f} BLEU={m.get('BLEU', 0):.4f} | cfg={cfg}")


if __name__ == "__main__":
    main()
