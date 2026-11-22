"""Modal entry points.
    modal run modal_app.py::eval_bamboogle --method direct --trials 3
    modal run modal_app.py::train --steps 15 --run grpo4   # trains, then evaluates rows 2 and 3 on one GPU
"""
import json
import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "peft==0.17.1", "faiss-cpu==1.10.0")  # vLLM 0.11 breaks on transformers 5.x
    .add_local_dir("data", "/root/data")
    .add_local_python_source("src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)  # keeps the 15 GB model download between runs
ckpt = modal.Volume.from_name("agentflow-team-ckpt", create_if_missing=True)  # trained planner adapters
wiki = modal.Volume.from_name("info7375-ghouse-retrieval", version=2)  # Wikipedia-2018 passages + E5 FAISS index
volumes = {"/root/.cache/huggingface": hf_cache, "/root/ckpt": ckpt, "/retrieval": wiki}
GPU_JOB = dict(gpu="H100", volumes=volumes, cpu=16, memory=112 * 1024, max_containers=1)  # 112 GiB RAM holds the 65 GB index
app = modal.App("agentflow-bamboogle", image=image)


@app.function(timeout=4 * 3600, **GPU_JOB)
def run_eval(method, trials, limit, lora):
    from src.eval import evaluate

    return evaluate(method, trials, limit, lora)


@app.function(timeout=8 * 3600, **GPU_JOB)
def run_train(steps, run):
    """On one GPU, in order: train the planner adapter, evaluate AgentFlow (base planner), evaluate the trained planner."""
    from src.eval import evaluate
    from src.flow_grpo import train

    out = f"/root/ckpt/{run}"
    os.makedirs(out, exist_ok=True)

    def log(row):  # print progress and keep it on the volume with the adapters
        print(json.dumps(row), flush=True)
        with open(f"{out}/log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        ckpt.commit()

    adapter = train(steps, out, log)
    results = evaluate("agentflow", 3, None, None), evaluate("agentflow", 3, None, adapter)
    with open(f"{out}/results.json", "w") as f:  # kept on the volume in case the local client disconnects
        json.dump(results, f)
    ckpt.commit()
    return results


def save(name, summary, records):
    print(json.dumps(summary, indent=2))
    with open(name, "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, records=records), f, indent=2, ensure_ascii=False)
    print(f"saved {name}")


@app.local_entrypoint()
def eval_bamboogle(method: str = "direct", trials: int = 3, limit: int = 0, lora: str = ""):
    summary, records = run_eval.remote(method, trials, limit or None, lora or None)
    save(f"results/{method}{'_lora' if lora else ''}{f'_n{limit}' if limit else ''}.json", summary, records)


@app.local_entrypoint()
def train(steps: int = 15, run: str = "grpo4"):
    base, trained = run_train.remote(steps, run)
    save("results/agentflow.json", *base)
    save("results/agentflow_flow_grpo.json", *trained)
