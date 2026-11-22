"""Modal entry points.
    modal run modal_app.py::eval_bamboogle --method direct --trials 3
    modal run modal_app.py::train --steps 15 --run grpo1
    modal run modal_app.py::eval_bamboogle --method agentflow --lora /root/ckpt/grpo1/step_15
"""
import json
import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "peft==0.17.1")  # vLLM 0.11 breaks on transformers 5.x
    .add_local_dir("data", "/root/data")
    .add_local_python_source("src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)  # keeps the 15 GB model download between runs
ckpt = modal.Volume.from_name("agentflow-team-ckpt", create_if_missing=True)  # trained planner adapters
volumes = {"/root/.cache/huggingface": hf_cache, "/root/ckpt": ckpt}
app = modal.App("agentflow-bamboogle", image=image)


@app.function(gpu="H100", volumes=volumes, timeout=4 * 3600, max_containers=1)
def run_eval(method, trials, limit, lora):
    from src.eval import evaluate

    return evaluate(method, trials, limit, lora)


@app.function(gpu="H100", volumes=volumes, timeout=8 * 3600, max_containers=1)
def run_train(steps, run):
    from src.flow_grpo import train

    out = f"/root/ckpt/{run}"
    os.makedirs(out, exist_ok=True)

    def log(row):  # print progress and keep it on the volume with the adapters
        print(json.dumps(row), flush=True)
        with open(f"{out}/log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        ckpt.commit()

    return train(steps, out, log)


@app.local_entrypoint()
def eval_bamboogle(method: str = "direct", trials: int = 3, limit: int = 0, lora: str = ""):
    summary, records = run_eval.remote(method, trials, limit or None, lora or None)
    print(json.dumps(summary, indent=2))
    name = f"results/{method}{'_lora' if lora else ''}{f'_n{limit}' if limit else ''}.json"
    with open(name, "w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, records=records), f, indent=2, ensure_ascii=False)
    print(f"saved {name}")


@app.local_entrypoint()
def train(steps: int = 15, run: str = "grpo1"):
    print("final adapter:", run_train.remote(steps, run))
