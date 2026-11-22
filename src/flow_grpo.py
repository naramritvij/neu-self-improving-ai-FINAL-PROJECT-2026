"""Flow-GRPO (paper §3.2) on a LoRA adapter for the planner.

Each step: take a batch of training questions, run G rollouts of AgentFlow per question with the current
adapter, score every final answer with the judge (0/1 reward), and give every planner turn of a trajectory
that trajectory's group-normalized advantage (Eq. 7). Then one gradient step on the LoRA weights:
    loss = -A * mean token log-prob of the planner output  +  beta * KL(adapter || base)
With one update per batch the PPO ratio is exactly 1, so the clipping term has no effect and is left out.
vLLM generates the rollouts; a Hugging Face copy of the model holds the trainable LoRA weights.
"""
import json
import random
import statistics
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import llm
from src.agentflow import run_agentflow
from src.judge import judge

GROUP = 8           # rollouts per question (G)
QUESTIONS = 16      # questions per step -> 128 trajectories
MAX_TURNS = 3       # planner turns per rollout during training (paper §C.1)
TEMPERATURE = 0.7   # planner sampling temperature during training
LR = 1e-5           # paper uses 1e-6 for full fine-tuning; LoRA needs a larger step
BETA = 0.001        # KL coefficient
MAX_PROMPT = 8192   # tokens; longer planner prompts are cut from the left


def load_train():
    with open(Path(__file__).parent.parent / "data" / "train.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def token_logprobs(model, tok, prompt, output):
    """Log-prob of each output token given the chat-formatted prompt."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    p_ids = tok(chat, add_special_tokens=False).input_ids[-MAX_PROMPT:]
    o_ids = tok(output, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + o_ids], device=model.device)
    logits = model(ids, logits_to_keep=len(o_ids) + 1).logits[0, :-1].float()  # predictions for the output tokens
    return torch.log_softmax(logits, -1).gather(1, ids[0, -len(o_ids):, None])[:, 0]


def train(steps, out_dir, log=print):
    llm.GPU_MEMORY = 0.4  # vLLM gets 40% of the H100; the trainable bf16 model + LoRA training use the rest
    data = load_train()
    random.Random(0).shuffle(data)
    model = tok = opt = None
    adapter = None  # step 0 rolls out the base planner
    for step in range(steps):
        start = time.time()
        batch = data[step * QUESTIONS:(step + 1) * QUESTIONS]
        questions = [row["question"] for row in batch for _ in range(GROUP)]
        truths = [row["answers"][0] + (f" (also accepted: {', '.join(row['answers'][1:])})" if row["answers"][1:] else "")
                  for row in batch for _ in range(GROUP)]
        runs = run_agentflow(questions, max_turns=MAX_TURNS, temperature=TEMPERATURE, lora=adapter)
        rewards = [float(c) for c in judge(questions, [r["answer"] for r in runs], truths)]

        samples = []  # (advantage, prompt, output) for every planner turn of every trajectory
        for g in range(0, len(runs), GROUP):
            group = rewards[g:g + GROUP]
            std = statistics.pstdev(group)
            if std == 0:  # all rollouts equally right or wrong: zero advantage, nothing to learn
                continue
            for run, r in zip(runs[g:g + GROUP], group):
                adv = (r - statistics.mean(group)) / (std + 1e-6)
                samples += [(adv, t["prompt"], t["output"]) for t in run["turns"] if t["output"]]

        if model is None:  # load after vLLM has reserved its share of the GPU
            tok = AutoTokenizer.from_pretrained(llm.MODEL)
            model = AutoModelForCausalLM.from_pretrained(llm.MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            model = get_peft_model(model, lora)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

        model.train()
        for adv, prompt, output in samples:  # one sample at a time keeps memory flat; gradients accumulate
            logp = token_logprobs(model, tok, prompt, output)
            with torch.no_grad(), model.disable_adapter():
                ref = token_logprobs(model, tok, prompt, output)
            kl = torch.exp(ref - logp) - (ref - logp) - 1  # low-variance KL estimate per token (k3)
            loss = (-adv * logp.mean() + BETA * kl.mean()) / len(samples)
            loss.backward()
        if samples:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

        adapter = f"{out_dir}/step_{step + 1}"
        model.save_pretrained(adapter)  # vLLM loads this adapter for the next step's rollouts
        log(dict(step=step + 1, reward=statistics.mean(rewards), samples=len(samples),
                 turns=statistics.mean(len(r["turns"]) for r in runs), seconds=round(time.time() - start)))
    return adapter
