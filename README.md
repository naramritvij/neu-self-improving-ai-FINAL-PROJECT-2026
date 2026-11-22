# neu-self-improving-ai-FINAL-PROJECT-2026

INFO 7375 Final Project - AgentFlow and Flow-GRPO replication on Bamboogle using LoRA - By Ritvij Naram and Ghouse Mohiuddin Mohammed

Paper: *In-the-Flow Agentic System Optimization for Effective Planning and Tool Use* (AgentFlow, ICLR 2026).

## Results (Bamboogle, 125 questions, mean ± std over 3 trials)

| Method | Paper | Ours | Results file |
|---|---|---|---|
| Qwen-2.5-7B-Instruct | 12.0 | 14.9 ± 0.8 | `results/direct.json` |
| AgentFlow | 58.4 | 40.5 ± 2.6 | `results/agentflow.json` |
| AgentFlow w/ Flow-GRPO (LoRA) | 69.6 | 43.5 ± 2.1 | `results/agentflow_flow_grpo.json` |

Every model, module, tool and the judge is Qwen2.5-7B-Instruct. Answers are judged with the paper's prompt (§E.3) by Qwen2.5-7B-Instruct instead of GPT-4o.

What the numbers show:
- **Tools help a lot:** the same 7B model goes from 14.9 answering directly to 40.5 inside the AgentFlow loop.
- **Flow-GRPO helps:** training only the planner's LoRA adapter for 15 steps raises accuracy to 43.5 (+3.0). The trained planner gets 53 questions right in at least 2 of 3 trials, against 48 before training. During training, the reward on the training questions rose from about 48% (steps 1-5) to about 54% (steps 12-15); see `results/training_log.jsonl`.
- **The gap to the paper is mostly setup, not method:**
  - **Wikipedia:** our search tool uses the 2018 Wikipedia dump. The paper's tool uses live Wikipedia.
  - **Google Search:** the paper also has a Google Search tool, and we have none (it needs a paid API key).
  - **Training length:** we trained on 240 questions (15 steps × 16 questions × 8 rollouts). The paper trains for 5 epochs over a much larger mixed set.

## How it works

- **AgentFlow loop** (`src/agentflow.py`, paper §3.1, prompts from §E.1). For each question:
  1. The planner analyses the question.
  2. Each turn: the **planner** picks a tool and a sub-goal; the **executor** writes the tool call; the tool runs; the **verifier** decides STOP or CONTINUE.
  3. After at most 10 turns (3 during training), the **solution generator** writes the final answer from the memory of all turns.
  4. All questions move through the loop together, so each module is one batched vLLM call per turn.
- **Tools** (`src/tools.py`, §E.2):
  - **Base Generator:** answers with the LLM directly.
  - **Python Coder:** writes and runs a short program.
  - **Wikipedia Search:** retrieves the top 5 passages from the Wikipedia-2018 dump that Search-R1 uses (21M passages, E5 embeddings, FAISS), then the LLM extracts the answer from them.
  - **Why the Wikipedia dump instead of live Wikipedia:** the live API returns HTTP 429 (too many requests) to Modal's servers after about 10 requests.
- **Flow-GRPO** (`src/flow_grpo.py`, §3.2). Each step:
  1. Run 16 training questions × 8 rollouts with the current adapter.
  2. Reward each final answer 1 or 0 with the judge.
  3. Normalise rewards within each question's 8 rollouts (Eq. 7), and give every planner turn of a rollout that rollout's advantage. This is how one final reward trains every turn.
  4. Take one gradient step on the LoRA weights (rank 16) with a KL penalty to the base model (β = 0.001). Because there is one update per batch, the PPO ratio is 1 and clipping has no effect.
  - Only the planner is trained; every other module stays frozen.
- **Training data** (`scripts/prepare_train_data.py`): 2,000 NQ + HotpotQA questions from Search-R1's training split. Bamboogle is never used for training.
- **Evaluation** (`src/eval.py`, `src/judge.py`): all 125 Bamboogle questions, 3 trials, mean ± std, judged with the paper's analyse-then-judge prompt.

## Run

```bash
pip install -r requirements.txt
python -m modal setup
python -m modal run modal_app.py::eval_bamboogle --method direct --trials 3   # row 1
python -m modal run modal_app.py::train --steps 15 --run grpo4                # trains, then rows 2 and 3 on one H100
```

Everything runs on one Modal H100. The Wikipedia index lives on the Modal volume `info7375-ghouse-retrieval`, and trained adapters are saved to `agentflow-team-ckpt` after every step.

## Differences from the paper

| Paper | Ours | Why |
|---|---|---|
| Full fine-tuning of the planner | LoRA, rank 16, lr 1e-5 | Assignment requires LoRA; LoRA needs a larger learning rate than the paper's 1e-6 |
| Google, Web, Wikipedia search, Python, Base Generator | Wikipedia (2018 dump), Python, Base Generator | Google needs a paid key; Web Search needs OpenAI embeddings; live Wikipedia rate-limits Modal |
| GPT-4o judge | Qwen2.5-7B-Instruct judge, same prompt | No OpenAI key; the 7B judge is slightly lenient (see `ROADMAP.md`) |
| Batch 32 questions × 8 rollouts, 5 epochs | 16 × 8, 15 steps | Time and budget |
| Training mix of search and math (DeepMath) | NQ + HotpotQA only | Part 1 evaluates only Bamboogle |
