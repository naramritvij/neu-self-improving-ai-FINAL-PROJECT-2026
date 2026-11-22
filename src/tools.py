"""The agent's tools (paper §E.2). Each takes a natural-language query and returns text for the memory.

We keep Base Generator, Python Coder and Wikipedia Search. Google Search (paid API key) and Web Search
(OpenAI embeddings) are dropped to keep the code small; Bamboogle answers come from Wikipedia.

Wikipedia Search runs inside our container over the Wikipedia-2018 dump that Search-R1 uses (21M passages,
E5 embeddings, FAISS index), because the live Wikipedia API rate-limits Modal's servers (HTTP 429).
"""
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

from src.llm import generate

METADATA = {  # what the planner and executor see about each tool
    "Generalist_Solution_Generator_Tool": {
        "description": "Answers a query directly with the language model, step by step.",
        "input": 'query: str, e.g. tool.execute(query="Summarize what is known about X")',
        "limitation": "No access to external facts; may be wrong or out of date on specific facts.",
    },
    "Python_Code_Generator_Tool": {
        "description": "Writes and runs a short Python program for arithmetic, dates or logic, and returns its printed output.",
        "input": 'query: str, e.g. tool.execute(query="How many years are between 1789 and 1837?")',
        "limitation": "Standard library only, no internet, 10-second limit. Put every needed number in the query.",
    },
    "Wikipedia_RAG_Search_Tool": {
        "description": "Searches Wikipedia and returns the information from the most relevant passages that answers the query.",
        "input": 'query: str, e.g. tool.execute(query="Who founded Citibank and in what year?")',
        "limitation": "Wikipedia only (2018 snapshot). Use specific queries about one entity or fact at a time.",
    },
}
TOOL_NAMES = list(METADATA)
WIKI = "/retrieval/wiki18-e5"  # Modal volume: e5_Flat.index, wiki-18.jsonl (+ byte offsets), e5-base-v2 encoder
_wiki = None


def base_generator(queries):
    return generate(queries, temperature=0.0, max_tokens=1024)


def run_python(code):
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        return (done.stdout + done.stderr)[-1500:] or "(no output)"
    except subprocess.TimeoutExpired:
        return "Timed out after 10 seconds."


def python_coder(queries):
    prompts = [f"Write a short, self-contained Python program that solves the task below and prints the answer.\n"
               f"Task: {q}\nReturn only the code in one ```python block." for q in queries]
    codes = [m.group(1) if (m := re.search(r"```python\n(.*?)```", reply, re.DOTALL)) else reply
             for reply in generate(prompts, temperature=0.0, max_tokens=1024)]
    with ThreadPoolExecutor(8) as pool:
        outputs = list(pool.map(run_python, codes))
    return [f"Code:\n{code}\nOutput:\n{out}" for code, out in zip(codes, outputs)]


def wiki_passages(queries, k=5):
    """Top-k Wikipedia passages per query: E5 query embedding -> inner-product search over all passages."""
    global _wiki
    if _wiki is None:  # load once: the index is 65 GB in RAM
        import faiss
        import numpy as np
        from transformers import AutoModel, AutoTokenizer
        _wiki = (faiss.read_index(f"{WIKI}/e5_Flat.index"),
                 np.memmap(f"{WIKI}/wiki-18.jsonl.offsets", dtype="<u8", mode="r"),  # byte offset of each passage
                 open(f"{WIKI}/wiki-18.jsonl", "rb"),
                 AutoTokenizer.from_pretrained(f"{WIKI}/e5-base-v2"),
                 AutoModel.from_pretrained(f"{WIKI}/e5-base-v2").cuda().eval())
    index, offsets, corpus, tok, encoder = _wiki
    with torch.no_grad():  # E5: "query: " prefix, mean-pooled, normalized
        x = tok(["query: " + q for q in queries], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        h = encoder(**x).last_hidden_state.masked_fill(~x.attention_mask[..., None].bool(), 0)
        emb = torch.nn.functional.normalize(h.sum(1) / x.attention_mask.sum(1)[:, None], dim=-1)
    _, ids = index.search(emb.cpu().numpy(), k)

    def passage(i):
        corpus.seek(int(offsets[i]))
        return json.loads(corpus.readline())["contents"]
    return ["\n\n".join(passage(i) for i in row if i >= 0) for row in ids]


def wikipedia_search(queries):
    found = wiki_passages(queries)
    prompts = [f"Using only the Wikipedia passages below, answer the query. If they do not contain the answer, say so.\n\n"
               f"Query: {q}\n\nPassages:\n{text}" for q, text in zip(queries, found)]
    answers = generate(prompts, temperature=0.0, max_tokens=512)
    titles = [re.findall(r'^"(.*?)"$', text, re.MULTILINE) for text in found]  # each passage starts with "Title"
    return [f"Pages: {t}\nRetrieved information: {a}" for t, a in zip(titles, answers)]


TOOLS = {"Generalist_Solution_Generator_Tool": base_generator, "Python_Code_Generator_Tool": python_coder,
         "Wikipedia_RAG_Search_Tool": wikipedia_search}


def run_tools(calls):
    """calls: list of (tool_name, query). Calls to the same tool run as one batch. Returns one result string each."""
    results = [f"Tool not found. Choose one of {TOOL_NAMES}."] * len(calls)
    for name, tool in TOOLS.items():
        idx = [i for i, (t, _) in enumerate(calls) if t == name]
        if idx:
            for i, result in zip(idx, tool([calls[i][1] for i in idx])):
                results[i] = result
    return results
