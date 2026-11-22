"""The agent's tools (paper §E.2). Each takes a natural-language query and returns text for the memory.

We keep Base Generator, Python Coder and Wikipedia Search. Google Search (paid API key) and Web Search
(OpenAI embeddings) are dropped to keep the code small; Bamboogle answers come from Wikipedia.
"""
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

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
        "description": "Searches Wikipedia and returns the information from the most relevant pages that answers the query.",
        "input": 'query: str, e.g. tool.execute(query="Who founded Citibank and in what year?")',
        "limitation": "Wikipedia only. Use specific queries about one entity or fact at a time.",
    },
}
TOOL_NAMES = list(METADATA)
USER_AGENT = "agentflow-course-replication/0.1 (university course project)"  # Wikipedia asks API clients for one


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


def wiki_api(**params):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as r:
        return json.load(r)


def wiki_passages(query, pages=2, paragraphs=4):
    """Top search hits; from each page keep the intro plus the paragraphs sharing the most words with the query."""
    try:
        hits = wiki_api(action="query", list="search", srsearch=query, srlimit=pages)["query"]["search"]
        words = set(re.findall(r"\w+", query.lower()))
        passages = []
        for hit in hits:
            page = wiki_api(action="query", prop="extracts", explaintext=1, titles=hit["title"])["query"]["pages"]
            paras = [p for p in next(iter(page.values())).get("extract", "").split("\n") if len(p) > 40]
            best = sorted(paras[1:], key=lambda p: -len(words & set(re.findall(r"\w+", p.lower()))))[:paragraphs]
            passages.append(f"[{hit['title']}]\n" + "\n".join(paras[:1] + best)[:3000])
        return "\n\n".join(passages) or "No Wikipedia pages found."
    except Exception as e:  # network errors become an observation the planner can react to
        return f"Wikipedia error: {e}"


def wikipedia_search(queries):
    with ThreadPoolExecutor(8) as pool:
        found = list(pool.map(wiki_passages, queries))
    prompts = [f"Using only the Wikipedia excerpts below, answer the query. If they do not contain the answer, say so.\n\n"
               f"Query: {q}\n\nExcerpts:\n{text}" for q, text in zip(queries, found)]
    answers = generate(prompts, temperature=0.0, max_tokens=512)
    titles = [re.findall(r"^\[(.*)\]$", text, re.MULTILINE) for text in found]
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
