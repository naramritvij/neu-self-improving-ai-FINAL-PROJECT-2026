"""AgentFlow (paper §3.1): planner -> executor -> tool -> verifier each turn, then the solution generator.

Prompts follow paper §E.1 (the official repo's non-multimodal prompts). All questions move through the
loop together, so each module is one batched vLLM call per turn. Only the planner's next-step call uses
`temperature` and `lora` (the trained adapter); every other module is the frozen base model at temperature 0.
"""
import re

from src.llm import generate
from src.tools import METADATA, TOOL_NAMES, run_tools

ANALYSIS = """Task: Analyze the given query to determine necessary skills and tools.

Inputs:
- Query: {question}
- Available tools: {tools}
- Metadata for tools: {metadata}

Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.

Format your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.

Be brief and precise with insight."""

NEXT_STEP = """Task: Determine the optimal next step to address the query using available tools and previous steps.

Context:
- **Query:** {question}
- **Query Analysis:** {analysis}
- **Available Tools:** {tools}
- **Toolbox Metadata:** {metadata}
- **Previous Steps:** {memory}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format:
1.  **Justification:** Explain your choice of tool and sub-goal.
2.  **Context:** Provide all necessary information for the tool.
3.  **Sub-Goal:** State the specific objective for the tool.
4.  **Tool Name:** State the exact name of the selected tool.

Rules:
- Select only ONE tool.
- The sub-goal must be directly achievable by the selected tool.
- The Context section must contain all information the tool needs to function.
- The response must end with the Context, Sub-Goal, and Tool Name sections in that order, with no extra content."""

COMMAND = """Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {question}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool}
- **Tool Metadata:** {metadata}
- **Relevant Data:** {context}

Instructions:
1.  Analyze the tool's required parameters from its metadata.
2.  Construct valid Python code that addresses the sub-goal using the provided context and data.
3.  The command must include at least one call to `tool.execute()`.
4.  Each `tool.execute()` call must be assigned to a variable named **`execution`**.
5.  Please give the exact numbers and parameters should be used in the `tool.execute()` call.

Output Format:
Present your response in the following structured format. Do not include any extra text or explanations.

Generated Command:
```python
<command>
```

Example:
Generated Command:
```python
execution = tool.execute(query="When was the first moon landing?")
```"""

VERIFY = """Task: Evaluate if the current memory is complete and accurate enough to answer the query, or if more tools are needed.

Context:
- **Query:** {question}
- **Available Tools:** {tools}
- **Toolbox Metadata:** {metadata}
- **Initial Analysis:** {analysis}
- **Memory (Tools Used & Results):** {memory}

Instructions:
1.  Review the query, initial analysis, and memory.
2.  Assess the completeness of the memory: Does it fully address all parts of the query?
3.  Check for potential issues:
    -   Are there any inconsistencies or contradictions?
    -   Is any information ambiguous or in need of verification?
4.  Determine if any unused tools could provide missing information.

Final Determination:
-   If the memory is sufficient, explain why and conclude with "STOP".
-   If more information is needed, explain what's missing, which tools could help, and conclude with "CONTINUE".

IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE"."""

SOLUTION = """Task: Generate a concise final answer to the query based on all provided context.

Context:
- **Query:** {question}
- **Initial Analysis:** {analysis}
- **Actions Taken:** {memory}

Instructions:
1. Review the query and the results from all actions.
2. Synthesize the key findings into a clear, step-by-step summary of the process.
3. Provide a direct, precise answer to the original query.

Output Structure:
1.  **Process Summary:** A clear, step-by-step breakdown of how the query was addressed, including the purpose and key results of each action.
2.  **Answer:** A direct and concise final answer to the query."""


def parse_plan(text):
    """Return (context, sub_goal, tool_name) from the planner's last Context / Sub-Goal / Tool Name block."""
    text = text.replace("**", "")
    found = re.findall(r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*([\w ]+)", text, re.DOTALL)
    if not found:
        return "", "", None
    context, sub_goal, tool = (s.strip() for s in found[-1])
    canonical = {t.lower(): t for t in TOOL_NAMES}  # accept "wikipedia rag search tool" etc.
    return context, sub_goal, canonical.get(re.sub(r"[ _]+", "_", tool).lower())


def parse_query(command, fallback):
    """The executor writes `execution = tool.execute(query="...")`; we take the query string instead of exec-ing it."""
    m = re.search(r"query\s*=\s*(\"\"\"|\"|')(.*?)\1", command, re.DOTALL)
    return m.group(2) if m else fallback


def run_agentflow(questions, max_turns=10, temperature=0.7, lora=None):
    tools = dict(tools=TOOL_NAMES, metadata=METADATA)
    analyses = generate([ANALYSIS.format(question=q, **tools) for q in questions], temperature=0.0, max_tokens=1024)
    memories = [{} for _ in questions]
    turns = [[] for _ in questions]
    active = list(range(len(questions)))  # questions the verifier has not stopped yet
    for step in range(1, max_turns + 1):
        if not active:
            break
        ctx = [dict(question=questions[i], analysis=analyses[i], memory=memories[i], **tools) for i in active]
        prompts = [NEXT_STEP.format(**c) for c in ctx]
        outputs = generate(prompts, temperature=temperature, max_tokens=1024, lora=lora)
        plans = [parse_plan(o) for o in outputs]
        commands = generate([COMMAND.format(question=c["question"], sub_goal=s, tool=t, metadata=METADATA.get(t), context=x)
                             for c, (x, s, t) in zip(ctx, plans)], temperature=0.0, max_tokens=512)
        results = run_tools([(t, parse_query(cmd, s)) for (_, s, t), cmd in zip(plans, commands)])
        for i, prompt, output, (_, s, t), cmd, result in zip(active, prompts, outputs, plans, commands, results):
            turns[i].append(dict(prompt=prompt, output=output))
            memories[i][f"Action Step {step}"] = dict(tool_name=t, sub_goal=s, command=cmd, result=result[:2000])
        verdicts = generate([VERIFY.format(**dict(c, memory=memories[i])) for i, c in zip(active, ctx)],
                            temperature=0.0, max_tokens=1024)
        active = [i for i, v in zip(active, verdicts) if "STOP" not in v.upper().split("CONCLUSION")[-1]]
    answers = generate([SOLUTION.format(question=q, analysis=a, memory=m) for q, a, m in zip(questions, analyses, memories)],
                       temperature=0.0, max_tokens=1024)
    return [dict(answer=a, turns=t) for a, t in zip(answers, turns)]
