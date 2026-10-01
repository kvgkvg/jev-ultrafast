"""Zero-shot local chooser: no training, no generation. Scores each offered label by next-token log-probability.

It answers the same request body the TypeSafe API receives and returns the same response shape, so `choose`
validates it identically. The target head is scored only for the operation that won (nothing to speculate on locally).
"""

import json
import math
import os
import time

from .questions import LOCAL_SYSTEM

_LOADED = {}


def load():
    if not _LOADED:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        device = os.environ.get("LOCAL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
        _LOADED.update(name=name, device=device, tokenizer=tokenizer, model=model)
    return _LOADED


def score(prompt, labels):
    """Log-probability of each label as the continuation of `prompt`, normalized over the offered labels."""
    import torch

    m = load()
    tokenizer, model, device = m["tokenizer"], m["model"], m["device"]
    start = tokenizer.encode(prompt, add_special_tokens=False)
    ends = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    totals = []
    for lo in range(0, len(labels), 16):
        chunk = ends[lo : lo + 16]
        width = len(start) + max(map(len, chunk))
        ids = torch.full((len(chunk), width), tokenizer.pad_token_id or 0)
        for row, end in enumerate(chunk):
            ids[row, : len(start) + len(end)] = torch.tensor(start + end)
        with torch.no_grad():
            logits = model(ids.to(device)).logits.float().log_softmax(-1)
        for row, end in enumerate(chunk):
            totals.append(sum(logits[row, len(start) + i - 1, t].item() for i, t in enumerate(end)))
    top = max(totals)
    z = sum(math.exp(t - top) for t in totals)
    return {label: math.exp(t - top) / z for label, t in zip(labels, totals)}


def render(state):
    elements = "\n".join(
        f"[{e['index']}] {e['role']} {e['label']!r} value={e.get('value', '')!r} ops={','.join(e['operations'])}"
        + "".join(f"\n    [{o['index']}] option {o['label']!r}" for o in e.get("options", []))
        for e in state["elements"]
    )
    recent = "\n".join(f"- {h['action']} ({h['kind']}) {h.get('text') or ''}" for h in state["recent_actions"])
    page = state["page"]
    head = f"URL: {page['url']}\nTitle: {page['title']}\nText: {page['text']}"
    return f"{head}\n\nElements:\n{elements}\n\nRecent:\n{recent}"


def ask(state, question, goal_line, header):
    criteria = question["criteria"]
    options = "\n".join(f"[{key}] {json.dumps(value)}" for key, value in criteria.items())
    rules = "\n".join(question["instructions"]["rules"]) if isinstance(question["instructions"]["rules"], list) else (
        question["instructions"]["rules"]
    )
    user = f"{render(state)}\n\nGoal: {goal_line}\n{rules}\n\n{header}\n{options}\n\nAnswer with the label in brackets."
    chat = [{"role": "system", "content": LOCAL_SYSTEM}, {"role": "user", "content": user}]
    prompt = load()["tokenizer"].apply_chat_template(chat, tokenize=False, add_generation_prompt=True) + "["
    probabilities = score(prompt, [f"{key}]" for key in criteria])
    probabilities = {key: probabilities[f"{key}]"] for key in criteria}
    choice = max(probabilities, key=probabilities.get)
    return {"choice": choice, "confidence": probabilities[choice], "probabilities": probabilities}


def answer(body):
    """Same contract as POST /v1/systemone for the heads `choose` will actually read."""
    started = time.perf_counter()
    questions, state = body["questions"], body["state"]
    goal = questions["operation"]["instructions"]["goal"]
    answers = {"operation": ask(state, questions["operation"], goal, "Choose the next operation:")}
    head = answers["operation"]["choice"].lower() + "_target"
    if head in questions:
        answers[head] = ask(state, questions[head], goal, f"Choose the target for {answers['operation']['choice']}:")
    return {
        "model": "local:" + load()["name"],
        "answers": answers,
        "usage": {"latency_ms": round((time.perf_counter() - started) * 1000)},
    }
