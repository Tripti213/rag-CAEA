"""
Answer-accuracy evaluation = step (c) of the experimental plan in Section XI.

Compares four settings at a fixed context size m:
  1. no retrieval                      (LLM alone)
  2. text-only retrieval               (all triples, ranked by text similarity)
  3. KG retrieval, omega = 0           (linking + k-hop, no concept term in scoring)
  4. KG retrieval, concept-aware       (linking + k-hop, omega > 0)

Per question it records: answer correctness (any gold answer string appears in the answer),
recall of gold triples in the context, and citation behaviour ([n] markers mapped back to
triples). Per-question outputs are saved so answers can be inspected by hand.

Questions need "gold_answers": [...] (acceptable answer strings) and optionally "split".
Backends: groq | anthropic | ollama | extractive. 'extractive' needs no LLM: it echoes the context,
so its accuracy equals "answer is in the retrieved context" (use it to test offline).

Use lam/omega chosen on the DEV split (tune.py), then run once with --split test.
"""
import argparse
import json

from medrag import KG, MedRAG, answer_question, get_encoder, tokens
from evaluate import load_questions


import re

ABSTAIN = re.compile(r"(cannot|can't|can not|unable to|not able to)\s+(\w+\s+){0,3}?(determine|answer|identify|say|tell)"
                     r"|(do|does|did)\s+not\s+(contain|include|provide|list|specify|mention|state|have)"
                     r"|(not|insufficient|inadequate)\s+(enough|sufficient)?\s*(information|facts|evidence)"
                     r"|insufficient", re.I)


def abstained(answer):
    """True if the answer says the facts are insufficient (a refusal that may still repeat gold words)."""
    return bool(ABSTAIN.search(answer or ""))


def contains(answer, gold):
    a, g = " ".join(tokens(answer or "")), " ".join(tokens(gold))
    return bool(g) and f" {g} " in f" {a} "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("--questions", default="demo_data/questions.jsonl")
    ap.add_argument("--split", default="test", help="dev | test | all")
    ap.add_argument("--encoder", default="hash")
    ap.add_argument("--llm", choices=["extractive", "ollama", "groq", "anthropic"], default="extractive")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--rewrite", action="store_true", help="rewrite linked mentions to entity names for text scoring")
    ap.add_argument("--out", default="answers.json")
    a = ap.parse_args()

    qs = [q for q in load_questions(a.questions)
          if a.split == "all" or q.get("split", "all") == a.split]
    qs = [q for q in qs if q.get("gold_answers")]
    kg, enc = KG.load(a.data), get_encoder(a.encoder)
    rag = MedRAG(kg, enc, a.lam, a.omega, a.k, a.m, "both", rewrite=a.rewrite)

    settings = [("no retrieval", "none", 0.0),
                ("text-only retrieval", "global", 0.0),
                ("KG retrieval, omega=0", "kg", 0.0),
                (f"KG retrieval, concept-aware (omega={a.omega})", "kg", a.omega)]
    print(f"split={a.split}  questions={len(qs)}  m={a.m}  k={a.k}  lam={a.lam}  llm={a.llm}")
    if a.llm != "extractive":
        print(f"making {len(qs) * len(settings)} LLM calls ...")

    summary, detail = {}, []
    print("answer acc = gold string found | strict = found AND the model did not say the facts are insufficient")
    print(f"\n{'setting':46s} {'acc':>6s} {'strict':>7s} {'abstain':>8s} {'gold recall':>12s} {'cites gold':>11s} {'bad cites':>10s}")
    for name, mode, om in settings:
        acc = strict = ab = rec_h = rec_t = cg = bad = 0
        for q in qs:
            out = answer_question(rag, q["question"], a.llm, mode, a.lam, om, a.m)
            ok = any(contains(out["answer"], g) for g in q["gold_answers"])
            ctx = {tuple(t) for _, t in out["triples"]}
            gold = [tuple(g) for g in q.get("gold_triples", [])]
            cap = min(len(gold), a.m)                       # normalised like evaluate.py
            rec_h += min(sum(g in ctx for g in gold), cap)
            rec_t += cap
            cited = {tuple(c["triple"]) for c in out["cited"]}
            cg += bool(cited & set(gold))
            bad += len(out["invalid_citations"])
            acc += ok
            ab_q = abstained(out["answer"])
            ab += ab_q
            strict += ok and not ab_q
            detail.append({"setting": name, "question": q["question"], "correct": ok, "abstained": ab_q,
                           "context": [list(t) for _, t in out["triples"]],
                           "answer": out["answer"], "links": [(t, e) for t, e, _ in out["links"]],
                           "cited": [(c["n"], list(c["triple"]), c["source"]) for c in out["cited"]],
                           "invalid_citations": out["invalid_citations"]})
        n = len(qs)
        summary[name] = {"answer_correct": [acc, n], "answer_correct_strict": [strict, n], "abstained": [ab, n], "gold_recall": [rec_h, rec_t],
                         "cites_gold": [cg, n], "invalid_citations": bad}
        print(f"{name:46s} {acc:>3d}/{n:<2d} {strict:>4d}/{n:<2d} {ab:>5d}/{n:<2d} {rec_h:>9d}/{rec_t:<2d} {cg:>8d}/{n:<2d} {bad:>10d}")

    json.dump({"args": vars(a), "summary": summary, "detail": detail},
              open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved {a.out}  (inspect 'detail' to check answers by hand)")
    if a.llm == "extractive":
        print("NOTE: extractive backend -> 'answer acc' is the answer-in-context rate, not LLM accuracy;"
              " 'cites gold' is not meaningful.")


if __name__ == "__main__":
    main()