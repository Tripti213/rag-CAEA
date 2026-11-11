"""
Choose lambda (linking) and omega (triple scoring) on the DEV split only, then report the
chosen setting on the TEST split once. Keeps gold facts of the test questions out of tuning
(Section XI: "gold facts must not leak into ... the tuning of lambda and omega").

Objective on dev (no LLM needed): (linking hits + gold-triple hits @ m) / (linking total + gold total).
Ties go to the earlier grid value, so ties never favour a larger concept weight.
"""
import argparse
import itertools
import json

from medrag import KG, MedRAG, get_encoder
from evaluate import link_accuracy, load_questions, recall_at_m


def score(rag, qs, lam, omega, m):
    lo, lt = link_accuracy(rag, qs, lam)
    ro, rt = recall_at_m(rag, qs, "kg", omega, m, lam=lam)
    return (lo + ro) / max(lt + rt, 1), (lo, lt), (ro, rt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("--questions", default="demo_data/questions.jsonl")
    ap.add_argument("--encoder", default="hash")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--grid", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--rewrite", action="store_true", help="rewrite linked mentions to entity names for text scoring")
    ap.add_argument("--out", default="tuned.json")
    a = ap.parse_args()

    qs = load_questions(a.questions)
    dev = [q for q in qs if q.get("split") == "dev"]
    test = [q for q in qs if q.get("split") == "test"]
    assert dev and test, "questions need a 'split' field with 'dev' and 'test' values"
    grid = [float(x) for x in a.grid.split(",")]
    rag = MedRAG(KG.load(a.data), get_encoder(a.encoder), 0.5, 0.5, a.k, a.m, "both", rewrite=a.rewrite)

    rows = []
    for lam, om in itertools.product(grid, grid):
        s, l, r = score(rag, dev, lam, om, a.m)
        rows.append((s, lam, om, l, r))
    print(f"DEV grid ({len(dev)} questions)   lam \\ omega -> " + "  ".join(f"{g:>5}" for g in grid))
    for lam in grid:
        print(f"  lam={lam:<5}      " + "  ".join(f"{[r[0] for r in rows if r[1]==lam and r[2]==g][0]:5.2f}" for g in grid))
    best = max(rows, key=lambda r: r[0])        # max() keeps the first of equal maxima
    _, blam, bom, bl, br = best
    print(f"\nBest on dev: lam={blam}  omega={bom}   linking {bl[0]}/{bl[1]}  recall@{a.m} {br[0]}/{br[1]}")

    ts, tl, tr = score(rag, test, blam, bom, a.m)
    ds, dl, dr = score(rag, test, 0.5, 0.0, a.m)
    print(f"TEST ({len(test)} questions), reported once:")
    print(f"  tuned  lam={blam} omega={bom}:  linking {tl[0]}/{tl[1]}  recall@{a.m} {tr[0]}/{tr[1]}")
    print(f"  no concept in scoring (lam=0.5, omega=0): linking {dl[0]}/{dl[1]}  recall@{a.m} {dr[0]}/{dr[1]}")
    json.dump({"lam": blam, "omega": bom, "dev": {"link": bl, "recall": br},
               "test": {"link": tl, "recall": tr}}, open(a.out, "w"), indent=2)
    print(f"\nNext: python evaluate_answers.py --split test --lam {blam} --omega {bom} --llm groq")


if __name__ == "__main__":
    main()