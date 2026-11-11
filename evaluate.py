"""
Evaluation for the experimental plan of Section XI:
  (b) entity-linking accuracy on ambiguous mentions, with / without the concept term
  (c) retrieval recall of gold triples at fixed m for:
        - text-only retrieval over all triples (no graph)         mode 'global'
        - KG retrieval with omega = 0 (no concept term)
        - KG retrieval with concept-aware scoring (omega > 0)
  (d) ablation of the relative / independent concept

Question file (JSONL), one object per line:
  {"question": "...",
   "gold_links": {"ms": "Multiple_sclerosis"},      # lower-cased mention -> entity (ambiguous ones)
   "gold_triples": [["Interferon_beta","treats","Multiple_sclerosis"]]}

Answer accuracy needs a language model and a QA set with gold answers; it is not
computed here (see medrag.py --llm).
"""
import argparse
import json

from medrag import KG, MedRAG, get_encoder


def load_questions(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def link_accuracy(rag, qs, lam, tie_seed=None, misses=None):
    ok = tot = 0
    for q in qs:
        gold = q.get("gold_links", {})
        if not gold:
            continue
        pred = {t.lower(): e for t, e, _ in rag.link(q["question"], lam=lam, tie_seed=tie_seed)}
        for mention, ent in gold.items():
            tot += 1
            good = pred.get(mention.lower()) == ent
            ok += good
            if misses is not None and not good:
                misses.append((q["question"], mention, pred.get(mention.lower()), ent))
    return ok, tot


def recall_at_m(rag, qs, mode, omega, m, lam=None, hit=False):
    """Normalised recall: gold triples found / min(#gold, m) per question (a question with 40 valid
    triples cannot need more than m of them). hit=True also returns questions with >=1 gold triple."""
    found = tot = qh = nq = 0
    for q in qs:
        gold = {tuple(g) for g in q["gold_triples"]}
        if not gold:
            continue
        got = {tuple(t) for _, t in rag.retrieve(q["question"], mode=mode, lam=lam,
                                                 omega=omega, m=m)["triples"]}
        f, cap = len(gold & got), min(len(gold), m)
        found += min(f, cap)
        tot += cap
        qh += f > 0
        nq += 1
    return (found, tot, qh, nq) if hit else (found, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("--questions", default="demo_data/questions.jsonl")
    ap.add_argument("--encoder", default="hash")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--omegas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--rewrite", action="store_true", help="rewrite linked mentions to entity names for text scoring")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--show-misses", action="store_true", help="list linking mistakes of the lam setting")
    a = ap.parse_args()

    kg, enc, qs = KG.load(a.data), get_encoder(a.encoder), load_questions(a.questions)
    omegas = [float(x) for x in a.omegas.split(",")]
    results = {"encoder": a.encoder, "k": a.k, "m": a.m, "lam": a.lam, "n_questions": len(qs)}

    rag = MedRAG(kg, enc, a.lam, 0.5, a.k, a.m, "both", rewrite=a.rewrite)

    print(f"\n== (b) Entity linking on ambiguous mentions (encoder={a.encoder}) ==")
    for name, lam in [("name only (lam=1), alphabetical", 1.0), (f"name + concept (lam={a.lam})", a.lam), ("concept only (lam=0)", 0.0)]:
        ok, tot = link_accuracy(rag, qs, lam)
        results[f"link[{name}]"] = [ok, tot]
        print(f"  {name:32s} {ok}/{tot}")
    exp = sum(link_accuracy(rag, qs, 1.0, tie_seed=s)[0] for s in range(200)) / 200
    results["link[name only, random ties]"] = [exp, tot]
    print(f"  {'name only (lam=1), random ties':32s} {exp:.1f}/{tot}   <- fair baseline (mean of 200 tie-break seeds)")
    if a.show_misses:
        miss = []
        link_accuracy(rag, qs, a.lam, misses=miss)
        print(f"\n  linking mistakes at lam={a.lam} ({len(miss)}):")
        for qq, mention, pred, gold in miss:
            print(f"    '{mention}': predicted {pred}, gold {gold}   | {qq}")

    print(f"\n== (c) Gold triples @ m={a.m}, k={a.k}  (recall normalised by min(#gold, m); hit = >=1 gold triple) ==")

    def row(name, mode, w):
        f, t, h, n = recall_at_m(rag, qs, mode, w, a.m, hit=True)
        results[f"recall[{name}]"] = {"norm_recall": [f, t], "hit": [h, n]}
        print(f"  {name:32s} recall {f}/{t}   hit@{a.m} {h}/{n}")

    row("text-only, no graph", "global", 0.0)
    for w in omegas:
        row("KG, omega=" + str(w), "kg", w)

    print("\n== (d) Concept ablation (KG retrieval, omega=0.5 and lam as given) ==")
    modes = ("rel", "attr", "both") if kg.attrs else ("rel",)
    if not kg.attrs:
        print("  (this KG has no attributes: only the relative concept exists, so the attr/both rows are skipped)")
    for mode in modes:
        r = MedRAG(kg, enc, a.lam, 0.5, a.k, a.m, mode, rewrite=a.rewrite)
        lo, lt = link_accuracy(r, qs, a.lam)
        ro, rt = recall_at_m(r, qs, "kg", 0.5, a.m)
        results[f"ablation[{mode}]"] = {"link": [lo, lt], "recall": [ro, rt]}
        print(f"  concept={mode:5s}  linking {lo}/{lt}   recall@{a.m} {ro}/{rt}")

    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()