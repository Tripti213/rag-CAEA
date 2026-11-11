"""
Question generator for a KG folder (Section XI step (a)). Writes JSONL for evaluate.py / tune.py /
evaluate_answers.py. Questions are answerable from the KG by construction: they test linking and
retrieval, not medical reasoning (add an independent hand-written set for the paper).

  # ambiguous-mention questions: an alias/name with candidates of DIFFERENT true types (needs types.tsv)
  python make_questions.py primekg_data --mode ambiguous --out q_amb.jsonl
  # plain questions from one relation
  python make_questions.py primekg_data --mode plain --rel indication --side tail -n 30 --out q_plain.jsonl
  python make_questions.py primekg_data --mode plain --rel side_effect --side head \
         --template "Which side effects can {} cause?" -n 20 --out q_se.jsonl

Every ambiguous question gets a context mention that is unambiguous and has the same true type as
the target (types.tsv is used here only, never by the method). Questions with other stray entity
mentions are dropped. dev/test is assigned per target entity, so a target never sits in both splits.
"""
import argparse
import json
import os
import random
from collections import defaultdict

from medrag import KG, pretty, tokens


def ngrams(toks, nmax=6):
    for n in range(1, nmax + 1):
        for i in range(len(toks) - n + 1):
            yield tuple(toks[i:i + n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--mode", choices=["plain", "ambiguous"], default="plain")
    ap.add_argument("--rel", default="indication")
    ap.add_argument("--side", choices=["head", "tail"], default="tail", help="plain: which endpoint is named")
    ap.add_argument("--template", default="Which drugs are indicated for {}?")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--ctx-per-target", type=int, default=3)
    ap.add_argument("--max-gold", type=int, default=50, help="gold triples kept per question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="questions_generated.jsonl")
    a = ap.parse_args()

    kg, rng = KG.load(a.data), random.Random(a.seed)
    cand = lambda text: kg.surface.get(tuple(tokens(text)), set())
    unamb = lambda e: len(cand(pretty(e))) == 1 and bool(kg.adj[e])

    def stray(q, allowed_texts):
        ok = set()
        for t in allowed_texts:
            ok |= set(ngrams(tokens(t)))
        return {g for g in ngrams(tokens(q)) if g in kg.surface} - ok

    rows = []
    if a.mode == "plain":
        groups = defaultdict(list)
        for t in kg.triples:
            if t[1] == a.rel:
                groups[t[2] if a.side == "tail" else t[0]].append(t)
        keys = sorted(k for k in groups if unamb(k))
        rng.shuffle(keys)
        for e in keys:
            q = a.template.format(pretty(e))
            if stray(q, [pretty(e)]):
                continue
            tris = groups[e]
            ans = sorted({pretty(t[0] if a.side == "tail" else t[2]) for t in tris})[:50]
            rows.append((e, {"question": q, "gold_links": {}, "gold_triples": [list(t) for t in tris[:a.max_gold]],
                             "gold_answers": ans, "kind": "plain"}))
    else:
        tp = os.path.join(a.data, "types.tsv")
        if not os.path.exists(tp):
            raise SystemExit("types.tsv not found: rebuild the KG with the current build_primekg.py")
        types = dict(l.rstrip("\n").split("\t")[:2] for l in open(tp, encoding="utf-8"))
        by_type = defaultdict(list)
        for e in kg.entities:
            if unamb(e):
                by_type[types.get(e)].append(e)
        for form, cands in sorted(kg.surface.items()):
            if len(cands) < 2 or len({types.get(c) for c in cands}) < 2:
                continue                                  # only cross-class ambiguity
            for target in sorted(cands):
                ty = types.get(target)
                if ty == "disease":
                    tris = [t for t in kg.triples if t[1] == "indication" and t[2] == target]
                    tmpl, ans = "Which drugs are indicated for {m}, as they are for {c}?", [pretty(t[0]) for t in tris]
                elif ty == "drug":
                    tris = [t for t in kg.triples if t[1] == "indication" and t[0] == target]
                    tmpl, ans = "For which conditions is {m} indicated, as {c} is?", [pretty(t[2]) for t in tris]
                else:
                    continue
                if not tris:
                    continue
                m = next((f for f in kg.forms[target] if tuple(tokens(f)) == form), " ".join(form))
                pool = [c for c in by_type[ty] if c != target]
                rng.shuffle(pool)
                made = 0
                for c in pool:
                    q = tmpl.format(m=m, c=pretty(c))
                    if stray(q, [m, pretty(c)]):
                        continue
                    rows.append((target, {"question": q, "gold_links": {" ".join(form): target},
                                          "gold_triples": [list(t) for t in tris[:a.max_gold]],
                                          "gold_answers": sorted(set(ans))[:50], "kind": "ambiguous",
                                          "candidates": sorted(cands)}))
                    made += 1
                    if made >= a.ctx_per_target:
                        break
    seen_t = sorted({k for k, _ in rows})
    rng.shuffle(seen_t)
    split_of = {k: ("dev" if i % 2 == 0 else "test") for i, k in enumerate(seen_t)}
    if a.mode == "plain":
        rows = rows[:a.n]
    with open(a.out, "w", encoding="utf-8") as f:
        for k, r in rows:
            f.write(json.dumps({"split": split_of[k], **r}, ensure_ascii=False) + "\n")
    n_dev = sum(split_of[k] == "dev" for k, _ in rows)
    print(f"wrote {len(rows)} questions -> {a.out}  (dev {n_dev}, test {len(rows) - n_dev})")


if __name__ == "__main__":
    main()