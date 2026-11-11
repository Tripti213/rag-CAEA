"""
Convert PrimeKG's kg.csv into the folder format used by medrag.py (Section XI, step (a)).

  wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620
  python build_primekg.py --list-relations --kg kg.csv          # see what is in the file
  python build_primekg.py --kg kg.csv --out primekg_data

Expected columns: relation, display_relation, x_index, x_id, x_type, x_name, x_source,
                  y_index, y_id, y_type, y_name, y_source   (PrimeKG's published layout)

Design choices (so the "no ontology" premise of the method is kept):
  * node TYPE is used only to orient edges (drug -> disease, ...) and to filter; it is NOT
    written out, so concept vectors come from relation names only (run with --concept rel).
  * same-named but different nodes get ids 'name__<index>' and the bare name as a shared
    alias -> genuine ambiguous mentions.
  * extra abbreviations: --aliases file with lines  <bare node name> TAB <alias>  (e.g.
    'multiple sclerosis<TAB>MS'); PrimeKG itself carries no abbreviations.
  * the 4th column of triples.tsv is the provenance, e.g. 'PrimeKG:DrugBank-MONDO'.
Licensing: the PrimeKG code is MIT; check the licences of the underlying datasets
(DrugBank, MONDO, ...) before redistributing any derived file. Not a clinical resource.
"""
import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict

TYPE_ORDER = ["drug", "disease", "effect/phenotype", "anatomy", "exposure", "gene/protein",
              "pathway", "biological_process", "molecular_function", "cellular_component"]
# parent-child is deliberately NOT a default: it is an ontology hierarchy (MONDO/HPO is-a links),
# which would give the concepts the external ontology that CAEA's premise says we do without.
DEFAULT_REL = "indication,contraindication,off-label use,side effect,phenotype present"


def clean(x):
    return re.sub(r"\s+", "_", x.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--out", default="primekg_data")
    ap.add_argument("--types", default="drug,disease,effect/phenotype")
    ap.add_argument("--relations", default=DEFAULT_REL, help="display_relation values, comma-separated")
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--focus-diseases", type=int, default=0,
                    help="keep a connected subgraph around N random diseases that have an indication "
                         "(plus their drugs' side effects). Better than --max-edges, which breaks neighbourhoods")
    ap.add_argument("--include", default="", help="disease names (comma-separated) always kept by --focus-diseases")
    ap.add_argument("--include-drugs", default="",
                    help="drug names (comma-separated) always kept by --focus-diseases, with the diseases they link to")
    ap.add_argument("--max-side-effects", type=int, default=0,
                    help="keep at most N randomly chosen side effects per drug (0 = all). Disclose this in the report")
    ap.add_argument("--max-edges", type=int, default=0, help="random subsample after filtering (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-relations", action="store_true")
    a = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    if a.list_relations:
        c = Counter()
        for r in csv.DictReader(open(a.kg, encoding="utf-8")):
            c[(r["display_relation"], r["x_type"], r["y_type"])] += 1
        for (rel, xt, yt), n in c.most_common():
            print(f"{n:>9d}  {rel}   [{xt} -> {yt}]")
        return

    types = set(a.types.split(","))
    rels = {r.strip() for r in a.relations.split(",")}
    edges, seen, hit = [], set(), Counter()
    names, ntype = {}, {}
    for r in csv.DictReader(open(a.kg, encoding="utf-8")):
        rel = r["display_relation"]
        if rel not in rels or r["x_type"] not in types or r["y_type"] not in types:
            continue
        hit[rel] += 1
        x = (r["x_index"], r["x_type"], r["x_name"], r["x_source"])
        y = (r["y_index"], r["y_type"], r["y_name"], r["y_source"])
        ox = TYPE_ORDER.index(x[1]) if x[1] in TYPE_ORDER else 99
        oy = TYPE_ORDER.index(y[1]) if y[1] in TYPE_ORDER else 99
        h, t = (x, y) if ox <= oy else (y, x)          # canonical orientation, e.g. drug -> disease
        key = (rel, frozenset((h[0], t[0])))             # PrimeKG stores each edge in both directions
        if key in seen or not h[2].strip() or not t[2].strip():
            continue
        seen.add(key)
        names[h[0]], names[t[0]] = h[2], t[2]
        ntype[h[0]], ntype[t[0]] = h[1], t[1]
        edges.append((h, rel, t))
    for rel in rels:
        if hit[rel] == 0:
            print(f"WARNING: no edges for relation '{rel}' (check with --list-relations)")
    if a.focus_diseases:
        inc = {x.strip().lower() for x in a.include.split(",") if x.strip()}
        dis_nodes = {n[0]: n[2] for h, _, t in edges for n in (h, t) if n[1] == "disease"}
        forced = {i for i, nm in dis_nodes.items() if nm.lower() in inc}
        missing = inc - {dis_nodes[i].lower() for i in forced}
        if missing:
            print("WARNING: --include names not found among diseases in the filtered edges:", sorted(missing))
        pool = sorted({t[0] for h, rel, t in edges if rel == "indication" and t[1] == "disease"} - forced)
        random.Random(a.seed).shuffle(pool)
        chosen = set(pool[:max(a.focus_diseases - len(forced), 0)]) | forced
        incd = {x.strip().lower() for x in a.include_drugs.split(",") if x.strip()}
        drug_nodes = {n[0]: n[2] for h, _, t in edges for n in (h, t) if n[1] == "drug"}
        forced_drugs = {i for i, nm in drug_nodes.items() if nm.lower() in incd}
        miss_d = incd - {drug_nodes[i].lower() for i in forced_drugs}
        if miss_d:
            print("WARNING: --include-drugs names not found among drugs in the filtered edges:", sorted(miss_d))
        for h, _, t in edges:                          # diseases linked to forced drugs get full neighbourhoods
            if h[0] in forced_drugs and t[1] == "disease":
                chosen.add(t[0])
            if t[0] in forced_drugs and h[1] == "disease":
                chosen.add(h[0])
        keep = [i for i, (h, _, t) in enumerate(edges) if h[0] in chosen or t[0] in chosen or h[0] in forced_drugs or t[0] in forced_drugs]
        ks = set(keep)
        drugs = {n[0] for i in keep for n in (edges[i][0], edges[i][2]) if n[1] == "drug"}
        for i, (h, _, t) in enumerate(edges):      # drug side effects etc. for the kept drugs, no extra diseases
            if i in ks or h[1] == "disease" or t[1] == "disease":
                continue
            if (h[1] == "drug" and h[0] in drugs) or (t[1] == "drug" and t[0] in drugs):
                keep.append(i)
        edges = [edges[i] for i in sorted(keep)]
    if a.max_side_effects:
        rng, per, rest = random.Random(a.seed), defaultdict(list), []
        for e in edges:
            (per[e[0][0]] if e[1] == "side effect" else rest).append(e)
        for lst in per.values():
            rng.shuffle(lst)
            rest += lst[:a.max_side_effects]
        edges = rest
    if a.max_edges and len(edges) > a.max_edges:
        random.Random(a.seed).shuffle(edges)
        edges = edges[:a.max_edges]

    by_name = defaultdict(set)
    used = {i for h, _, t in edges for i in (h[0], t[0])}
    for i in used:
        by_name[names[i]].add(i)
    ent = {i: (clean(names[i]) if len(by_name[names[i]]) == 1 else f"{clean(names[i])}__{i}") for i in used}

    import os
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/triples.tsv", "w", encoding="utf-8") as f:
        for h, rel, t in edges:
            f.write(f"{ent[h[0]]}\t{clean(rel)}\t{ent[t[0]]}\tPrimeKG:{h[3]}-{t[3]}\n")
    with open(f"{a.out}/types.tsv", "w", encoding="utf-8") as f:   # true node types: for building/analysing the
        for i in sorted(used, key=lambda x: ent[x]):               # test set ONLY - medrag.py never reads this
            f.write(f"{ent[i]}\t{ntype[i]}\n")
    n_al = 0
    with open(f"{a.out}/aliases.tsv", "w", encoding="utf-8") as f:
        for name, ids in by_name.items():            # collisions -> ambiguous bare name
            if len(ids) > 1:
                for i in ids:
                    f.write(f"{ent[i]}\t{name}\n"); n_al += 1
    asked = []
    if a.aliases:
        low = defaultdict(set)
        for i in used:
            low[names[i].lower()].add(i)
        with open(f"{a.out}/aliases.tsv", "a", encoding="utf-8") as f:
            for line in open(a.aliases, encoding="utf-8"):
                p = line.rstrip("\n").split("\t")
                if len(p) < 2:
                    continue
                nm, _, ty = p[0].partition("|")          # 'name' or 'name|type' (type: disease, drug, ...)
                ids = {i for i in low.get(nm.strip().lower(), ()) if not ty or ntype[i] == ty.strip()}
                asked.append((p[1], nm, ids))
                for i in ids:
                    f.write(f"{ent[i]}\t{p[1]}\n"); n_al += 1
    print(f"{len(edges)} triples, {len(used)} entities, {n_al} alias rows -> {a.out}/")
    if asked:
        print("alias check (alias -> matched entities [type]); more than one entity = ambiguous mention:")
        for al, nm, ids in asked:
            print(f"  {al:20s} <- {nm:34s} -> " + (", ".join(f"{ent[i]} [{ntype[i]}]" for i in sorted(ids)) or "NOT FOUND"))
    print("triples per relation:", dict(Counter(rel for _, rel, _ in edges)))
    print("Run with:  --data", a.out, " --concept rel   (no attributes are written)")


if __name__ == "__main__":
    main()