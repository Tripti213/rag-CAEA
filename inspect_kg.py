"""
Show what a mention can mean in a KG folder, and whether concept vectors can separate the candidates.

  python inspect_kg.py primekg_data "rheumatoid arthritis" "ms" "codeine"     # candidates per mention
  python inspect_kg.py primekg_data                                           # ambiguity inventory of the KG

For every candidate: number of triples, its relation set (what its concept vector is built from)
and a few triples. For ambiguous mentions it prints the cosine between the candidates' concept
vectors: close to 1.0 means the concept term cannot tell them apart (a same-class tie).
"""
import argparse
import itertools

from medrag import KG, MedRAG, build_concepts, cos, get_encoder, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("mentions", nargs="*")
    ap.add_argument("--encoder", default="hash")
    ap.add_argument("--concept", default="rel", choices=["rel", "attr", "both"])
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--relation-cos", action="store_true",
                    help="mean concept cosine cos(c_head, c_tail) per relation (what Eq. 23's concept term rewards)")
    a = ap.parse_args()

    kg = KG.load(a.data)
    print(f"{len(kg.entities)} entities, {len(kg.triples)} triples")
    if a.relation_cos:
        from collections import defaultdict as dd
        C = build_concepts(kg, get_encoder(a.encoder), a.concept)
        acc = dd(list)
        for h, r, t in kg.triples:
            acc[r].append(cos(C[kg.idx[h]], C[kg.idx[t]]))
        print("relation                          triples   mean cos(c_h, c_t)")
        for r, v in sorted(acc.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            print(f"  {r:30s} {len(v):8d}   {sum(v) / len(v):.3f}")
        return
    if not a.mentions:
        amb = [(k, v) for k, v in kg.surface.items() if len(v) > 1]
        print(f"{len(amb)} surface forms map to more than one entity (natural + alias ambiguity). Examples:")
        for k, v in sorted(amb, key=lambda kv: -len(kv[1]))[:a.top]:
            print(f"  '{' '.join(k)}' -> {sorted(v)}")
        return

    C = build_concepts(kg, get_encoder(a.encoder), a.concept)
    for m in a.mentions:
        cands = sorted(kg.surface.get(tuple(tokens(m)), []))
        print(f"\n== '{m}' -> {len(cands)} candidate(s)")
        for e in cands:
            tri = [kg.triples[i] for i in kg.adj[e]]
            print(f"  {e}: {len(tri)} triples; relations: {sorted(kg.rels[e])}")
            for t in tri[:3]:
                print(f"      {MedRAG.verbalize(t)}")
        if len(cands) > 1:
            print("  concept cosine between candidates:")
            for x, y in itertools.combinations(cands, 2):
                print(f"    {x}  vs  {y}: {cos(C[kg.idx[x]], C[kg.idx[y]]):.2f}")


if __name__ == "__main__":
    main()