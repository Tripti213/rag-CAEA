"""Print per-question answers from answers.json so they can be checked by hand.
Usage: python check_answers.py [answers.json] [setting-substring]
Default setting: 'omega=0' (matches 'KG retrieval, omega=0'). Try 'no retrieval', 'text-only', 'concept-aware'."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "answers.json"
want = sys.argv[2] if len(sys.argv) > 2 else "KG retrieval, omega=0"
detail = json.load(open(path))["detail"]
# 'omega=0' is a substring of 'omega=0.25', so prefer an exact setting-name match when there is one
exact = any(r["setting"] == want for r in detail)
for r in detail:
    if (r["setting"] == want) if exact else (want in r["setting"]):
        print(("OK  " if r["correct"] else "MISS"), "|", r["question"])
        print("     links:", r["links"], "| abstained:", r.get("abstained"))
        print("     context:", r.get("context", "(not saved in this file - rerun evaluate_answers.py)"))
        print("     answer:", (r["answer"] or "").replace("\n", " ")[:300])
        print("     cited:", [c[0] for c in r["cited"]], "\n")