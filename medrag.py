"""
Concept-aware retrieval for medical KG-RAG (training-free CAEA-style concept vectors).

Pipeline (see Section XI of the report):
  1. concept vectors   c_i = [ mean(relation embeddings) || mean(attribute-name embeddings) ]
  2. entity linking    s(m,e) = lam*cos(enc(m), enc(name_e)) + (1-lam)*cos(c_e, c_q)
  3. triple scoring    psi    = omega*cos(c_h, c_t) + (1-omega)*cos(enc(q), enc(txt(h,r,t)))
  4. generation        an LLM answers from the top-m triples only (optional backend)

No model is trained. Encoders: "hash" (offline fallback, lexical only) or any
Hugging Face encoder name (mean-pooled), e.g. "bert-base-uncased".
"""
from __future__ import annotations

import argparse
import random
import json
import os
import re
import zlib
from collections import defaultdict, deque

import numpy as np


# --------------------------------------------------------------------------
# Encoders (all return L2-normalised float32 vectors, so cosine == dot product)
# --------------------------------------------------------------------------
def _norm_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class HashEncoder:
    """Offline fallback: hashed character n-grams. Captures spelling overlap only,
    NOT medical semantics. Use for tests; use a real encoder for experiments."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = " " + re.sub(r"[^a-z0-9]+", " ", t.lower()).strip() + " "
            for n in (3, 4):
                for j in range(len(t) - n + 1):
                    out[i, zlib.crc32(t[j:j + n].encode()) % self.dim] += 1.0
        return _norm_rows(out)


class HFEncoder:
    """Mean-pooled Hugging Face encoder (BERT-style). Uses MPS/CUDA if available."""

    def __init__(self, name: str = "bert-base-uncased", batch: int = 64, max_len: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.batch, self.max_len = batch, max_len

    def encode(self, texts):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                chunk = list(texts[i:i + self.batch])
                enc = self.tok(chunk, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
                hid = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                out.append(((hid * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
        return _norm_rows(np.concatenate(out).astype(np.float32))


class CachedEncoder:
    """Encode each distinct string once. With cache_path, encodings persist on disk between runs
    (keyed by text only, one file per model), so BERT over a large KG is paid for once."""

    def __init__(self, base, cache_path=None):
        self.base, self.cache, self.cache_path, self._dirty = base, {}, cache_path, False
        if cache_path and os.path.exists(cache_path):
            try:
                import pickle
                with open(cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}                      # unreadable cache: start fresh
        if cache_path:
            import atexit
            atexit.register(self.save)

    def save(self):
        if not (self.cache_path and self._dirty):
            return
        import pickle
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.cache_path)
        self._dirty = False

    def encode(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            for t, v in zip(missing, self.base.encode(missing)):
                self.cache[t] = v
            self._dirty = True
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)


def get_encoder(name: str, cache_dir: str = ".enc_cache") -> CachedEncoder:
    if name == "hash":
        return CachedEncoder(HashEncoder())          # cheap: no disk cache
    path = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".pkl")
    return CachedEncoder(HFEncoder(name), cache_path=path)


# --------------------------------------------------------------------------
# Knowledge graph container
# --------------------------------------------------------------------------
def pretty(s: str) -> str:
    """Readable form. A '__suffix' (used to keep same-named entities apart) is hidden."""
    return s.split("__")[0].replace("_", " ")


def tokens(s: str):
    return re.findall(r"[a-z0-9\+\-]+", s.lower())


class KG:
    """Triples (head, relation, tail), entity attributes, and surface forms.

    Folder format (tab-separated, UTF-8):
      triples.tsv     head <TAB> relation <TAB> tail [<TAB> source]
      attributes.tsv  entity <TAB> attribute_name <TAB> value      (optional)
      aliases.tsv     entity <TAB> alias                           (optional)
    Entity ids are used as names (underscores become spaces).
    """

    def __init__(self, triples, attrs=None, aliases=None):
        self.triples, self.sources = [], {}
        for t in triples:
            tr = tuple(t[:3])
            self.triples.append(tr)
            if len(t) > 3 and t[3]:
                self.sources[tr] = t[3]   # provenance, shown to the LLM and cited
        self.attrs = defaultdict(set)
        for e, a, *_ in (attrs or []):
            self.attrs[e].add(a)
        self.aliases = defaultdict(set)
        for e, a in (aliases or []):
            self.aliases[e].add(a)

        ents = {x for h, _, t in self.triples for x in (h, t)} | set(self.attrs)
        self.entities = sorted(ents)
        self.idx = {e: i for i, e in enumerate(self.entities)}

        self.adj = defaultdict(list)  # entity -> indices of triples touching it
        self.rels = defaultdict(set)  # entity -> relation strings (incl. inverse)
        for i, (h, r, t) in enumerate(self.triples):
            self.adj[h].append(i)
            self.adj[t].append(i)
            self.rels[h].add(pretty(r))
            self.rels[t].add("inverse of " + pretty(r))

        # surface form (token tuple) -> set of entities
        self.surface = defaultdict(set)
        self.forms = defaultdict(list)  # entity -> readable surface strings
        for e in self.entities:
            for form in [pretty(e)] + sorted(self.aliases.get(e, [])):
                self.forms[e].append(form)
                self.surface[tuple(tokens(form))].add(e)

    @classmethod
    def load(cls, folder: str):
        def read(name, n):
            p = os.path.join(folder, name)
            if not os.path.exists(p):
                return []
            rows = []
            for line in open(p, encoding="utf-8"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= n:
                    rows.append(parts)
            return rows

        return cls(read("triples.tsv", 3), read("attributes.tsv", 2), read("aliases.tsv", 2))


# --------------------------------------------------------------------------
# 1. Concept vectors (mean aggregation of relation / attribute-name embeddings)
# --------------------------------------------------------------------------
def build_concepts(kg: KG, enc, mode: str = "both") -> np.ndarray:
    """c_i = c_i^r || c_i^a. mode: 'rel', 'attr' or 'both' (for the ablation)."""
    rel_names = sorted({r for rs in kg.rels.values() for r in rs})
    attr_names = sorted({a for aa in kg.attrs.values() for a in aa})
    rel_emb = dict(zip(rel_names, enc.encode(rel_names))) if rel_names else {}
    att_emb = dict(zip(attr_names, enc.encode(attr_names))) if attr_names else {}
    d = (next(iter(rel_emb.values())).shape[0] if rel_emb
         else next(iter(att_emb.values())).shape[0])

    C = np.zeros((len(kg.entities), 2 * d), dtype=np.float32)
    for e, i in kg.idx.items():
        if mode in ("rel", "both") and kg.rels.get(e):
            v = np.mean([rel_emb[r] for r in kg.rels[e]], axis=0)
            C[i, :d] = v / max(np.linalg.norm(v), 1e-12)
        if mode in ("attr", "both") and kg.attrs.get(e):
            v = np.mean([att_emb[a] for a in kg.attrs[e]], axis=0)
            C[i, d:] = v / max(np.linalg.norm(v), 1e-12)
    return C


def cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(a @ b / (na * nb))


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class MedRAG:
    def __init__(self, kg: KG, enc, lam=0.5, omega=0.5, k=1, m=8,
                 concept_mode="both", max_neighbors=200, rewrite=False):
        self.rewrite = rewrite        # replace linked mentions by the entity's name in the TEXT term only
        self.kg, self.enc = kg, enc
        self.lam, self.omega, self.k, self.m = lam, omega, k, m
        self.max_neighbors = max_neighbors
        self.C = build_concepts(kg, enc, concept_mode)
        self.maxlen = max((len(f) for f in kg.surface), default=1)

    # -- mention detection: greedy longest match of KG surface forms ---------
    def find_mentions(self, question: str):
        toks = tokens(question)
        out, i = [], 0
        while i < len(toks):
            hit = None
            for n in range(min(self.maxlen, len(toks) - i), 0, -1):
                key = tuple(toks[i:i + n])
                if key in self.kg.surface:
                    hit = (" ".join(key), sorted(self.kg.surface[key]), n)
                    break
            if hit:
                out.append(hit[:2])
                i += hit[2]
            else:
                i += 1
        return out

    # -- 2. entity linking ---------------------------------------------------
    def link(self, question: str, lam=None, tie_seed=None):
        lam = self.lam if lam is None else lam
        mentions = self.find_mentions(question)
        unamb = [self.kg.idx[c[0]] for _, c in mentions if len(c) == 1]
        cq = self.C[unamb].mean(0) if unamb else None
        links = []
        for text, cands in mentions:
            if len(cands) == 1:
                links.append((text, cands[0], 1.0))
                continue
            mv = self.enc.encode([text])[0]
            best = None
            order = list(cands)
            if tie_seed is not None:                      # random order -> ties are broken at random
                random.Random(f"{tie_seed}|{text}").shuffle(order)
            for e in order:
                name_sim = max(float(mv @ v) for v in self.enc.encode(self.kg.forms[e]))
                con = cos(self.C[self.kg.idx[e]], cq) if cq is not None else 0.0
                s = lam * name_sim + (1 - lam) * con if cq is not None else name_sim
                if best is None or s > best[2]:
                    best = (text, e, s)
            links.append(best)
        return links

    # -- 3. candidate triples and concept-aware scoring ----------------------
    def khop_triples(self, seeds, k=None):
        k = self.k if k is None else k
        seen, frontier, tri = set(seeds), deque((s, 0) for s in seeds), set()
        while frontier:
            e, depth = frontier.popleft()
            for ti in self.kg.adj[e][: self.max_neighbors]:
                tri.add(ti)
                if depth < k:
                    h, _, t = self.kg.triples[ti]
                    for nb in (h, t):
                        if nb not in seen:
                            seen.add(nb)
                            frontier.append((nb, depth + 1))
        return sorted(tri)

    @staticmethod
    def verbalize(triple):
        h, r, t = triple
        return f"{pretty(h)} {pretty(r)} {pretty(t)}"

    def score_triples(self, question, tri_ids, omega=None):
        omega = self.omega if omega is None else omega
        if not tri_ids:
            return []
        trips = [self.kg.triples[i] for i in tri_ids]
        q = self.enc.encode([question])[0]
        txt = self.enc.encode([self.verbalize(t) for t in trips])
        text_sim = txt @ q
        scored = []
        for t, ts in zip(trips, text_sim):
            ch, ct = self.C[self.kg.idx[t[0]]], self.C[self.kg.idx[t[2]]]
            scored.append((omega * cos(ch, ct) + (1 - omega) * float(ts), t))
        return sorted(scored, key=lambda x: -x[0])

    @staticmethod
    def rewrite_question(question, links):
        """Link-aware query rewriting: 'Which drugs treat MS?' -> 'Which drugs treat multiple sclerosis?'.
        Triples are verbalised with full entity names, so an abbreviation in the question would
        otherwise score poorly against them. Used only for text similarity; the LLM sees the original."""
        out = question
        for text, ent, _ in links:
            name = pretty(ent)
            if text.lower() != name.lower():
                out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(text).replace(r"\ ", r"\s+") + r"(?![A-Za-z0-9])",
                             lambda _m, n=name: n, out, count=1, flags=re.I)
        return out

    def retrieve(self, question, mode="kg", lam=None, omega=None, m=None):
        """mode 'kg': link -> k-hop -> score.  mode 'global': text-only over all triples."""
        m = self.m if m is None else m
        if mode == "global":
            ranked = self.score_triples(question, list(range(len(self.kg.triples))), omega=0.0)
            return {"links": [], "triples": ranked[:m]}
        links = self.link(question, lam)
        seeds = [e for _, e, _ in links]
        qtext = self.rewrite_question(question, links) if self.rewrite else question
        ranked = self.score_triples(qtext, self.khop_triples(seeds), omega)
        return {"links": links, "triples": ranked[:m]}


# --------------------------------------------------------------------------
# 4. Generation (optional)
# --------------------------------------------------------------------------
def build_prompt(question, scored_triples, kg=None):
    """Grounded prompt. With no triples it becomes the no-retrieval baseline prompt."""
    if not scored_triples:
        return ("Answer the medical question briefly from your own knowledge. "
                "If you are unsure, say so. This is a research prototype, not medical advice.\n\n"
                f"Question: {question}\nAnswer:")
    lines = []
    for i, (_, t) in enumerate(scored_triples):
        src = kg.sources.get(t) if kg is not None else None
        lines.append(f"[{i + 1}] {MedRAG.verbalize(t)}" + (f"  (source: {src})" if src else ""))
    facts = "\n".join(lines)
    return (
        "You answer medical questions using ONLY the numbered facts below. "
        "Cite every fact you use like [1]. If the facts are not sufficient, say so. "
        "This is a research prototype, not medical advice.\n\n"
        f"Facts:\n{facts}\n\nQuestion: {question}\nAnswer:"
    )


def parse_citations(text, scored_triples, kg=None):
    """Map citation markers in an answer back to triples (and their sources).
    Accepts [1], [1][2], [1, 2], ranges [2-8] (any dash) and fullwidth brackets 【1】 (some models emit these).
    Returns (cited, invalid): cited = [{n, triple, source}], invalid = out-of-range numbers."""
    nums = []
    for grp in re.findall(r"[\[【]\s*([\d,\s\-\u2010-\u2015]+?)\s*[\]】]", text or ""):
        for a_, b_, c_ in re.findall(r"(\d+)\s*[\-\u2010-\u2015]\s*(\d+)|(\d+)", grp):   # '2-8' (any dash) is a range
            if c_:
                nums.append(int(c_))
            elif int(a_) <= int(b_) <= int(a_) + 50:
                nums += list(range(int(a_), int(b_) + 1))
    cited, invalid, seen = [], [], set()
    for n in nums:
        if n in seen:
            continue
        seen.add(n)
        if 1 <= n <= len(scored_triples):
            t = scored_triples[n - 1][1]
            cited.append({"n": n, "triple": t, "source": (kg.sources.get(t, "") if kg else "")})
        else:
            invalid.append(n)
    return cited, invalid


def answer_question(rag, question, backend="none", mode="kg", lam=None, omega=None, m=None):
    """Full pipeline for one question.
    mode: 'none' (no retrieval), 'global' (text-only over all triples), 'kg' (linking + k-hop + concept scoring)."""
    if mode == "none":
        res = {"links": [], "triples": []}
    else:
        res = rag.retrieve(question, mode=mode, lam=lam, omega=omega, m=m)
    prompt = build_prompt(question, res["triples"], rag.kg)
    answer = generate(prompt, backend)
    cited, invalid = parse_citations(answer, res["triples"], rag.kg) if answer else ([], [])
    return {"links": res["links"], "triples": res["triples"], "prompt": prompt,
            "answer": answer, "cited": cited, "invalid_citations": invalid}


def generate(prompt: str, backend: str = "none"):
    if backend == "none":
        return None
    if backend == "extractive":
        # No LLM: echoes the numbered facts. Accuracy with this backend == "answer is in the
        # retrieved context". Useful for offline tests and as an upper-bound proxy.
        return "\n".join(re.findall(r"(?m)^\[\d+\] .*$", prompt))
    if backend == "ollama":
        import urllib.request
        body = json.dumps({"model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
                           "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                     {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["response"]
    if backend == "groq":
        # OpenAI-compatible API; needs GROQ_API_KEY (console.groq.com/keys).
        # GROQ_MODEL (default openai/gpt-oss-120b; try openai/gpt-oss-20b if rate-limited).
        # Model access differs per account and changes over time: list yours with
        #   curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
        import time
        import urllib.error
        import urllib.request
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (export GROQ_API_KEY=gsk_...)")
        url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1") + "/chat/completions"
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": 1500}   # reasoning models spend tokens thinking
        if model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                   "User-Agent": "medrag/1.0"}   # a default Python UA can be rejected with HTTP 403
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
                resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
                text = (resp["choices"][0]["message"].get("content") or "").strip()
                if not text:
                    raise RuntimeError(f"Groq returned an empty answer (model {model} may have used all "
                                       "tokens on reasoning); try another GROQ_MODEL")
                return text
            except urllib.error.HTTPError as e:
                err = e.read().decode(errors="replace")
                if e.code == 400 and "reasoning_effort" in payload and "reasoning" in err.lower():
                    payload.pop("reasoning_effort")           # parameter not accepted: retry without it
                    continue
                if e.code in (429, 500, 502, 503) and attempt < 7:   # free tier: rate limited
                    wait = float(e.headers.get("retry-after") or min(2 ** attempt, 30))
                    print(f"  [groq] HTTP {e.code}, retrying in {wait:.0f}s ...", flush=True)
                    time.sleep(wait + 0.5)
                    continue
                hint = (" -> your key cannot use this model. List the models you can use with the curl "
                        "command in medrag.py, then set GROQ_MODEL." if e.code in (403, 404) else "")
                raise RuntimeError(f"Groq API error {e.code}: {err[:300]}{hint}")
        raise RuntimeError("Groq API: too many retries")
    if backend == "anthropic":
        import anthropic  # pip install anthropic; needs ANTHROPIC_API_KEY
        msg = anthropic.Anthropic().messages.create(
            model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"), max_tokens=500,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    raise ValueError(f"unknown backend {backend}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Concept-aware medical KG-RAG")
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("--question", required=True)
    ap.add_argument("--encoder", default="hash", help="'hash' or a Hugging Face model name")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--rewrite", action="store_true", help="rewrite linked mentions to entity names for text scoring")
    ap.add_argument("--concept", choices=["rel", "attr", "both"], default="both")
    ap.add_argument("--llm", choices=["none", "extractive", "ollama", "groq", "anthropic"], default="none")
    a = ap.parse_args()

    rag = MedRAG(KG.load(a.data), get_encoder(a.encoder), a.lam, a.omega, a.k, a.m, a.concept, rewrite=a.rewrite)
    out = answer_question(rag, a.question, a.llm, "kg", a.lam, a.omega, a.m)
    print("Links:")
    for text, ent, sc in out["links"]:
        print(f"  '{text}' -> {ent}  (score {sc:.3f})")
    print("Top triples:")
    for i, (sc, t) in enumerate(out["triples"], 1):
        src = rag.kg.sources.get(t)
        print(f"  [{i}] {sc:.3f}  {MedRAG.verbalize(t)}" + (f"   <{src}>" if src else ""))
    if out["answer"] is None:
        print("\nPrompt for the language model:\n" + out["prompt"])
    else:
        print("\nAnswer:\n" + out["answer"])
        print("\nSources cited:")
        for c in out["cited"]:
            print(f"  [{c['n']}] {MedRAG.verbalize(c['triple'])}" + (f"  <{c['source']}>" if c["source"] else ""))
        if out["invalid_citations"]:
            print("  WARNING: citations to non-existent facts:", out["invalid_citations"])


if __name__ == "__main__":
    main()