"""What a smaller page costs the shipped ranker, offline (for A6-multiday, D164).

    .venv\\Scripts\\python.exe spike\\scripts\\22_truncation_relevance.py

Each of D149's 20 judged searches is rebuilt from A3's recorded pages exactly as
`scholiast.eval.ranking.load_searches` builds it, but with each source's page cut to
its first k records (k_openalex in {25, 15}, k_arxiv in {50, 25}): A2 found an
OpenAlex page of 15 to be exactly the first 15 of its page of 25, so truncation is
what a smaller page returns. The works are merged by the matcher of record, scored
for dense relevance by the production encoder (local, no network), ranked with the
shipped weights, and scored by nDCG@10 against the judgments of the FULL pool, so a
relevant paper the smaller page no longer returns counts as lost rather than
shrinking the ideal. Differences against the full pages are paired by search, with a
seeded bootstrap. No third party is asked.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import random
import sys
from typing import Any

from m4_common import RESULTS_DIR

from scholiast.application.ports.ml import Precision
from scholiast.config.settings import Settings
from scholiast.domain import dedup, ranking
from scholiast.domain.discovery import interleave
from scholiast.eval import ranking as ev
from scholiast.eval.metrics import ndcg_at_k
from scholiast.infrastructure.discovery.arxiv import parse_feed
from scholiast.infrastructure.discovery.openalex import parse_works
from scholiast.infrastructure.ml.embedder import LocalEmbedder

CONFIGS = ((25, 50), (15, 50), (25, 25), (15, 25))
BOOTSTRAP = 10_000
SEED = "20260930/A6-multiday/truncation"
CONFIGS_NAME = "openalex25/arxiv50"
OUT = RESULTS_DIR / "22_truncation_relevance.json"


def pages() -> dict[tuple[str, str], dict[str, tuple[Any, ...]]]:
    out: dict[tuple[str, str], dict[str, tuple[Any, ...]]] = {}
    for entry in json.loads(gzip.decompress(ev.POOLS.read_bytes()))["entries"]:
        if entry["status"] != 200:
            continue
        body = entry["body"]
        page = parse_works(json.loads(body)) if entry["source"] == "openalex" else parse_feed(body)
        out.setdefault((entry["topic"], entry["phrasing"]), {})[entry["source"]] = page.candidates
    return out


def judged_map() -> tuple[dict[tuple[str, str], str], dict[str, int], dict[str, str], dict[tuple[str, str], str]]:
    data = json.loads(ev.JUDGMENTS.read_text(encoding="utf-8"))
    owner = {(t["topic"], f"{m[0]}:{m[1]}"): w["work"] for t in data["topics"] for w in t["works"] for m in w["members"]}
    grade = {w["work"]: w.get("owner_grade", w["grade"]) for t in data["topics"] for w in t["works"]}
    split = {t["topic"]: t["split"] for t in data["topics"]}
    text = {(t["topic"], ph): t[ph] for t in data["topics"] for ph in ("keyword", "natural_language")}
    return owner, grade, split, text


async def score(embedder: LocalEmbedder) -> dict[str, Any]:
    owner, grade, split, text = judged_map()
    recorded = pages()
    full_relevant: dict[str, dict[str, int]] = {}
    per_config: dict[str, dict[str, float]] = {}
    returned: dict[str, dict[str, list[str]]] = {}
    for k_oa, k_ax in CONFIGS:
        name = f"openalex{k_oa}/arxiv{k_ax}"
        per_config[name] = {}
        returned[name] = {}
        for (topic, phrasing), by_source in sorted(recorded.items()):
            qid = f"{topic}-{phrasing}"
            records = interleave([by_source.get("openalex", ())[:k_oa], by_source.get("arxiv", ())[:k_ax]])
            works = dedup.works(records, ev.JUDGED_WITH)
            judged = []
            for w in works:
                owners = {owner[(topic, ev.record_key(m))] for m in w.members}
                assert len(owners) == 1, (qid, owners)
                judged.append(owners.pop())
            returned[name][qid] = sorted({j for j in judged if grade[j] > 0})
            if (k_oa, k_ax) == CONFIGS[0]:
                full_relevant[qid] = {j: grade[j] for j in set(judged) if grade[j] > 0}
            values = await ev.dense_relevance(embedder, text[(topic, phrasing)], works)
            ranked = ranking.rank(works, values, ranking.SHIPPED_WEIGHTS, ranking_year=ev.RANKING_YEAR)
            order: list[str] = []
            for r in ranked:
                if judged[r.position] not in order:
                    order.append(judged[r.position])
            per_config[name][qid] = ndcg_at_k(order, full_relevant[qid], ev.CUTOFF)
    lost = {name: {q: len(set(returned[CONFIGS_NAME][q]) - set(returned[name][q])) for q in returned[name]}
            for name in returned}
    total = {q: len(returned[CONFIGS_NAME][q]) for q in returned[CONFIGS_NAME]}
    return {"per_config": per_config, "lost": lost, "relevant_full": total,
            "split": {f"{t}-{p}": split[t] for (t, p) in recorded}}


def compare(per: dict[str, float], base: dict[str, float], qids: list[str], seed: str) -> dict[str, Any]:
    diffs = [per[q] - base[q] for q in qids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP))
    return {"n": len(diffs), "mean": round(sum(per[q] for q in qids) / len(qids), 4),
            "difference": round(sum(diffs) / len(diffs), 4),
            "interval": [round(boots[int(0.025 * BOOTSTRAP)], 4), round(boots[int(0.975 * BOOTSTRAP) - 1], 4)],
            "searches_worse": sum(d < -1e-9 for d in diffs)}


def main() -> int:
    settings = Settings()
    embedder = LocalEmbedder(settings.embedding_model, device=settings.embedding_device,
                             precision=Precision(settings.embedding_precision))
    scored = asyncio.run(score(embedder))
    per, split = scored["per_config"], scored["split"]
    base = per["openalex25/arxiv50"]
    groups = {"all": sorted(base), "held_out": sorted(q for q in base if split[q] == "held_out")}
    report: dict[str, Any] = {
        "what": "nDCG@10 of the shipped ranker when each source's page is cut to its first k records, against the "
                "full pool's judgments (A3 pools, D149 judgments, D153 weights)",
        "embedder_version": embedder.version, "seed": SEED, "per_search": per,
    }
    for group, qids in groups.items():
        report[group] = {name: compare(per[name], base, qids, f"{SEED}/{group}/{name}") for name in per}
        for name in per:
            report[group][name]["relevant_lost"] = sum(scored["lost"][name][q] for q in qids)
            report[group][name]["relevant_in_full"] = sum(scored["relevant_full"][q] for q in qids)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for group in groups:
        for name, row in report[group].items():
            print(f"{group:8s} {name:22s} nDCG@10 {row['mean']:.3f}  vs full {row['difference']:+.4f} "
                  f"{row['interval']}  worse on {row['searches_worse']}/{row['n']}  "
                  f"relevant lost {row['relevant_lost']}/{row['relevant_in_full']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
