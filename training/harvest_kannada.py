"""Harvest frequency-ranked Kannada spellings with a capped, resumable Meta job."""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import hashlib
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from indic_transliteration import sanscript
from indicate.llm_indic import IndicLLMTransliterator, completion

MODEL = "meta/muse-spark-1.3-contributor"
MAX_OUTPUT = 3072
PRESERVATION = (
    "4. Preserve the written source exactly, including unusual names, consonant "
    "clusters, repeated syllables, aspiration, and endings. Never correct spelling, "
    "complete OCR fragments, substitute a familiar name, translate meaning, or drop "
    "or reorder syllables. Use plain ASCII Roman letters. If the source cannot be "
    "romanized without guessing, return ABSTAIN for that item."
)
EXAMPLES = [
    {"source": a, "target": b}
    for a, b in [
        ("ಬೆಂಗಳೂರು", "Bengaluru"),
        ("ಮೈಸೂರು", "Mysuru"),
        ("ಮಂಗಳೂರು", "Mangaluru"),
        ("ಹುಬ್ಬಳ್ಳಿ", "Hubballi"),
        ("ಕರ್ನಾಟಕ", "Karnataka"),
    ]
]


def dump(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def skeleton(text):
    """Normalize vowel length without erasing consonants or aspiration."""
    text = unicodedata.normalize(
        "NFKD", text.lower().replace("ṛ", "ri").replace("ṝ", "ri")
    )
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("ow", "au").replace("ou", "au")
    return re.sub("[aeiou]+", "A", text).rstrip("A")


def source_pattern(native):
    """Allow spelling alternatives only at their corresponding native positions."""
    reference = sanscript.transliterate(native, sanscript.KANNADA, sanscript.IAST)
    for source, marker in [
        ("ph", "\ue000"),
        ("ś", "\ue001"),
        ("ṣ", "\ue001"),
        ("ṃ", "\ue002"),
        ("ṁ", "\ue002"),
        ("c", "\ue003"),
        ("v", "\ue004"),
        ("j", "\ue005"),
    ]:
        reference = reference.replace(source, marker)
    shaped = skeleton(reference)
    pattern = re.escape(shaped)
    for marker, alternative in [
        ("\ue000", "(?:ph|f)"),
        ("\ue001", "(?:s|sh)"),
        ("\ue002", "[mn]"),
        ("\ue003", "ch?"),
        ("\ue004", "[vw]"),
        ("\ue005", "[jz]"),
    ]:
        pattern = pattern.replace(marker, alternative)
    return pattern


def validate(native, latin):
    """Reject abstentions, malformed output and changed consonant sequences."""
    latin = latin.strip().lower()
    if latin == "abstain":
        return None, "model-abstained"
    if not re.fullmatch(r"[a-z]+", latin):
        return None, "non-ascii-word"
    if not re.fullmatch(source_pattern(native), skeleton(latin)):
        return None, "syllable-structure-mismatch"
    letters = sum(unicodedata.category(c) == "Lo" for c in native)
    if not letters or not 0.5 <= len(latin) / letters <= 4:
        return None, "length-ratio"
    return latin, None


def parse_response(record):
    data = record["response"]
    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    if review := record.get("reviewed_suffix"):
        if (
            hashlib.sha256(content.encode()).hexdigest() != review["original_sha256"]
            or not review["reason"]
            or not content.endswith(review["content"])
        ):
            raise ValueError("Reviewed suffix does not match the original response")
        content = review["content"]
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    matches = [re.fullmatch(r"(\d+)\.\s*(.+)", line) for line in lines]
    tokens = record["request"]["tokens"]
    if choice["finish_reason"] != "stop":
        raise ValueError("Response did not finish normally")
    if len(matches) != len(tokens) or not all(
        m and int(m[1]) == i for i, m in enumerate(matches, 1)
    ):
        raise ValueError("Malformed response numbering")
    if (
        data["usage"]["completion_tokens"]
        > record["parameters"]["max_completion_tokens"] * 0.6
    ):
        raise ValueError("Output reserve gate exceeded")
    return [
        (native, *validate(native, m[2]))
        for native, m in zip(tokens, matches, strict=True)
    ]


def make_plan(root):
    trans = IndicLLMTransliterator(
        "kannada", "english", provider="meta", model=MODEL, temperature=0
    )
    with gzip.open(root / "missing_tokens_by_frequency.csv.gz", "rt") as handle:
        ranked = list(csv.DictReader(handle))[:50_000]
    eligible = [r for r in ranked if unicodedata.category(r["kannada"][0]) == "Lo"]
    tokens = [r["kannada"] for r in eligible]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Inventory has duplicate tokens")
    cache = {}
    for path in sorted(
        (root / "muse_spark_preservation_pilot").glob("response_*.json")
    ):
        record = json.loads(path.read_text())
        lines = record["response"]["choices"][0]["message"]["content"].splitlines()
        if record["response"]["choices"][0]["finish_reason"] != "stop":
            raise ValueError("Cached pilot response did not finish")
        if len(lines) != len(record["request"]["tokens"]) or not all(
            re.fullmatch(rf"{i}\.\s*.+", line.strip())
            for i, line in enumerate(lines, 1)
        ):
            raise ValueError("Malformed cached pilot response")
        for native, line in zip(record["request"]["tokens"], lines, strict=True):
            roman = line.split(". ", 1)[1]
            cache[native] = {
                "native": native,
                "roman": roman,
                "source": str(path.relative_to(root)),
            }
    cached = [cache[t] for t in tokens if t in cache]
    tokens = [t for t in tokens if t not in cache]
    requests = []
    for start in range(0, len(tokens), 25):
        group = tokens[start : start + 25]
        messages = trans.build_group_messages(group, EXAMPLES)
        messages[0]["content"] = messages[0]["content"].replace(
            "4. For proper nouns (names, places), use standard spellings when known",
            PRESERVATION,
        )
        requests.append({"tokens": group, "messages": messages})
    input_bound = sum(
        sum(len(m["content"].encode()) for m in r["messages"]) + 64 for r in requests
    )
    maximum_cost = input_bound * 0.10e-6 + len(requests) * MAX_OUTPUT * 0.20e-6
    if maximum_cost + 0.10 >= 5:
        raise ValueError("Maximum cost exceeds the total budget")
    return requests, {
        "model": MODEL,
        "ranked_items": len(ranked),
        "eligible_items": len(eligible),
        "cached_items": cached,
        "new_items": len(tokens),
        "excluded_leading_marks": len(ranked) - len(eligible),
        "eligible_occurrences": sum(int(r["occurrences"]) for r in eligible),
        "requests": len(requests),
        "max_completion_tokens": MAX_OUTPUT,
        "request_sha256": hashlib.sha256(
            json.dumps(requests, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "input_token_upper_bound_utf8_bytes_plus_framing": input_bound,
        "maximum_estimated_cost_usd": maximum_cost,
        "total_budget_usd": 5,
        "prior_pilot_budget_reserved_usd": 0.10,
        "input_per_million_usd": 0.10,
        "cached_input_per_million_usd": 0.002,
        "output_per_million_usd": 0.20,
        "pricing_is_estimate_not_invoice": True,
        "reasoning_effort": "minimal",
        "num_retries": 0,
        "selection": (
            "First 50000 missing keys ranked by frequency; exclude leading marks."
        ),
        "quality": (
            "Model abstention plus ASCII, length, consonant-order, and "
            "vowel-run-position checks. These do not certify vowel "
            "accuracy or identity."
        ),
        "storage": (
            "Contributor tier; no claim that store=False prevents "
            "retention or training."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit-requests", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit_requests <= 0:
        parser.error("--limit-requests must be positive")
    requests, plan = make_plan(args.root)
    print(
        json.dumps(
            {k: v for k, v in plan.items() if k != "cached_items"}
            | {"cached_items": len(plan["cached_items"])}
        ),
        flush=True,
    )
    if args.dry_run:
        return
    out = args.root / "muse_spark_harvest"
    out.mkdir(exist_ok=True)
    if (out / "plan.json").exists():
        if json.loads((out / "plan.json").read_text()) != plan:
            raise ValueError("Saved plan differs from this run")
    else:
        dump(out / "plan.json", plan)
    selected = list(enumerate(requests[: args.limit_requests]))
    pending = []
    for index, request in selected:
        path = out / f"response_{index:04d}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["request"] != request:
                raise ValueError("Saved response belongs to a different request")
            parse_response(record)
        else:
            if (out / f"attempt_{index:04d}.json").exists():
                raise RuntimeError(
                    f"Unresolved attempt {index}; inspect before retrying"
                )
            pending.append((index, request))
    print(f"Selected {len(selected)} requests; {len(pending)} need calls.", flush=True)
    key = getpass.getpass("Meta API key (hidden): ") if pending else None

    def run(item):
        index, request = item
        dump(
            out / f"attempt_{index:04d}.json",
            {"started_utc": datetime.now(UTC).isoformat(), "index": index},
        )
        start = time.monotonic()
        try:
            response = completion(
                model=MODEL,
                messages=request["messages"],
                api_key=key,
                temperature=0,
                reasoning_effort="minimal",
                max_completion_tokens=MAX_OUTPUT,
                timeout=120,
                num_retries=0,
                store=False,
                stream=False,
                allowed_openai_params=["reasoning_effort"],
            ).model_dump()
        except Exception as exc:
            dump(
                out / f"error_{index:04d}.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc).replace(key, "[REDACTED]")[:1500],
                },
            )
            raise RuntimeError(
                f"Request {index} failed; inspect saved receipt"
            ) from None
        usage = response["usage"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        cost = (
            (usage["prompt_tokens"] - cached) * 0.10e-6
            + cached * 0.002e-6
            + usage["completion_tokens"] * 0.20e-6
        )
        record = {
            "request": request,
            "response": response,
            "parameters": {
                "model": MODEL,
                "max_completion_tokens": MAX_OUTPUT,
                "reasoning_effort": "minimal",
                "temperature": 0,
            },
            "estimated_cost_usd": cost,
            "wall_seconds": time.monotonic() - start,
        }
        dump(out / f"response_{index:04d}.json", record)
        pairs = parse_response(record)
        return index, cost, len([p for p in pairs if p[1]])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(pending), args.workers):
            results = list(pool.map(run, pending[offset : offset + args.workers]))
            print(
                f"Completed through request {max(r[0] for r in results) + 1}"
                f"/{len(selected)}; group cost ${sum(r[1] for r in results):.6f}",
                flush=True,
            )
    records = [
        json.loads((out / f"response_{i:04d}.json").read_text()) for i, _ in selected
    ]
    pairs = [
        (r["native"], *validate(r["native"], r["roman"])) for r in plan["cached_items"]
    ] + [pair for record in records for pair in parse_response(record)]
    from collections import Counter

    audit = {
        "requests": len(records),
        "items": len(pairs),
        "accepted": sum(p[1] is not None for p in pairs),
        "rejected": dict(Counter(p[2] for p in pairs if p[2])),
        "estimated_cost_usd": sum(r["estimated_cost_usd"] for r in records),
        "prompt_tokens": sum(r["response"]["usage"]["prompt_tokens"] for r in records),
        "completion_tokens": sum(
            r["response"]["usage"]["completion_tokens"] for r in records
        ),
        "peak_completion_tokens": max(
            r["response"]["usage"]["completion_tokens"] for r in records
        ),
        "complete": len(records) == len(requests),
    }
    dump(out / "audit.json", audit)
    with (out / "validated_tokens.jsonl").open("w") as handle:
        for native, latin, reason in pairs:
            handle.write(
                json.dumps(
                    {"kannada": native, "english": latin, "rejection": reason},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
