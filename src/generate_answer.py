import argparse
import json
import os
import re

import pandas as pd

ALLOWED_ROLES = {"exact", "substitute", "irrelevant"}
ALLOWED_DECISIONS = {
    "recommend_exact", "recommend_exact_with_warning",
    "recommend_substitute", "no_good_match", "ask_clarification",
}

SYSTEM_INSTRUCTION = """You are a product search assistant. Return only valid JSON.
Use only the product metadata provided. Do not invent price, availability, reviews,
materials, or stock status. If a field is unknown, mark it as unknown or leave it out."""

SCHEMA_DESCRIPTION = """Return a single JSON object with exactly these keys:
{
  "interpreted_need": {
    "category": string,
    "use_case": string,
    "positive_preferences": [string, ...],
    "negative_constraints": [string, ...],
    "visual_preferences": [string, ...],
    "uncertain_fields": [string, ...]
  },
  "product_judgements": [
    {
      "product_id": string (must be one of the candidate product_ids given below),
      "role": one of "exact" | "substitute" | "irrelevant",
      "evidence": [string, ...],
      "constraint_violations": [string, ...],
      "reason": string
    }, ...
  ],
  "decision": one of "recommend_exact" | "recommend_exact_with_warning" |
              "recommend_substitute" | "no_good_match" | "ask_clarification",
  "customer_response": string
}
Judge every candidate product listed below. Do not invent products not listed."""


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt(query, products):
    lines = [SYSTEM_INSTRUCTION, "", SCHEMA_DESCRIPTION, ""]
    if query.get("query_text"):
        lines.append(f"User query text: {query['query_text']}")
    if query.get("query_image_path"):
        lines.append("A query image was also provided by the user.")
    lines.append("\nCandidate products:")
    for p in products:
        lines.append(
            f"- product_id={p['product_id']} | title={p['title']} | "
            f"type={p.get('product_type')} | category={p.get('category_path')} | "
            f"color={p.get('color')} | material={p.get('material')} | "
            f"style={p.get('style')} | dimensions={p.get('dimensions')}"
        )
    lines.append("\nRespond with the JSON object only, no markdown fences, no extra text.")
    return "\n".join(lines)


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start:end + 1])


def validate_schema(obj, valid_product_ids):
    required = {"interpreted_need", "product_judgements", "decision", "customer_response"}
    missing = required - set(obj)
    if missing:
        return f"missing keys: {missing}"
    need = obj["interpreted_need"]
    if not isinstance(need, dict) or not isinstance(need.get("category"), str):
        return "interpreted_need.category must be a string"
    for field in ["positive_preferences", "negative_constraints", "visual_preferences", "uncertain_fields"]:
        if not isinstance(need.get(field), list):
            return f"interpreted_need.{field} must be a list"
    judgements = obj["product_judgements"]
    if not isinstance(judgements, list) or not judgements:
        return "product_judgements must be a non-empty list"
    for j in judgements:
        if j.get("product_id") not in valid_product_ids:
            return f"product_judgements references unknown product_id {j.get('product_id')}"
        if j.get("role") not in ALLOWED_ROLES:
            return f"invalid role {j.get('role')}"
        if not isinstance(j.get("evidence"), list) or not isinstance(j.get("constraint_violations"), list):
            return "evidence / constraint_violations must be lists"
        if not isinstance(j.get("reason"), str) or not j.get("reason"):
            return "judgement reason must be a non-empty string"
    if obj["decision"] not in ALLOWED_DECISIONS:
        return f"invalid decision {obj['decision']}"
    if not isinstance(obj["customer_response"], str) or len(obj["customer_response"].split()) < 8:
        return "customer_response must be a substantive string"
    return None


def fallback_answer(query_id, products):
    return {
        "query_id": query_id,
        "interpreted_need": {
            "category": "unknown", "use_case": "unknown",
            "positive_preferences": [], "negative_constraints": [],
            "visual_preferences": [], "uncertain_fields": ["could not parse model output"],
        },
        "product_judgements": [{
            "product_id": p["product_id"], "role": "substitute", "evidence": [],
            "constraint_violations": [],
            "reason": "Automated fallback: language model output could not be validated.",
        } for p in products[:3]],
        "decision": "ask_clarification",
        "customer_response": (
            "We found some candidate products for your query, but could not confidently "
            "structure a recommendation. Please refine your query or check the top results directly."
        ),
    }


def load_local_pipeline(args):
    import torch
    from transformers import pipeline

    device = args.device or ("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
    model_kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    pipe = pipeline("text-generation", model=args.llm_model,
                     device=None if args.load_in_4bit else device,
                     torch_dtype="auto", model_kwargs=model_kwargs or None)

    if args.adapter_path:
        from peft import PeftModel
        pipe.model = PeftModel.from_pretrained(pipe.model, args.adapter_path)

    return pipe


def call_llm(pipe, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    out = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False,
               temperature=None, top_p=None)
    return out[0]["generated_text"][-1]["content"]


def main():
    ap = argparse.ArgumentParser(description="Stage 5: structured LLM output")
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--llm_backend", default="local", choices=["local"])
    ap.add_argument("--llm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=700)
    ap.add_argument("--local_json_repair_attempts", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/final_answers.jsonl")
    ap.add_argument("--summary", default="reports/llm_output_summary.md")
    ap.add_argument("--debug_raw_out", default=None)
    args = ap.parse_args()

    pipe = load_local_pipeline(args)

    catalog = pd.read_parquet(args.catalog).set_index("product_id")
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    reranked = {r["query_id"]: r for r in load_jsonl(args.reranked)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_total = n_json_valid = n_schema_valid = n_repaired = 0
    examples, failure_case = [], None
    debug_records = []

    with open(args.out, "w", encoding="utf-8") as out_fh:
        for qid, q in queries.items():
            hits = sorted(reranked.get(qid, {}).get("results", []), key=lambda h: h["rank"])[: args.top_k]
            products = [
                {"product_id": h["product_id"], **catalog.loc[h["product_id"]].to_dict()}
                for h in hits
            ]
            valid_ids = {p["product_id"] for p in products}
            prompt = build_prompt(q, products)

            answer, error, repaired = None, None, False
            json_valid_first_try = False
            last_raw = ""
            attempt_prompt = prompt
            for attempt in range(args.local_json_repair_attempts + 1):
                last_raw = call_llm(pipe, attempt_prompt, args.max_new_tokens)
                try:
                    obj = extract_json(last_raw)
                    if attempt == 0:
                        json_valid_first_try = True
                        n_json_valid += 1
                except Exception as e:
                    error = f"invalid JSON: {e}"
                    attempt_prompt = prompt + f"\n\nYour previous output was invalid: {error}. Return valid JSON only."
                    repaired = attempt > 0
                    continue
                schema_error = validate_schema(obj, valid_ids)
                if schema_error is None:
                    answer = obj
                    n_schema_valid += 1
                    if attempt > 0:
                        repaired = True
                    break
                error = schema_error
                attempt_prompt = prompt + f"\n\nYour previous output failed validation: {error}. Fix it and return valid JSON only."
                repaired = True

            n_total += 1
            used_fallback = answer is None
            n_repaired += bool(repaired and not used_fallback)
            if used_fallback:
                answer = fallback_answer(qid, products)
                if failure_case is None:
                    failure_case = (qid, error)
            answer["query_id"] = qid

            out_fh.write(json.dumps(answer) + "\n")
            if len(examples) < 3:
                examples.append(answer)

            debug_records.append({
                "query_id": qid,
                "raw_output": last_raw,
                "json_valid_first_try": json_valid_first_try,
                "repaired": bool(repaired),
                "used_fallback": used_fallback,
                "last_error": error,
            })

    if args.debug_raw_out:
        os.makedirs(os.path.dirname(args.debug_raw_out) or ".", exist_ok=True)
        with open(args.debug_raw_out, "w", encoding="utf-8") as fh:
            for rec in debug_records:
                fh.write(json.dumps(rec) + "\n")

    os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as fh:
        fh.write("# Stage 5 — LLM Output Summary\n\n")
        fh.write(f"- Language model: `{args.llm_model}`"
                 + (f" + adapter `{args.adapter_path}`" if args.adapter_path else "") + "\n")
        fh.write("- Prompting template: system instruction forbidding invented attributes, "
                  "explicit JSON schema description, candidate product list with metadata only\n")
        fh.write(f"- Number of generated answers: {n_total}\n")
        fh.write(f"- JSON validity rate (first attempt): {n_json_valid / max(n_total, 1):.2%}\n")
        fh.write(f"- Schema validity rate (after retries): {n_schema_valid / max(n_total, 1):.2%}\n")
        fh.write(f"- Number of repaired outputs (needed a retry or fallback): {n_repaired}\n\n")
        fh.write("## Example final answers\n\n")
        for a in examples:
            fh.write("```json\n" + json.dumps(a, indent=2) + "\n```\n\n")
        fh.write("## Limitations / failure case\n\n")
        if failure_case:
            fh.write(f"Query `{failure_case[0]}` fell back to the deterministic fallback answer "
                      f"after {args.local_json_repair_attempts} retries; last validation error: {failure_case[1]}\n")
        else:
            fh.write("No query required the deterministic fallback in this run.\n")

    print(f"Wrote {n_total} answers to {args.out}")


if __name__ == "__main__":
    main()
