import argparse
import gzip
import json
import os
import re
import shutil
from collections import Counter

import pandas as pd

from tests import run_stage0

HOME_CATEGORY_KEYWORDS = [
    "Furniture", "Home & Kitchen", "Home Decor", "Home Décor",
    "Lighting", "Bedding", "Storage & Organization", "Kitchen & Dining",
]
PREFERRED_LANG_TAGS = ["en_US", "en_GB", "en_IN", "en_CA", "en_AU"]
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _clean_html(text):
    if not text:
        return None
    text = HTML_TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text or None


def _pick_english(values):
    if not values:
        return None
    by_tag = {v.get("language_tag"): v.get("value") for v in values if v.get("value")}
    for tag in PREFERRED_LANG_TAGS:
        if tag in by_tag:
            return by_tag[tag]
    for tag, value in by_tag.items():
        if tag and tag.startswith("en"):
            return value
    for v in values:
        if v.get("value"):
            return v["value"]
    return None


def _pick_english_list(values, sep=" "):
    if not values:
        return ""
    by_tag = {}
    for v in values:
        tag = v.get("language_tag")
        val = v.get("value")
        if val:
            by_tag.setdefault(tag, []).append(val)
    for tag in PREFERRED_LANG_TAGS:
        if tag in by_tag:
            return sep.join(by_tag[tag])
    for tag, vals in by_tag.items():
        if tag and tag.startswith("en"):
            return sep.join(vals)
    flat = [v.get("value") for v in values if v.get("value")]
    return sep.join(flat)


def _category_path(item):
    nodes = item.get("node") or []
    if not nodes:
        return None
    # ABO node entries look like {"node_id": ..., "node_name": "A/B/C"}
    names = [n.get("node_name") for n in nodes if n.get("node_name")]
    return names[0] if names else None


def _matches_home_category(category_path, product_type, keywords):
    haystack = " ".join(filter(None, [category_path or "", product_type or ""]))
    return any(kw.lower() in haystack.lower() for kw in keywords)


def load_listings(listings_dir):
    for name in sorted(os.listdir(listings_dir)):
        if not (name.startswith("listings_") and name.endswith(".json.gz")):
            continue
        path = os.path.join(listings_dir, name)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_image_metadata(images_metadata_dir):
    mapping = {}
    for name in sorted(os.listdir(images_metadata_dir)):
        if not name.endswith(".csv.gz") and not name.endswith(".csv"):
            continue
        path = os.path.join(images_metadata_dir, name)
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
            idx = {col: i for i, col in enumerate(header)}
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) < len(header):
                    continue
                image_id = parts[idx["image_id"]]
                mapping[image_id] = {
                    "path": parts[idx["path"]],
                    "height": parts[idx.get("height", -1)] if "height" in idx else None,
                    "width": parts[idx.get("width", -1)] if "width" in idx else None,
                }
    return mapping


def build_product_text(row):
    template = (
        "Title: {title}\n"
        "Category: {category_path}\n"
        "Product type: {product_type}\n"
        "Brand: {brand}\n"
        "Color: {color}\n"
        "Material: {material}\n"
        "Style: {style}\n"
        "Dimensions: {dimensions}\n"
        "Description: {description}\n"
        "Features: {bullet_points}"
    )
    filled = {
        k: (row.get(k) or "unknown")
        for k in ["title", "category_path", "product_type", "brand", "color",
                   "material", "style", "dimensions", "description",
                   "bullet_points"]
    }
    return template.format(**filled)


def truncate_tokens(text, max_tokens=384):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def parse_listings_to_rows(listings_dir, images_metadata_dir, small_images_dir,
                            keywords, stats):
    image_meta = load_image_metadata(images_metadata_dir)
    rows = []
    for item in load_listings(listings_dir):
        stats["raw_scanned"] += 1

        title = _pick_english(item.get("item_name"))
        if not title:
            continue
        stats["has_english_title"] += 1

        main_image_id = item.get("main_image_id")
        if not main_image_id:
            continue
        stats["has_main_image_id"] += 1

        product_type_list = item.get("product_type") or []
        product_type = product_type_list[0]["value"] if product_type_list else None
        category_path = _category_path(item)
        if not (category_path or product_type):
            continue
        if not _matches_home_category(category_path, product_type, keywords):
            continue
        stats["in_home_category"] += 1

        img = image_meta.get(main_image_id)
        if img is None:
            continue
        image_path = os.path.join(small_images_dir, img["path"])
        if not os.path.exists(image_path):
            continue
        stats["after_image_join"] += 1

        brand = _pick_english(item.get("brand"))
        color = _pick_english(item.get("color"))
        material = _pick_english(item.get("material"))
        style = _pick_english(item.get("style"))
        dims = item.get("item_dimensions")
        dimensions = None
        if dims:
            parts = []
            for axis in ("height", "width", "length"):
                v = dims.get(axis)
                if v and v.get("normalized_value"):
                    parts.append(f"{v['normalized_value'].get('value')}{v['normalized_value'].get('unit', '')}")
            dimensions = " x ".join(parts) if parts else None
        description = _clean_html(_pick_english(item.get("product_description")))
        bullets = _clean_html(_pick_english_list(item.get("bullet_point"), sep="; "))

        item_id = item["item_id"]
        domain_name = item.get("domain_name", "amazon.com")
        product_id = f"{item_id}|{domain_name}"

        row = {
            "product_id": product_id,
            "item_id": item_id,
            "domain_name": domain_name,
            "title": title,
            "brand": brand,
            "product_type": product_type,
            "category_path": category_path,
            "color": color,
            "material": material,
            "style": style,
            "dimensions": dimensions,
            "description": description,
            "bullet_points": bullets or "",
            "image_id": main_image_id,
            "image_path": image_path,
        }
        rows.append(row)
    return rows, stats


def stratified_sample(rows, per_type_cap, max_products, seed):
    import random
    rng = random.Random(seed)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["product_type"] or "UNKNOWN", []).append(r)
    for values in by_type.values():
        rng.shuffle(values)

    sampled = []
    type_order = sorted(by_type)
    rng.shuffle(type_order)
    cursor = {t: 0 for t in type_order}
    while len(sampled) < max_products and any(
        cursor[t] < min(per_type_cap, len(by_type[t])) for t in type_order
    ):
        progressed = False
        for t in type_order:
            if len(sampled) >= max_products:
                break
            if cursor[t] < min(per_type_cap, len(by_type[t])):
                sampled.append(by_type[t][cursor[t]])
                cursor[t] += 1
                progressed = True
        if not progressed:
            break
    return sampled


def main():
    ap = argparse.ArgumentParser(description="Stage 0: build ABO-Home-2K catalog")
    ap.add_argument("--listings_dir", default="data/raw/listings/metadata")
    ap.add_argument("--images_metadata_dir", default="data/raw/images/metadata")
    ap.add_argument("--small_images_dir", default="data/raw/images/small")
    ap.add_argument("--category_mode", default="home", choices=["home"])
    ap.add_argument("--max_products", type=int, default=2000)
    ap.add_argument("--per_type_cap", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_text_tokens", type=int, default=384)
    ap.add_argument("--out", default="data/catalog_subset.parquet")
    ap.add_argument("--stats_out", default="data/prep_stats.json")
    ap.add_argument("--report_out", default="reports/data_preparation_summary.md")
    args = ap.parse_args()

    stats = Counter()
    rows, stats = parse_listings_to_rows(
        args.listings_dir, args.images_metadata_dir, args.small_images_dir,
        HOME_CATEGORY_KEYWORDS, stats,
    )

    seen = set()
    deduped = []
    for r in rows:
        if r["product_id"] in seen:
            continue
        seen.add(r["product_id"])
        deduped.append(r)
    stats["after_dedup_item_id"] = len(deduped)

    sampled = stratified_sample(deduped, args.per_type_cap, args.max_products, args.seed)

    final_rows = []
    for r in sampled:
        r["product_text"] = truncate_tokens(build_product_text(r), args.max_text_tokens)
        r["has_image"] = os.path.exists(r["image_path"])
        r["text_length"] = len(r["product_text"])
        final_rows.append(r)

    df = pd.DataFrame(final_rows)
    df = df[df["has_image"]].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    df.to_csv(args.out.replace(".parquet", ".csv"), index=False)

    stats_out = dict(stats)
    stats_out.update({
        "num_final_products": len(df),
        "num_product_types_final": int(df["product_type"].nunique()),
        "seed": args.seed,
        "per_type_cap": args.per_type_cap,
        "max_products": args.max_products,
    })
    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
    with open(args.stats_out, "w") as fh:
        json.dump(stats_out, fh, indent=2)

    write_report(df, stats_out, args.report_out)

    print(f"Wrote {len(df)} products to {args.out}")
    run_stage0(df)


def write_report(df, stats, report_path):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    missing_title_rate = (df["title"].fillna("").str.len() == 0).mean()
    missing_image_rate = 1 - df["has_image"].mean()
    type_counts = df["product_type"].value_counts()
    examples = df.sample(min(10, len(df)), random_state=42)[
        ["product_id", "title", "product_type", "category_path"]
    ]

    lines = [
        "# Stage 0 — Data Preparation Summary\n",
        f"- Raw products scanned: {stats.get('raw_scanned', 'n/a')}",
        f"- Products with an English title: {stats.get('has_english_title', 'n/a')}",
        f"- Products with a main_image_id: {stats.get('has_main_image_id', 'n/a')}",
        f"- Products in the home/furniture/household category set: {stats.get('in_home_category', 'n/a')}",
        f"- Products after joining with image metadata: {stats.get('after_image_join', 'n/a')}",
        f"- Products after de-duplicating product_id: {stats.get('after_dedup_item_id', 'n/a')}",
        f"- **Final catalog size: {len(df)}**",
        f"- Unique product types in final catalog: {df['product_type'].nunique()}",
        f"- Missing title rate: {missing_title_rate:.4%}",
        f"- Missing image rate: {missing_image_rate:.4%}",
        f"- Average product_text length (chars): {df['text_length'].mean():.1f}",
        f"- Median product_text length (chars): {df['text_length'].median():.1f}",
        "",
        "## product_type distribution (top 20)",
        "",
        "| product_type | count |",
        "|---|---:|",
    ]
    for pt, cnt in type_counts.head(20).items():
        lines.append(f"| {pt} | {cnt} |")

    lines += ["", "## 10 example final products", "",
              "| product_id | title | product_type | category_path |",
              "|---|---|---|---|"]
    for _, r in examples.iterrows():
        title = (r["title"] or "")[:70].replace("|", "/")
        lines.append(f"| {r['product_id']} | {title} | {r['product_type']} | {r['category_path']} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Only the ABO small-image archive (largest axis <= 256px) and the "
        "listings/image metadata archives were used.",
        "- The full-resolution image archive, 360-degree image archive, and "
        "3D model archive were **not** used, as required for the base project.",
    ]
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
