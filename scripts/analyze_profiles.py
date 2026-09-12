#!/usr/bin/env python3
"""Analyze a batch of DeepPersona profiles for demographic distribution and
attribute-taxonomy coverage.

Reads output/profile_ind.json (the file generate_profile.py writes) and
reports:

  DEMOGRAPHICS  — age, gender, country, city, occupation distributions.
    Requires the Base Info fix in generate_single_profile (upstream deleted
    'Base Info' from the saved profile; without that fix this section is
    empty for every profile).

  ATTRIBUTE COVERAGE — of the ~2,300 leaf attributes in the taxonomy, how many
    were ever selected across the batch, which are over/under-represented, and
    how selections break down by top-level category.
    Requires the '_selected_attributes' field added to each profile.

  CATEGORY LEAKAGE — the taxonomy has several near-duplicate top-level
    categories (e.g. four "Core Values..." variants). generate_profile.py only
    writes content for 7 canonical section names, so attributes selected under
    a duplicate category get silently dropped. This section quantifies exactly
    how much of your attribute budget that leakage is costing you.

  SECTION FILL RATE — how often each of the 7 canonical sections actually
    produced content vs. was skipped for lack of a matching template. This is
    the visible symptom of the leakage above.

Usage:
    python scripts/analyze_profiles.py
    python scripts/analyze_profiles.py --input output/profile_ind.json --outdir output/analysis
    python scripts/analyze_profiles.py --no-charts   # numbers only, no matplotlib dependency
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CANONICAL_SECTIONS = [
    "Demographic Information",
    "Career and Work Identity",
    "Core Values, Beliefs, and Philosophy",
    "Lifestyle and Daily Routine",
    "Cultural and Social Context",
    "Hobbies, Interests, and Lifestyle",
    "Other Attributes",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_profiles(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [v for k, v in data.items() if k != "metadata" and isinstance(v, dict)]


def flatten_taxonomy(attributes: Dict) -> List[str]:
    """Same flattening rule used throughout the pipeline: a leaf is an empty
    dict or a non-dict value."""
    result = []

    def _walk(node, prefix=""):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                if not value:
                    result.append(path)
                else:
                    _walk(value, path)
            else:
                result.append(path)

    _walk(attributes)
    return result


def load_taxonomy_categories(taxonomy_path: str) -> Tuple[Dict[str, int], List[str]]:
    """Return per-category leaf counts and every fully qualified leaf path."""
    if not os.path.exists(taxonomy_path):
        return {}, []
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        attributes = json.load(f)
    per_category = {}
    all_leaves = []
    for category, subtree in attributes.items():
        leaves = flatten_taxonomy({category: subtree})
        per_category[category] = len(leaves)
        all_leaves.extend(leaves)
    return per_category, all_leaves


def top_level(path: str) -> str:
    return path.split(".", 1)[0]


# --------------------------------------------------------------------------
# Demographics
# --------------------------------------------------------------------------

def analyze_demographics(profiles: List[Dict]) -> Dict:
    ages, age_groups, genders, countries, cities, occupations = [], [], [], [], [], []
    missing_base_info = 0

    for p in profiles:
        base = p.get("Base Info")
        if not base:
            missing_base_info += 1
            continue
        age_info = base.get("age_info", {})
        if "age" in age_info:
            ages.append(age_info["age"])
        if "age_group" in age_info:
            age_groups.append(age_info["age_group"])
        if base.get("gender"):
            genders.append(base["gender"])
        location = base.get("location", {})
        if location.get("country"):
            countries.append(location["country"])
        if location.get("city"):
            cities.append(location["city"])
        career = base.get("career_info", {})
        if career.get("status"):
            occupations.append(career["status"])

    n = len(profiles)
    result = {
        "profiles_total": n,
        "profiles_missing_base_info": missing_base_info,
    }
    if missing_base_info == n and n > 0:
        result["warning"] = (
            "Every profile is missing 'Base Info'. This is expected if your "
            "generate_profile.py still deletes it before returning — apply the "
            "Base Info preservation fix, then regenerate."
        )
        return result

    if ages:
        result["age"] = {
            "min": min(ages), "max": max(ages),
            "mean": round(sum(ages) / len(ages), 1),
            "histogram_10y": dict(sorted(Counter(a - (a % 10) for a in ages).items())),
        }
    result["age_group"] = dict(Counter(age_groups).most_common())
    result["gender"] = dict(Counter(genders).most_common())
    result["country"] = {
        "unique_countries": len(set(countries)),
        "top_20": Counter(countries).most_common(20),
    }
    result["city"] = {
        "unique_cities": len(set(cities)),
        "top_20": Counter(cities).most_common(20),
    }
    result["occupation"] = {
        "unique_occupations": len(set(occupations)),
        "top_20": Counter(occupations).most_common(20),
    }
    return result


# --------------------------------------------------------------------------
# Attribute coverage + category leakage
# --------------------------------------------------------------------------

def analyze_attribute_coverage(profiles: List[Dict], taxonomy_path: str) -> Dict:
    per_category_size, taxonomy_leaves = load_taxonomy_categories(taxonomy_path)
    taxonomy_leaf_set = set(taxonomy_leaves)
    total_leaves = len(taxonomy_leaf_set)

    all_selected: List[str] = []
    per_profile_counts = []
    missing_selection_data = 0

    for p in profiles:
        sel = p.get("_selected_attributes")
        if sel is None:
            missing_selection_data += 1
            continue
        all_selected.extend(sel)
        per_profile_counts.append(len(sel))

    result = {
        "profiles_total": len(profiles),
        "profiles_missing_selection_data": missing_selection_data,
    }
    if missing_selection_data == len(profiles) and profiles:
        result["warning"] = (
            "No profile has '_selected_attributes'. This requires the "
            "instrumentation added to generate_single_profile — regenerate "
            "with the updated script to get coverage data."
        )
        return result

    freq = Counter(all_selected)
    unique_selected = len(freq)

    result["taxonomy_total_leaves"] = total_leaves
    result["unique_leaves_selected"] = unique_selected
    result["coverage_pct"] = (
        round(100 * unique_selected / total_leaves, 1) if total_leaves else None
    )
    result["avg_attributes_per_profile"] = (
        round(sum(per_profile_counts) / len(per_profile_counts), 1)
        if per_profile_counts else None
    )
    result["most_selected"] = freq.most_common(20)
    result["least_selected_observed"] = sorted(freq.items(), key=lambda kv: (kv[1], kv[0]))[:20]

    # A coverage report must include attributes selected zero times.  The old
    # script only reported the least frequent *observed* attributes, which hid
    # the largest coverage gap in every run.
    unselected = sorted(taxonomy_leaf_set - set(freq))
    unknown_selected = sorted(set(freq) - taxonomy_leaf_set) if taxonomy_leaf_set else []
    result["never_selected_count"] = len(unselected)
    result["never_selected_sample"] = unselected[:100]
    result["selected_paths_missing_from_taxonomy"] = unknown_selected[:100]
    result["selected_paths_missing_from_taxonomy_count"] = len(unknown_selected)
    if not taxonomy_leaf_set:
        result["taxonomy_warning"] = (
            f"Taxonomy not found or contains no leaves: {taxonomy_path}. "
            "Demographic and section-fill analysis is valid, but attribute "
            "coverage cannot be calculated."
        )

    # Per-category selection vs category size in the taxonomy.
    cat_selected = Counter(top_level(p) for p in all_selected)
    per_category = {}
    for cat, size in per_category_size.items():
        chosen = cat_selected.get(cat, 0)
        per_category[cat] = {
            "taxonomy_leaves": size,
            "times_selected_total": chosen,
            "pct_of_all_selections": round(100 * chosen / len(all_selected), 1) if all_selected else 0,
        }
    # Categories that appear in selections but not in the loaded taxonomy file
    # (shouldn't happen, but surfaces a taxonomy/version mismatch if it does).
    unknown_cats = set(cat_selected) - set(per_category_size)
    for cat in unknown_cats:
        per_category[cat] = {
            "taxonomy_leaves": None,
            "times_selected_total": cat_selected[cat],
            "pct_of_all_selections": round(100 * cat_selected[cat] / len(all_selected), 1),
        }
    result["per_category"] = dict(sorted(per_category.items(), key=lambda kv: -kv[1]["times_selected_total"]))

    # Category leakage: selections whose top-level category name isn't one of
    # the 7 canonical section names generate_profile.py writes output under.
    canonical = set(CANONICAL_SECTIONS)
    leaked = [p for p in all_selected if top_level(p) not in canonical]
    leaked_by_category = Counter(top_level(p) for p in leaked)
    result["leakage"] = {
        "description": (
            "Attributes selected under a category name that doesn't match one "
            "of the 7 canonical output sections. These get chosen and counted "
            "toward your --attribute-count, then silently dropped when "
            "generate_single_profile looks up selected_paths.get(<canonical name>)."
        ),
        "canonical_sections": CANONICAL_SECTIONS,
        "total_selected": len(all_selected),
        "leaked_count": len(leaked),
        "leaked_pct": round(100 * len(leaked) / len(all_selected), 1) if all_selected else 0,
        "leaked_by_category": leaked_by_category.most_common(),
    }
    return result


# --------------------------------------------------------------------------
# Section fill rate — the visible symptom of leakage
# --------------------------------------------------------------------------

def analyze_section_fill_rate(profiles: List[Dict]) -> Dict:
    n = len(profiles)
    result = {}
    for section in CANONICAL_SECTIONS:
        filled = sum(1 for p in profiles if p.get(section))
        result[section] = {
            "filled": filled,
            "total": n,
            "fill_rate_pct": round(100 * filled / n, 1) if n else None,
        }
    return result


# --------------------------------------------------------------------------
# Charts (optional — falls back gracefully without matplotlib)
# --------------------------------------------------------------------------

def make_charts(demographics: Dict, coverage: Dict, fill_rate: Dict, outdir: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping charts (numbers are still in summary.json). "
              "Install with: pip install matplotlib")
        return []

    os.makedirs(outdir, exist_ok=True)
    written = []

    def _save(fig, name):
        path = os.path.join(outdir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    if "age" in demographics:
        hist = demographics["age"]["histogram_10y"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([f"{k}s" for k in hist.keys()], hist.values(), color="#4C72B0")
        ax.set_title("Age distribution (10-year buckets)")
        ax.set_ylabel("profiles")
        _save(fig, "age_distribution.png")

    if demographics.get("gender"):
        fig, ax = plt.subplots(figsize=(4, 4))
        genders = demographics["gender"]
        ax.bar(genders.keys(), genders.values(), color="#DD8452")
        ax.set_title("Gender distribution")
        _save(fig, "gender_distribution.png")

    if demographics.get("country", {}).get("top_20"):
        top = demographics["country"]["top_20"][:15]
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = [t[0] for t in top][::-1]
        values = [t[1] for t in top][::-1]
        ax.barh(labels, values, color="#55A868")
        ax.set_title("Top countries")
        _save(fig, "country_distribution.png")

    if coverage.get("per_category"):
        items = list(coverage["per_category"].items())[:15]
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = [k[:28] for k, _ in items][::-1]
        values = [v["pct_of_all_selections"] for _, v in items][::-1]
        ax.barh(labels, values, color="#8172B2")
        ax.set_title("Selected attributes by category (% of all selections)")
        ax.set_xlabel("%")
        _save(fig, "category_selection_share.png")

    if fill_rate:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [k[:28] for k in fill_rate.keys()]
        values = [v["fill_rate_pct"] or 0 for v in fill_rate.values()]
        colors = ["#55A868" if v >= 90 else "#DD8452" if v >= 50 else "#C44E52" for v in values]
        ax.barh(labels, values, color=colors)
        ax.set_title("Canonical section fill rate (% of profiles with content)")
        ax.set_xlabel("%")
        ax.set_xlim(0, 100)
        _save(fig, "section_fill_rate.png")

    return written


# --------------------------------------------------------------------------
# Report printing
# --------------------------------------------------------------------------

def print_report(demographics: Dict, coverage: Dict, fill_rate: Dict) -> None:
    print("=" * 72)
    print("DEMOGRAPHICS")
    print("=" * 72)
    if "warning" in demographics:
        print(f"  WARNING: {demographics['warning']}")
    else:
        print(f"  Profiles analyzed: {demographics['profiles_total']}")
        if "age" in demographics:
            a = demographics["age"]
            print(f"  Age: min={a['min']} max={a['max']} mean={a['mean']}")
        print(f"  Gender: {demographics.get('gender')}")
        c = demographics.get("country", {})
        print(f"  Countries: {c.get('unique_countries')} unique. Top 5: {c.get('top_20', [])[:5]}")
        o = demographics.get("occupation", {})
        print(f"  Occupations: {o.get('unique_occupations')} unique. Top 5: {o.get('top_20', [])[:5]}")

    print()
    print("=" * 72)
    print("ATTRIBUTE COVERAGE")
    print("=" * 72)
    if "warning" in coverage:
        print(f"  WARNING: {coverage['warning']}")
    else:
        if coverage.get("taxonomy_warning"):
            print(f"  WARNING: {coverage['taxonomy_warning']}")
        print(f"  Taxonomy leaves: {coverage['taxonomy_total_leaves']}")
        print(f"  Unique leaves ever selected: {coverage['unique_leaves_selected']} "
              f"({coverage['coverage_pct']}% coverage)")
        print(f"  Avg attributes per profile: {coverage['avg_attributes_per_profile']}")
        print(f"  Most selected (top 5): {coverage['most_selected'][:5]}")
        print(f"  Never selected: {coverage['never_selected_count']} "
              f"(sample of up to 10: {coverage['never_selected_sample'][:10]})")
        if coverage["selected_paths_missing_from_taxonomy_count"]:
            print("  WARNING: selected paths not found in the supplied taxonomy: "
                  f"{coverage['selected_paths_missing_from_taxonomy_count']}")
        leak = coverage["leakage"]
        print(f"\n  CATEGORY LEAKAGE: {leak['leaked_count']}/{leak['total_selected']} "
              f"selections ({leak['leaked_pct']}%) fall under a non-canonical category "
              f"and are silently dropped from output.")
        if leak["leaked_by_category"]:
            print(f"  Leaked by category: {leak['leaked_by_category']}")

    print()
    print("=" * 72)
    print("SECTION FILL RATE (canonical output sections)")
    print("=" * 72)
    for section, stats in fill_rate.items():
        print(f"  {section:42} {stats['filled']:>4}/{stats['total']:<4} "
              f"({stats['fill_rate_pct']}%)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join(REPO_ROOT, "output", "profile_ind.json"))
    parser.add_argument("--taxonomy", default=os.path.join(REPO_ROOT, "data", "large_attributes.json"))
    parser.add_argument("--outdir", default=os.path.join(REPO_ROOT, "output", "analysis"))
    parser.add_argument("--no-charts", action="store_true", help="Skip matplotlib charts.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run generate_profile.py first.")
        return 1

    profiles = load_profiles(args.input)
    if not profiles:
        print(f"ERROR: no profiles found in {args.input}.")
        return 1

    demographics = analyze_demographics(profiles)
    coverage = analyze_attribute_coverage(profiles, args.taxonomy)
    fill_rate = analyze_section_fill_rate(profiles)

    print_report(demographics, coverage, fill_rate)

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_profiles": len(profiles),
            "demographics": demographics,
            "attribute_coverage": coverage,
            "section_fill_rate": fill_rate,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nFull summary written to {summary_path}")

    if not args.no_charts:
        charts = make_charts(demographics, coverage, fill_rate, args.outdir)
        if charts:
            print(f"Charts written to {args.outdir}:")
            for c in charts:
                print(f"  {c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
