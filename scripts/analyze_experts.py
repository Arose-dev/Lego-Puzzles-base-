"""
Analyze MoE expert routing per LEGO category.

Usage:
    python scripts/analyze_experts.py \
        --routing outputs/Qwen3-VL-30B-A3B-Instruct/<eval_id>/Qwen3-VL-30B-A3B-Instruct_LEGO_expert_routing.json \
        --top 10
"""

import argparse
import glob
import json
from collections import defaultdict, Counter


def load_routing(path):
    matches = glob.glob(path, recursive=True)
    if not matches:
        raise FileNotFoundError(f'No file found matching: {path}')
    if len(matches) > 1:
        print(f'Multiple matches found, using: {matches[-1]}')
    with open(matches[-1]) as f:
        return json.load(f)


def aggregate_by_category(data):
    """Return {category: Counter(expert_id -> total_activations)} aggregated across all layers."""
    cat_experts = defaultdict(Counter)
    for entry in data.values():
        cat = entry['category']
        for layer_counts in entry['layer_experts'].values():
            for expert_id, count in layer_counts.items():
                cat_experts[cat][int(expert_id)] += count
    return cat_experts


def aggregate_by_category_and_layer(data):
    """Return {category: {layer: Counter(expert_id -> total_activations)}}."""
    result = defaultdict(lambda: defaultdict(Counter))
    for entry in data.values():
        cat = entry['category']
        for layer, counts in entry['layer_experts'].items():
            for expert_id, count in counts.items():
                result[cat][layer][int(expert_id)] += count
    return result


def print_top_experts(cat_experts, top_k):
    categories = sorted(cat_experts.keys())
    print(f"\n{'='*60}")
    print(f"Top-{top_k} experts per category (aggregated across all layers)")
    print(f"{'='*60}")
    for cat in categories:
        counter = cat_experts[cat]
        total = sum(counter.values())
        top = counter.most_common(top_k)
        top_str = ', '.join(f'E{eid}({cnt/total:.1%})' for eid, cnt in top)
        print(f"  {cat:<20} {top_str}")


def print_layer_breakdown(cat_layer_experts, category, top_k):
    print(f"\n{'='*60}")
    print(f"Layer-by-layer expert routing for: {category}")
    print(f"{'='*60}")
    layers = sorted(cat_layer_experts[category].keys())
    for layer in layers:
        counter = cat_layer_experts[category][layer]
        total = sum(counter.values())
        top = counter.most_common(top_k)
        top_str = ', '.join(f'E{eid}({cnt/total:.1%})' for eid, cnt in top)
        short_layer = layer.split('.')[-3] if layer.count('.') >= 3 else layer
        print(f"  {short_layer:<30} {top_str}")


def save_csv(cat_experts, out_path):
    all_experts = sorted({eid for c in cat_experts.values() for eid in c})
    categories = sorted(cat_experts.keys())
    rows = [['category'] + [f'expert_{e}' for e in all_experts]]
    for cat in categories:
        total = sum(cat_experts[cat].values()) or 1
        row = [cat] + [f"{cat_experts[cat].get(e, 0) / total:.4f}" for e in all_experts]
        rows.append(row)
    with open(out_path, 'w') as f:
        for row in rows:
            f.write(','.join(map(str, row)) + '\n')
    print(f"\nCSV saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--routing', required=True, help='Path to *_expert_routing.json')
    parser.add_argument('--top', type=int, default=5, help='Top-K experts to show per category')
    parser.add_argument('--layer-breakdown', type=str, default=None,
                        help='Print layer-by-layer breakdown for a specific category')
    parser.add_argument('--save-csv', type=str, default=None, help='Save expert usage matrix as CSV')
    args = parser.parse_args()

    data = load_routing(args.routing)
    print(f"Loaded {len(data)} questions")

    cat_counts = Counter(e['category'] for e in data.values())
    print("\nQuestions per category:")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat:<20} {n}")

    cat_experts = aggregate_by_category(data)
    print_top_experts(cat_experts, args.top)

    if args.layer_breakdown:
        cat_layer = aggregate_by_category_and_layer(data)
        if args.layer_breakdown in cat_layer:
            print_layer_breakdown(cat_layer, args.layer_breakdown, args.top)
        else:
            print(f"\nCategory '{args.layer_breakdown}' not found. Available: {sorted(cat_layer.keys())}")

    if args.save_csv:
        save_csv(cat_experts, args.save_csv)


if __name__ == '__main__':
    main()
