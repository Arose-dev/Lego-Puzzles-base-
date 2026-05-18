"""
Analyze MoE expert routing per LEGO category.

Usage:
    python scripts/analyze_experts.py \
        --routing outputs/Qwen3-VL-30B-A3B-Instruct/<eval_id>/01_LEGO_expert_routing.json \
        --top 10

    # Compare correct vs incorrect activations and export flat CSV:
    python scripts/analyze_experts.py \
        --routing outputs/.../01_LEGO_Lite_expert_routing.json \
        --top 10 --correctness --save-correctness-csv out.csv
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


def _layer_key(layer):
    short = layer.split('.')[-3] if layer.count('.') >= 3 else layer
    try:
        return int(short)
    except ValueError:
        return short


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


def aggregate_by_correctness(data):
    """Return {True: Counter, False: Counter} aggregated across all questions and layers."""
    result = {True: Counter(), False: Counter(), None: Counter()}
    for entry in data.values():
        correct = entry.get('correct', None)
        for layer_counts in entry['layer_experts'].values():
            for expert_id, count in layer_counts.items():
                result[correct][int(expert_id)] += count
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
        print(f"\n  [{cat}]  (total activations: {total:,})")
        print(f"  {'Rank':<6} {'Expert':>8} {'Activations':>14} {'Share':>8}")
        print(f"  {'-'*40}")
        for rank, (eid, cnt) in enumerate(top, 1):
            print(f"  {rank:<6} {'E' + str(eid):>8} {cnt:>14,} {cnt/total:>7.2%}")


def print_correctness_comparison(data, top_k):
    by_correct = aggregate_by_correctness(data)
    correct_ctr = by_correct[True]
    wrong_ctr = by_correct[False]

    has_correctness = any(e.get('correct') is not None for e in data.values())
    if not has_correctness:
        print("\n[!] No correctness data found — re-run inference with the updated code.")
        return

    n_correct = sum(1 for e in data.values() if e.get('correct') is True)
    n_wrong = sum(1 for e in data.values() if e.get('correct') is False)

    print(f"\n{'='*60}")
    print(f"Correct vs Incorrect expert routing  (correct={n_correct}, wrong={n_wrong})")
    print(f"{'='*60}")

    for label, ctr in [('CORRECT', correct_ctr), ('INCORRECT', wrong_ctr)]:
        total = sum(ctr.values()) or 1
        top = ctr.most_common(top_k)
        top10_share = sum(cnt for _, cnt in ctr.most_common(10)) / total
        print(f"\n  [{label}]  (total activations: {total:,}, top-10 share: {top10_share:.1%})")
        print(f"  {'Rank':<6} {'Expert':>8} {'Activations':>14} {'Share':>8}")
        print(f"  {'-'*40}")
        for rank, (eid, cnt) in enumerate(top, 1):
            print(f"  {rank:<6} {'E' + str(eid):>8} {cnt:>14,} {cnt/total:>7.2%}")

    correct_top10 = {eid for eid, _ in correct_ctr.most_common(10)}
    wrong_top10 = {eid for eid, _ in wrong_ctr.most_common(10)}
    overlap = correct_top10 & wrong_top10
    print(f"\n  Top-10 overlap: {len(overlap)}/10 experts shared")
    if correct_top10 - wrong_top10:
        print(f"  Correct-only:   {sorted('E'+str(e) for e in correct_top10 - wrong_top10)}")
    if wrong_top10 - correct_top10:
        print(f"  Wrong-only:     {sorted('E'+str(e) for e in wrong_top10 - correct_top10)}")


def print_layer_breakdown(cat_layer_experts, category, top_k):
    print(f"\n{'='*60}")
    print(f"Layer-by-layer expert routing for: {category}")
    print(f"{'='*60}")
    layers = sorted(cat_layer_experts[category].keys(), key=_layer_key)
    for layer in layers:
        counter = cat_layer_experts[category][layer]
        total = sum(counter.values())
        top = counter.most_common(top_k)
        short_layer = layer.split('.')[-3] if layer.count('.') >= 3 else layer
        print(f"\n  [{short_layer}]  (total: {total:,})")
        print(f"  {'Rank':<6} {'Expert':>8} {'Activations':>14} {'Share':>8}")
        print(f"  {'-'*40}")
        for rank, (eid, cnt) in enumerate(top, 1):
            print(f"  {rank:<6} {'E' + str(eid):>8} {cnt:>14,} {cnt/total:>7.2%}")


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


def save_correctness_csv(data, out_path):
    """Write flat CSV: question_id, correct, layer, expert, activations"""
    with open(out_path, 'w') as f:
        f.write('question_id,correct,layer,expert,activations\n')
        for qid, entry in sorted(data.items(), key=lambda x: int(x[0])):
            correct = entry.get('correct', '')
            for layer, counts in entry['layer_experts'].items():
                layer_idx = layer.split('.')[-3] if layer.count('.') >= 3 else layer
                for expert_id, count in counts.items():
                    f.write(f"{qid},{correct},{layer_idx},{expert_id},{count}\n")
    print(f"\nCorrectness CSV saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--routing', required=True, help='Path to *_expert_routing.json')
    parser.add_argument('--top', type=int, default=20, help='Top-K experts to show per category')
    parser.add_argument('--layer-breakdown', type=str, default=None,
                        help='Print layer-by-layer breakdown for a specific category')
    parser.add_argument('--correctness', action='store_true',
                        help='Compare expert activations for correct vs incorrect answers')
    parser.add_argument('--save-csv', type=str, default=None, help='Save expert usage matrix as CSV')
    parser.add_argument('--save-correctness-csv', type=str, default=None,
                        help='Save flat CSV: question_id, correct, layer, expert, activations')
    args = parser.parse_args()

    data = load_routing(args.routing)
    print(f"Loaded {len(data)} questions")

    cat_counts = Counter(e['category'] for e in data.values())
    print("\nQuestions per category:")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat:<20} {n}")

    cat_experts = aggregate_by_category(data)
    print_top_experts(cat_experts, args.top)

    if args.correctness:
        print_correctness_comparison(data, args.top)

    if args.layer_breakdown:
        cat_layer = aggregate_by_category_and_layer(data)
        if args.layer_breakdown in cat_layer:
            print_layer_breakdown(cat_layer, args.layer_breakdown, args.top)
        else:
            print(f"\nCategory '{args.layer_breakdown}' not found. Available: {sorted(cat_layer.keys())}")

    if args.save_csv:
        save_csv(cat_experts, args.save_csv)

    if args.save_correctness_csv:
        save_correctness_csv(data, args.save_correctness_csv)


if __name__ == '__main__':
    main()
