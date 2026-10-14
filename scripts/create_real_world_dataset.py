from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_processing.nup_real_world_enrichment import (
    DEFAULT_CAMERA_ROOT,
    DEFAULT_DB_ROOT,
    DEFAULT_MAP_ROOT,
    copy_subset_with_real_world_context,
    format_graph_summary,
    resolve_dataset_path,
    subset_jobs,
)


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('must be an integer') from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create a separate NuPlan dataset copy enriched with real_world_context.')
    parser.add_argument('--input-root', required=True, help='Subset root containing graphs/, or dataset root containing subset directories.')
    parser.add_argument('--output-root', required=True, help='Output subset root, or output dataset root when --input-root is a dataset root.')
    parser.add_argument('--subsets', nargs='*', default=None, help='Subset names to mirror when --input-root is a dataset root. Defaults to clean if available.')
    parser.add_argument('--db-root', default=str(DEFAULT_DB_ROOT), help='Directory containing raw NuPlan .db files.')
    parser.add_argument('--camera-root', default=str(DEFAULT_CAMERA_ROOT), help='Directory containing NuPlan camera folders.')
    parser.add_argument('--map-root', default=str(DEFAULT_MAP_ROOT), help='Directory containing nuplan-maps-v1.0.json and map folders.')
    parser.add_argument('--camera-channel', default='CAM_F0', help='Camera channel used for nearest image matching.')
    parser.add_argument('--composition-radius', type=float, default=30.0, help='HD-map query radius in meters.')
    parser.add_argument('--limit', type=positive_int, default=None, help='Sample mode: process at most N graph JSONs per subset.')
    parser.add_argument('--graph-ids', nargs='*', default=None, help='Explicit graph ids, e.g. 10428 10429. Cannot be combined with --limit.')
    parser.add_argument('--graph-files', nargs='*', default=None, help='Explicit graph JSON filenames, e.g. 10428_graph.json. Cannot be combined with --limit.')
    parser.add_argument('--overwrite', action='store_true', help='Allow writing into an existing non-empty output root.')
    parser.add_argument('--no-weather', action='store_true', help='Skip Open-Meteo weather lookup.')
    args = parser.parse_args()
    if args.limit is not None and (args.graph_ids or args.graph_files):
        parser.error('--limit cannot be combined with --graph-ids or --graph-files')
    return args


def main() -> None:
    args = parse_args()
    input_root = resolve_dataset_path(Path(args.input_root), REPO_ROOT)
    output_root = resolve_dataset_path(Path(args.output_root), REPO_ROOT)
    jobs = subset_jobs(input_root, output_root, args.subsets, repo_root=REPO_ROOT)
    explicit_graphs = []
    if args.graph_ids:
        explicit_graphs.extend(args.graph_ids)
    if args.graph_files:
        explicit_graphs.extend(args.graph_files)
    explicit_graphs = explicit_graphs or None
    combined = Counter()
    reports = []

    for in_subset, out_subset in jobs:
        print(f'=== Enriching {in_subset} -> {out_subset} ===')
        report = copy_subset_with_real_world_context(
            input_root=in_subset,
            output_root=out_subset,
            db_root=Path(args.db_root),
            camera_root=Path(args.camera_root),
            map_root=Path(args.map_root),
            camera_channel=args.camera_channel,
            composition_radius_m=args.composition_radius,
            include_weather=not args.no_weather,
            limit=args.limit,
            graph_files=explicit_graphs,
            overwrite=args.overwrite,
            repo_root=REPO_ROOT,
        )
        reports.append(report)
        combined.update(report.get('counts', {}))
        print(json.dumps(report.get('counts', {}), indent=2, sort_keys=True))
        if report.get('sample_run'):
            print('Selected graph files:')
            for name in report.get('selected_filenames', []):
                print(f'  {name}')
            print('Graph summaries:')
            for graph_summary in report.get('graph_summaries', []):
                print(format_graph_summary(graph_summary))

    summary = {'jobs': [{'input_root': r['input_root'], 'output_root': r['output_root'], 'limit': r.get('limit'), 'available_graphs': r.get('available_graphs'), 'selected_graphs': r.get('selected_graphs'), 'counts': r['counts']} for r in reports], 'counts': dict(combined)}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'real_world_enrichment_summary.json').write_text(json.dumps(summary, indent=2))
    print('=== Combined summary ===')
    print(json.dumps(summary['counts'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
