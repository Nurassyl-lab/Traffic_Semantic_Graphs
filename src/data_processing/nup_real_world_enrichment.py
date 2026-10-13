from __future__ import annotations

import copy
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from test_real_coordinate import (
    NuPlanCompositionProvider,
    epsg_xy_to_latlon,
    format_db_sample_metadata,
    local_datetime_for_db_sample,
    nuplan_timestamp_us_to_utc,
    weather_for_latlon,
    yaw_from_quaternion,
)


def _token_params(token: Optional[str]) -> List[Any]:
    if not token:
        return []
    values: List[Any] = []
    try:
        values.append(bytes.fromhex(token))
    except ValueError:
        pass
    values.append(token)
    return values


def _repo_relative(path: Optional[Path], repo_root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _graph_key(path: Path) -> Tuple[int, str]:
    first = path.name.split('_', 1)[0]
    return (int(first) if first.isdigit() else 10**12, path.name)


def sorted_graph_paths(graph_dir: Path) -> List[Path]:
    return sorted(graph_dir.glob('*.json'), key=_graph_key)


def normalize_graph_filename(value: str) -> str:
    name = Path(str(value)).name
    if name.endswith('.json'):
        return name
    return f'{name}_graph.json'


def select_graph_paths(graph_dir: Path, limit: Optional[int] = None, graph_files: Optional[Sequence[str]] = None) -> Tuple[List[Path], int, List[str], str]:
    all_paths = sorted_graph_paths(graph_dir)
    available_graphs = len(all_paths)
    if graph_files:
        by_name = {path.name: path for path in all_paths}
        selected: List[Path] = []
        missing: List[str] = []
        seen = set()
        for value in graph_files:
            name = normalize_graph_filename(str(value))
            if name in seen:
                continue
            seen.add(name)
            path = by_name.get(name)
            if path is None:
                missing.append(name)
            else:
                selected.append(path)
        if missing:
            raise FileNotFoundError('Graph file(s) not found under %s: %s' % (graph_dir, missing))
        return selected, available_graphs, [path.name for path in selected], 'explicit'
    if limit is not None:
        if limit <= 0:
            raise ValueError('limit must be a positive integer')
        selected = all_paths[:limit]
        return selected, available_graphs, [path.name for path in selected], 'limit'
    return all_paths, available_graphs, [], 'all'


def camera_delta_stats(graph_summaries):
    values = []
    for item in graph_summaries:
        value = item.get('camera_delta_ms')
        if value is not None:
            values.append(float(value))
    values = sorted(values)
    result = {}
    if not values:
        result['camera_delta_ms_min'] = None
        result['camera_delta_ms_median'] = None
        result['camera_delta_ms_max'] = None
        return result
    mid = len(values) // 2
    result['camera_delta_ms_min'] = values[0]
    result['camera_delta_ms_median'] = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0
    result['camera_delta_ms_max'] = values[-1]
    return result

def summarize_graph_context(filename, source_graph, context):
    metadata = source_graph.get('metadata', {})
    source = context.get('source') or {}
    pose = context.get('pose') or {}
    time_info = context.get('time') or {}
    camera = context.get('camera') or {}
    road = context.get('road') or {}
    weather = context.get('weather')
    road_ok = bool(road) and not road.get('composition_unavailable') and road.get('status') != 'unavailable'
    weather_status = weather.get('status') if isinstance(weather, dict) else 'missing'
    result = {}
    result['graph'] = filename
    result['db'] = source.get('db_file') or metadata.get('db_file')
    result['scene'] = source.get('scene_name') or metadata.get('scene_name')
    result['scene_token'] = source.get('scene_token') or metadata.get('scene_token')
    result['anchor_source'] = 'first_lidar_pc_in_scene'
    result['anchor_timestamp_us'] = source.get('timestamp_us')
        'delta_us': delta_us,
        'delta_ms': (float(delta_us) / 1000.0) if delta_us is not None else None,
    result['ego_pose_timestamp_us'] = source.get('ego_pose_timestamp_us')
    result['latitude'] = pose.get('latitude')
    result['longitude'] = pose.get('longitude')
    result['epsg'] = pose.get('epsg')
    result['datetime_local'] = time_info.get('datetime_local')
    result['camera'] = camera.get('channel')
    result['camera_timestamp_us'] = camera.get('timestamp_us')
    result['camera_delta_ms'] = camera.get('delta_ms')
    result['camera_exists'] = camera.get('exists')
    result['image_path'] = camera.get('image_path')
    result['road'] = 'OK' if road_ok else road.get('status') or road.get('composition_unavailable') or 'missing'
    result['weather'] = 'OK' if weather_status == 'ok' else weather_status
    result['errors'] = context.get('errors', [])
    return result

def format_graph_summary(summary):
    lat = summary.get('latitude')
    lon = summary.get('longitude')
    latlon = 'None' if lat is None or lon is None else '%.8f, %.8f' % (lat, lon)
    rows = [
        str(summary.get('graph')),
        '  db: %s' % summary.get('db'),
        '  scene: %s token=%s' % (summary.get('scene'), summary.get('scene_token')),
        '  anchor_timestamp_us: %s' % summary.get('anchor_timestamp_us'),
        '  lat/lon: %s' % latlon,
        '  datetime_local: %s' % summary.get('datetime_local'),
        '  camera: %s exists=%s' % (summary.get('camera'), summary.get('camera_exists')),
        '  camera_delta_ms: %s' % summary.get('camera_delta_ms'),
        '  road: %s' % summary.get('road'),
        '  weather: %s' % summary.get('weather'),
    ]
    return chr(10).join(rows)

def validate_preserved_graph(source_graph: Dict[str, Any], output_graph: Dict[str, Any]) -> None:
    stripped = copy.deepcopy(output_graph)
    stripped.pop('real_world_context', None)
    if stripped != source_graph:
        raise AssertionError('Output graph changed after removing real_world_context')


class NuPlanRealWorldResources:
    def __init__(self, db_root: Path, camera_root: Path, map_root: Path, repo_root: Optional[Path] = None):
        self.db_root = Path(db_root)
        self.camera_root = Path(camera_root)
        self.map_root = Path(map_root)
        self.repo_root = Path(repo_root or Path.cwd())
        self._db_paths: Dict[str, Optional[Path]] = {}
        self._connections: Dict[Path, sqlite3.Connection] = {}
        self._log_info: Dict[Path, Dict[str, Any]] = {}
        self._map_providers: Dict[Tuple[Path, float], NuPlanCompositionProvider] = {}
        self._weather_cache: Dict[Tuple[float, float, str], Optional[Dict[str, Any]]] = {}

    def close(self) -> None:
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def resolve_db_path(self, db_file: Optional[str]) -> Optional[Path]:
        if not db_file:
            return None
        if db_file in self._db_paths:
            return self._db_paths[db_file]
        db_name = Path(db_file).name
        candidates = [Path(db_file), self.db_root / db_file, self.db_root / db_name]
        found = next((p for p in candidates if p.exists()), None)
        if found is None and self.db_root.exists():
            found = next(self.db_root.rglob(db_name), None)
        self._db_paths[db_file] = found
        return found

    def connect(self, db_path: Path) -> sqlite3.Connection:
        db_path = db_path.resolve()
        if db_path not in self._connections:
            self._connections[db_path] = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        return self._connections[db_path]

    def log_info(self, db_path: Path) -> Dict[str, Any]:
        db_path = db_path.resolve()
        if db_path in self._log_info:
            return self._log_info[db_path]
        row = self.connect(db_path).execute('SELECT location, map_version, logfile, date, timestamp FROM log LIMIT 1').fetchone()
        info: Dict[str, Any] = {}
        if row:
            info = {'location': row[0], 'map_version': row[1], 'logfile': row[2], 'date': row[3], 'timestamp': row[4]}
        self._log_info[db_path] = info
        return info

    def map_provider(self, db_path: Path, radius_m: float) -> NuPlanCompositionProvider:
        key = (db_path.resolve(), float(radius_m))
        if key not in self._map_providers:
            self._map_providers[key] = NuPlanCompositionProvider(db_path, str(self.map_root), radius_m)
        return self._map_providers[key]

    def resolve_image_path(self, filename: Optional[str]) -> Tuple[Optional[Path], bool]:
        if not filename:
            return None, False
        raw = Path(filename)
        candidates = [raw] if raw.is_absolute() else [self.camera_root / filename, raw]
        for candidate in candidates:
            if candidate.exists():
                return candidate, True
        return candidates[0], False

    def weather_context(self, lat: Optional[float], lon: Optional[float], date_value: Optional[str], time_value: Optional[str], enabled: bool) -> Optional[Dict[str, Any]]:
        if not enabled or lat is None or lon is None or not date_value:
            return None
        key = (round(float(lat), 4), round(float(lon), 4), date_value)
        if key in self._weather_cache:
            return self._weather_cache[key]
        raw = weather_for_latlon(float(lat), float(lon), date_value, time_value)
        if raw.startswith('weather='):
            value = raw.split('=', 1)[1]
            result = {'status': 'ok', 'description': value} if value not in {'unknown', 'unavailable'} else {'status': value}
        else:
            result = {'status': 'unknown', 'raw': raw}
        self._weather_cache[key] = result
        return result


def read_anchor_pose(conn: sqlite3.Connection, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    query = (
        'SELECT lp.timestamp, ep.timestamp, ep.x, ep.y, COALESCE(ep.z, 0.0), ep.epsg, '
        'ep.qw, ep.qx, ep.qy, ep.qz, s.name '
        'FROM lidar_pc AS lp '
        'JOIN scene AS s ON lp.scene_token = s.token '
        'JOIN ego_pose AS ep ON lp.ego_pose_token = ep.token '
        'WHERE {} '
        'ORDER BY lp.timestamp LIMIT 1'
    )
    row = None
    for token in _token_params(metadata.get('scene_token')):
        row = conn.execute(query.format('s.token = ?'), (token,)).fetchone()
        if row:
            break
    if row is None and metadata.get('scene_name'):
        row = conn.execute(query.format('s.name = ?'), (metadata['scene_name'],)).fetchone()
    if row is None:
        return None
    return {
        'lidar_timestamp_us': int(row[0]),
        'ego_pose_timestamp_us': int(row[1]),
        'x': float(row[2]),
        'y': float(row[3]),
        'z': float(row[4]),
        'epsg': int(row[5]) if row[5] is not None else None,
        'heading': yaw_from_quaternion(float(row[6]), float(row[7]), float(row[8]), float(row[9])),
        'scene_name': row[10],
    }


def read_nearest_camera(conn: sqlite3.Connection, timestamp_us: int, channel: str, resources: NuPlanRealWorldResources) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        'SELECT im.timestamp, c.channel, im.filename_jpg, ABS(im.timestamp - ?) AS dt '
        'FROM image AS im JOIN camera AS c ON im.camera_token = c.token '
        'WHERE c.channel = ? ORDER BY dt LIMIT 1',
        (int(timestamp_us), channel),
    ).fetchone()
    if row is None:
        return None
    path, exists = resources.resolve_image_path(row[2])
    delta_us = int(row[3]) if row[3] is not None else None
    return {
        'channel': row[1],
        'timestamp_us': int(row[0]),
        'anchor_timestamp_us': int(timestamp_us),
        'filename_jpg': row[2],
        'image_path': _repo_relative(path, resources.repo_root),
        'exists': bool(exists),
    }


def build_real_world_context(graph: Dict[str, Any], resources: NuPlanRealWorldResources, camera_channel: str = 'CAM_F0', composition_radius_m: float = 30.0, include_weather: bool = True) -> Tuple[Dict[str, Any], Counter]:
    counters: Counter = Counter()
    metadata = graph.get('metadata', {})
    db_path = resources.resolve_db_path(metadata.get('db_file'))
    context: Dict[str, Any] = {
        'schema_version': 1,
        'source': {
            'dataset': 'nuplan',
            'log_name': Path(metadata.get('db_file', '')).stem if metadata.get('db_file') else None,
            'db_file': metadata.get('db_file'),
            'db_path': _repo_relative(db_path, resources.repo_root) if db_path else None,
            'scene_token': metadata.get('scene_token'),
            'scene_name': metadata.get('scene_name'),
            'timestamp_us': None,
        },
        'pose': None,
        'time': None,
        'weather': None,
        'road': None,
        'camera': None,
        'errors': [],
    }
    if db_path is None:
        context['errors'].append('missing_db')
        counters['missing_db'] += 1
        return context, counters
    try:
        conn = resources.connect(db_path)
        pose = read_anchor_pose(conn, metadata)
    except sqlite3.Error as exc:
        context['errors'].append(f'db_query_failed:{type(exc).__name__}:{exc}')
        counters['missing_ego_pose'] += 1
        return context, counters
    if pose is None:
        context['errors'].append('missing_ego_pose')
        counters['missing_ego_pose'] += 1
        return context, counters

    ts = pose['lidar_timestamp_us']
    context['source']['timestamp_us'] = ts
    context['source']['ego_pose_timestamp_us'] = pose['ego_pose_timestamp_us']
    context['source']['log'] = resources.log_info(db_path)
    lat = lon = method = None
    if pose['epsg'] is None:
        context['errors'].append('missing_epsg')
        counters['missing_latlon'] += 1
    else:
        try:
            ll = epsg_xy_to_latlon(pose['x'], pose['y'], pose['epsg'])
            lat, lon, method = ll.latitude, ll.longitude, ll.method
        except Exception as exc:
            context['errors'].append(f'latlon_conversion_failed:{type(exc).__name__}:{exc}')
            counters['missing_latlon'] += 1
    context['pose'] = {'global_x': pose['x'], 'global_y': pose['y'], 'z': pose['z'], 'heading': pose['heading'], 'epsg': pose['epsg'], 'latitude': lat, 'longitude': lon, 'conversion_method': method}

    local_dt = local_datetime_for_db_sample(db_path, ts)
    utc_dt = nuplan_timestamp_us_to_utc(ts)
    if local_dt is not None:
        date_value = local_dt.date().isoformat()
        time_value = local_dt.strftime('%H:%M:%S')
        timezone_name = getattr(local_dt.tzinfo, 'key', str(local_dt.tzinfo))
        datetime_local = local_dt.isoformat()
    else:
        meta = format_db_sample_metadata(db_path, ts)
        date_value = next((x.split('=', 1)[1] for x in meta.split() if x.startswith('date=')), None)
        time_value = next((x.split('=', 1)[1] for x in meta.split() if x.startswith('time=')), None)
        timezone_name = 'unknown'
        datetime_local = None
    context['time'] = {'datetime_utc': utc_dt.isoformat(), 'datetime_local': datetime_local, 'date': date_value, 'time': time_value, 'timezone': timezone_name}

    if ts <= 0:
        context['errors'].append('invalid_timestamp')
        counters['invalid_timestamp'] += 1
    if lat is not None and not (-90 <= lat <= 90):
        context['errors'].append('invalid_latitude')
        counters['invalid_latlon'] += 1
    if lon is not None and not (-180 <= lon <= 180):
        context['errors'].append('invalid_longitude')
        counters['invalid_latlon'] += 1

    weather = resources.weather_context(lat, lon, date_value, time_value, include_weather)
    context['weather'] = weather
    if weather is None or weather.get('status') != 'ok':
        counters['missing_weather'] += 1
    try:
        provider = resources.map_provider(db_path, composition_radius_m)
        road = provider.describe(pose['x'], pose['y'], pose['heading'])
        road.setdefault('source', 'nuplan_hd_map')
        road.setdefault('map', getattr(provider, 'map_name', None))
        for key in ['nearby_lane_ids', 'nearby_lane_connector_ids', 'nearby_crosswalk_ids', 'nearby_stop_line_ids', 'nearby_intersection_ids']:
            road.setdefault(key, [])
        context['road'] = road
        if road.get('composition_unavailable') or road.get('status') == 'unavailable':
            counters['missing_road'] += 1
    except Exception as exc:
        context['road'] = {'source': 'nuplan_hd_map', 'status': 'unavailable', 'error': f'{type(exc).__name__}:{exc}'}
        counters['missing_road'] += 1

    try:
        camera = read_nearest_camera(conn, ts, camera_channel, resources)
    except sqlite3.Error as exc:
        camera = None
        context['errors'].append(f'camera_query_failed:{type(exc).__name__}:{exc}')
    if camera is None:
        counters['missing_camera'] += 1
    else:
        context['camera'] = camera
        if not camera.get('exists'):
            counters['missing_camera'] += 1
    if not context['errors']:
        context.pop('errors')
    return context, counters


def copy_subset_with_real_world_context(input_root: Path, output_root: Path, db_root: Path, camera_root: Path, map_root: Path, camera_channel: str = 'CAM_F0', composition_radius_m: float = 30.0, include_weather: bool = True, limit: Optional[int] = None, graph_files: Optional[Sequence[str]] = None, overwrite: bool = False, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    graph_dir = input_root / 'graphs'
    out_graph_dir = output_root / 'graphs'
    if not graph_dir.exists():
        raise FileNotFoundError(f'Input graph directory not found: {graph_dir}')
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f'Output root already exists and is not empty: {output_root}')
    if overwrite and out_graph_dir.exists():
        shutil.rmtree(out_graph_dir)
    out_graph_dir.mkdir(parents=True, exist_ok=True)
    for child in input_root.iterdir():
        if child.is_file():
            shutil.copy2(child, output_root / child.name)
    resources = NuPlanRealWorldResources(db_root, camera_root, map_root, repo_root=repo_root)
    counters: Counter = Counter()
    failures: List[Dict[str, str]] = []
    examples: List[Dict[str, Any]] = []
    graph_summaries: List[Dict[str, Any]] = []
    if limit is not None and graph_files:
        raise ValueError('--limit cannot be combined with explicit graph selection')
    paths, available_graphs, selected_filenames, selection_mode = select_graph_paths(graph_dir, limit, graph_files)
    selected_graphs = len(paths)
    sample_run = selection_mode != 'all'
    try:
        for graph_path in paths:
            counters['processed'] += 1
            out_path = out_graph_dir / graph_path.name
            try:
                source_graph = json.loads(graph_path.read_text())
                if 'real_world_context' in source_graph:
                    raise AssertionError('source graph already has real_world_context')
                output_graph = copy.deepcopy(source_graph)
                context, local_counts = build_real_world_context(output_graph, resources, camera_channel=camera_channel, composition_radius_m=composition_radius_m, include_weather=include_weather)
                summary = summarize_graph_context(graph_path.name, source_graph, context)
                graph_summaries.append(summary)
                counters.update(local_counts)
                output_graph['real_world_context'] = context
                validate_preserved_graph(source_graph, output_graph)
                out_path.write_text(json.dumps(output_graph, indent=2))
                validate_preserved_graph(source_graph, json.loads(out_path.read_text()))
                counters['written'] += 1
                if context.get('pose') and context['pose'].get('latitude') is not None:
                    counters['enriched'] += 1
                if len(examples) < 3:
                    examples.append({'input_graph': graph_path.as_posix(), 'output_graph': out_path.as_posix(), 'db_file': source_graph.get('metadata', {}).get('db_file'), 'scene_token': source_graph.get('metadata', {}).get('scene_token'), 'timestamp_us': context.get('source', {}).get('timestamp_us'), 'image_path': (context.get('camera') or {}).get('image_path'), 'errors': context.get('errors', [])})
            except Exception as exc:
                if isinstance(exc, AssertionError):
                    raise
                counters['failed'] += 1
                failures.append({'graph': graph_path.as_posix(), 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        resources.close()
    report = {'input_root': input_root.as_posix(), 'output_root': output_root.as_posix(), 'limit': limit, 'available_graphs': available_graphs, 'selected_graphs': selected_graphs, 'counts': dict(counters), 'examples': examples, 'failures': failures[:200], 'failure_count_total': len(failures)}
    report['sample_run'] = sample_run
    report['selection_mode'] = selection_mode
    report['selected_filenames'] = selected_filenames if sample_run else []
    report['selected_filename_count'] = len(selected_filenames)
    report['graph_summaries'] = graph_summaries
    report.update(camera_delta_stats(graph_summaries))
    report.update(dict(counters))
    (output_root / 'real_world_enrichment_report.json').write_text(json.dumps(report, indent=2))
    return report


def subset_jobs(input_root: Path, output_root: Path, subsets: Optional[Sequence[str]]) -> List[Tuple[Path, Path]]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    if (input_root / 'graphs').exists():
        return [(input_root, output_root)]
    selected = list(subsets or [])
    if not selected:
        selected = ['clean'] if (input_root / 'clean' / 'graphs').exists() else [child.name for child in sorted(input_root.iterdir()) if (child / 'graphs').exists()]
    return [(input_root / subset, output_root / subset) for subset in selected]
