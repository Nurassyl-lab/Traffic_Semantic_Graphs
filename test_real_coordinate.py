#!/usr/bin/env python3
"""
Probe whether NuPlan-style x/y coordinates can be viewed on a world map.

Default behavior:
  python3 test_real_coordinate.py

This samples a few graphs from data/NuPlan/evaluation_data/noisy_true/graphs and
prints latitude/longitude values you can paste into a map. It can also read
raw NuPlan mini DBs directly with --db.

Important:
The evaluation graphs in this repo are post-processed to be scene-relative:
ego_0 is usually x=0, y=0. Exact world coordinates need the original NuPlan DB
so we can recover the first ego pose and its EPSG code. If the raw DB is not
present, this script falls back to an approximate city-origin conversion.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import math
import os
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError:  # Optional dependency; weather output is graceful when absent.
    requests = None


for devkit_path in (Path("lib/lib/nuplan-devkit"), Path("lib/nuplan-devkit")):
    if devkit_path.exists() and str(devkit_path) not in sys.path:
        sys.path.insert(0, str(devkit_path))


try:
    from pyproj import CRS, Transformer
except ImportError:  # Keep the test script runnable in the base environment.
    CRS = None
    Transformer = None

try:
    from PIL import Image
except ImportError:
    Image = None


# Same coarse city origins used by src/data_processing/nup_process_jsons.py.
# These are useful for quick visual checks, but not exact scene anchors.
CITY_ORIGINS: Dict[str, Tuple[float, float, float]] = {
    "pittsburgh": (40.440624, -79.995888, 0.0),
    "singapore": (1.3521, 103.8198, 0.0),
    "las_vegas": (36.1699, -115.1398, 0.0),
    "boston": (42.3601, -71.0589, 0.0),
}

CITY_ALIASES = {
    "nuplan_pittsburgh": "pittsburgh",
    "nuplan_singapore": "singapore",
    "nuplan_las_vegas": "las_vegas",
    "nuplan_boston": "boston",
    "us-pa-pittsburgh-hazelwood": "pittsburgh",
    "sg-one-north": "singapore",
    "us-nv-las-vegas-strip": "las_vegas",
    "us-ma-boston": "boston",
}

CITY_TIMEZONES = {
    "boston": "America/New_York",
    "pittsburgh": "America/New_York",
    "las_vegas": "America/Los_Angeles",
    "singapore": "Asia/Singapore",
}


@dataclass(frozen=True)
class RawAnchor:
    """Original first ego pose from a raw NuPlan DB."""

    x: float
    y: float
    z: float
    epsg: Optional[int]
    db_path: Path


@dataclass(frozen=True)
class LatLon:
    latitude: float
    longitude: float
    method: str


def normalize_city(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    key = city.strip().lower().replace(" ", "_")
    if key in CITY_ORIGINS:
        return key
    if key in CITY_ALIASES:
        return CITY_ALIASES[key]
    key_dash = key.replace("_", "-")
    return CITY_ALIASES.get(key_dash)


def timezone_for_city(city: Optional[str]) -> ZoneInfo:
    normalized = normalize_city(city)
    return ZoneInfo(CITY_TIMEZONES.get(normalized or "", "UTC"))


def timezone_for_db(db_path: Path) -> ZoneInfo:
    info = db_log_info(db_path)
    map_version = info.get("map_version")
    if isinstance(map_version, str):
        city = normalize_city(map_version)
        if city:
            return timezone_for_city(city)

    location = info.get("location")
    if isinstance(location, str):
        return timezone_for_city(location)

    return ZoneInfo("UTC")


def nuplan_timestamp_us_to_utc(timestamp_us: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(int(timestamp_us) / 1_000_000.0, tz=datetime.timezone.utc)


def nuplan_timestamp_us_to_local(timestamp_us: int, timezone: ZoneInfo) -> datetime.datetime:
    return nuplan_timestamp_us_to_utc(timestamp_us).astimezone(timezone)


def local_xy_to_latlon(
    x_m: float,
    y_m: float,
    city: str,
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None,
    origin_alt: float = 0.0,
) -> LatLon:
    """Convert local east/north meters to WGS84 using a city-level origin."""

    if origin_lat is None or origin_lon is None:
        if city not in CITY_ORIGINS:
            raise ValueError(f"Unknown city '{city}'. Known cities: {sorted(CITY_ORIGINS)}")
        origin_lat, origin_lon, origin_alt = CITY_ORIGINS[city]

    if CRS is not None and Transformer is not None:
        geodetic = CRS.from_epsg(4979)
        local_enu = CRS.from_proj4(
            "+proj=tmerc "
            f"+lat_0={origin_lat} +lon_0={origin_lon} "
            "+k=1 +x_0=0 +y_0=0 +ellps=WGS84"
        )
        transformer = Transformer.from_crs(local_enu, geodetic, always_xy=True)
        lon, lat, _alt = transformer.transform(x_m, y_m, origin_alt)
        return LatLon(float(lat), float(lon), "approx_city_origin_pyproj")

    # Small-distance fallback. Accurate enough for a quick "does this land in
    # the city?" check, but use pyproj for final work.
    earth_radius_m = 6_378_137.0
    lat = origin_lat + math.degrees(y_m / earth_radius_m)
    lon = origin_lon + math.degrees(x_m / (earth_radius_m * math.cos(math.radians(origin_lat))))
    return LatLon(float(lat), float(lon), "approx_city_origin_spherical")


def utm_epsg_to_latlon(x_m: float, y_m: float, epsg: int) -> LatLon:
    """Convert WGS84 UTM EPSG coordinates to lat/lon without external packages."""

    epsg = int(epsg)
    if 32601 <= epsg <= 32660:
        zone = epsg - 32600
        northern = True
    elif 32701 <= epsg <= 32760:
        zone = epsg - 32700
        northern = False
    else:
        raise ValueError(f"EPSG:{epsg} is not a WGS84 UTM zone")

    a = 6378137.0
    f = 1.0 / 298.257223563
    k0 = 0.9996
    e2 = f * (2.0 - f)
    e_prime2 = e2 / (1.0 - e2)
    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))

    x = x_m - 500000.0
    y = y_m if northern else y_m - 10000000.0

    m = y / k0
    mu = m / (a * (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0))

    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)

    n1 = a / math.sqrt(1.0 - e2 * sin_phi1**2)
    r1 = a * (1.0 - e2) / (1.0 - e2 * sin_phi1**2) ** 1.5
    t1 = tan_phi1**2
    c1 = e_prime2 * cos_phi1**2
    d = x / (n1 * k0)

    lat_rad = phi1 - (n1 * tan_phi1 / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * e_prime2) * d**4 / 24.0
        + (
            61.0
            + 90.0 * t1
            + 298.0 * c1
            + 45.0 * t1**2
            - 252.0 * e_prime2
            - 3.0 * c1**2
        )
        * d**6
        / 720.0
    )
    lon_origin_rad = math.radians((zone - 1) * 6 - 180 + 3)
    lon_rad = lon_origin_rad + (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * e_prime2 + 24.0 * t1**2)
        * d**5
        / 120.0
    ) / cos_phi1

    return LatLon(math.degrees(lat_rad), math.degrees(lon_rad), f"exact_utm_epsg_{epsg}")


def epsg_xy_to_latlon(x_m: float, y_m: float, epsg: int) -> LatLon:
    """Convert projected EPSG x/y meters to WGS84 lat/lon."""

    epsg = int(epsg)
    if CRS is not None and Transformer is not None:
        transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(x_m, y_m)
        return LatLon(float(lat), float(lon), f"exact_epsg_{epsg}_pyproj")

    if 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760:
        return utm_epsg_to_latlon(x_m, y_m, epsg)

    raise RuntimeError(
        f"pyproj is required for EPSG:{epsg}. This script has a no-dependency "
        "fallback only for WGS84 UTM EPSG zones 32601-32660 and 32701-32760."
    )


def numeric_sort_key(path: Path) -> Tuple[int, str]:
    match = re.match(r"(\d+)_graph\.json$", path.name)
    return (int(match.group(1)) if match else 10**12, path.name)


def graph_paths(args: argparse.Namespace) -> List[Path]:
    if args.graph_dir:
        graph_dir = Path(args.graph_dir)
    else:
        graph_dir = Path(args.eval_root) / args.split / "graphs"

    if args.graph:
        paths = []
        for value in args.graph:
            p = Path(value)
            if not p.is_absolute() and not p.exists():
                p = graph_dir / value
            paths.append(p)
        return paths

    paths = sorted(graph_dir.glob("*_graph.json"), key=numeric_sort_key)
    if not args.limit:
        return paths

    selected: List[Path] = []
    seen_cities = set()
    for path in paths:
        try:
            with path.open("r") as f:
                city = normalize_city(json.load(f).get("metadata", {}).get("city"))
        except (OSError, json.JSONDecodeError):
            city = None
        if city and city not in seen_cities:
            selected.append(path)
            seen_cities.add(city)
            if len(selected) >= args.limit:
                return selected

    for path in paths:
        if path not in selected:
            selected.append(path)
            if len(selected) >= args.limit:
                break
    return selected


def is_scene_relative(graph: dict) -> bool:
    ego_nodes = graph.get("nodes", {}).get("ego", [])
    if not ego_nodes:
        return False
    features = ego_nodes[0].get("features", {})
    x0 = features.get("x")
    y0 = features.get("y")
    return x0 is not None and y0 is not None and abs(float(x0)) < 1e-6 and abs(float(y0)) < 1e-6


def iter_sample_nodes(graph: dict, node_types: Sequence[str], max_nodes: int) -> Iterable[Tuple[str, str, dict]]:
    """Yield a small, readable sample of nodes with x/y features."""

    yielded = 0
    for node_type in node_types:
        nodes = graph.get("nodes", {}).get(node_type, [])
        for node in nodes:
            features = node.get("features", {})
            if features.get("x") is None or features.get("y") is None:
                continue
            yield node_type, node.get("id", "?"), features
            yielded += 1
            if yielded >= max_nodes:
                return


def candidate_db_paths(raw_db_root: Path, db_file: str, city: Optional[str]) -> Iterable[Path]:
    if not db_file or not raw_db_root.exists():
        return []

    city = normalize_city(city)
    candidates: List[Path] = []
    if city:
        train_dir = f"train_{city}"
        candidates.extend(
            [
                raw_db_root / train_dir / "nuplan-v1.1" / "train" / db_file,
                raw_db_root / train_dir / "nuplan-v1.1" / "val" / db_file,
                raw_db_root / train_dir / "nuplan-v1.1" / "mini" / db_file,
                raw_db_root / f"{city}_frames" / db_file,
            ]
        )

    candidates.extend(
        [
            raw_db_root / "nuplan-v1.1" / "train" / db_file,
            raw_db_root / "nuplan-v1.1" / "val" / db_file,
            raw_db_root / "nuplan-v1.1" / "mini" / db_file,
            raw_db_root / db_file,
        ]
    )

    seen = set()
    existing = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            existing.append(path)

    if existing:
        return existing

    # Last resort. This can be slower on a full dataset, but it makes the script
    # convenient when raw DBs are stored in a slightly different layout.
    return list(raw_db_root.rglob(db_file))


def _token_params(scene_token: Optional[str]) -> List[object]:
    if not scene_token:
        return []
    params: List[object] = [scene_token]
    try:
        params.insert(0, bytes.fromhex(scene_token))
    except ValueError:
        pass
    return params


def read_raw_anchor(db_path: Path, metadata: dict) -> Optional[RawAnchor]:
    scene_name = metadata.get("scene_name")
    scene_token = metadata.get("scene_token")

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(ego_pose);")}
        epsg_expr = "ep.epsg" if "epsg" in cols else "NULL"

        base_query = f"""
            SELECT ep.x, ep.y, COALESCE(ep.z, 0.0) AS z, {epsg_expr} AS epsg
            FROM lidar_pc AS lp
            JOIN scene AS s ON lp.scene_token = s.token
            JOIN ego_pose AS ep ON lp.ego_pose_token = ep.token
            WHERE {{where_clause}}
            ORDER BY lp.timestamp
            LIMIT 1
        """

        if scene_name:
            row = conn.execute(base_query.format(where_clause="s.name = ?"), (scene_name,)).fetchone()
            if row:
                epsg = int(row[3]) if row[3] is not None else None
                return RawAnchor(float(row[0]), float(row[1]), float(row[2]), epsg, db_path)

        for token_param in _token_params(scene_token):
            row = conn.execute(base_query.format(where_clause="s.token = ?"), (token_param,)).fetchone()
            if row:
                epsg = int(row[3]) if row[3] is not None else None
                return RawAnchor(float(row[0]), float(row[1]), float(row[2]), epsg, db_path)

    return None


def find_raw_anchor(metadata: dict, raw_db_root: Path) -> Optional[RawAnchor]:
    db_file = metadata.get("db_file")
    city = metadata.get("city")
    if not db_file:
        return None

    for db_path in candidate_db_paths(raw_db_root, db_file, city):
        try:
            anchor = read_raw_anchor(db_path, metadata)
        except sqlite3.Error:
            continue
        if anchor:
            return anchor
    return None


def decimal_to_dms(value: float, positive_hemi: str, negative_hemi: str) -> str:
    hemi = positive_hemi if value >= 0 else negative_hemi
    value = abs(value)
    degrees = int(value)
    minutes_float = (value - degrees) * 60.0
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60.0
    return f"{degrees:d}°{minutes:02d}'{seconds:05.2f}\"{hemi}"


def latlon_to_dms(lat: float, lon: float) -> str:
    return f"{decimal_to_dms(lat, 'N', 'S')} {decimal_to_dms(lon, 'E', 'W')}"


def print_conversion(label: str, x: float, y: float, latlon: LatLon, extra: str = "") -> None:
    parts = [f"[{latlon.latitude:.8f},{latlon.longitude:.8f}]"]
    if extra:
        parts.append(extra)
    print(" ".join(parts))


def default_image_roots(db_path: Path) -> List[Path]:
    roots = [
        Path("nuplan_actual/nuplan-v1.1_mini_camera"),
    ]
    for parent in [db_path.parent, *db_path.parents]:
        roots.extend(
            [
                parent,
                parent / "sensor_blobs" / "mini",
                parent / "data" / "sensor_blobs" / "mini",
                parent / "data" / "sets" / "nuplan" / "mini",
            ]
        )

    roots.extend(
        [
            Path("nuplan_actual/data/sensor_blobs/mini"),
            Path("nuplan_actual/data/sets/nuplan/mini"),
            Path("data/sensor_blobs/mini"),
            Path("data/sets/nuplan/mini"),
        ]
    )

    unique = []
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def candidate_image_paths(filename: Optional[str], db_path: Path, image_roots: Optional[Sequence[str]]) -> List[Path]:
    if not filename:
        return []

    image_path = Path(filename)
    roots = [Path(root) for root in image_roots] if image_roots else default_image_roots(db_path)
    if image_path.is_absolute():
        candidates = [image_path]
    else:
        candidates = [root / filename for root in roots]
        candidates.append(image_path)

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def resolve_image_path(filename: Optional[str], db_path: Path, image_roots: Optional[Sequence[str]]) -> Optional[Path]:
    for candidate in candidate_image_paths(filename, db_path, image_roots):
        if candidate.exists():
            return candidate
    return None


def potential_image_path(filename: Optional[str], db_path: Path, image_roots: Optional[Sequence[str]]) -> Optional[Path]:
    candidates = candidate_image_paths(filename, db_path, image_roots)
    return candidates[0] if candidates else None


def display_image_kitty(image_path: Path) -> None:
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    chunk_size = 4096
    for start in range(0, len(data), chunk_size):
        chunk = data[start : start + chunk_size]
        more = 1 if start + chunk_size < len(data) else 0
        sys.stdout.write(f"\033_Gf=100,a=T,m={more};{chunk}\033\\")
    sys.stdout.write("\n")


def display_image_iterm(image_path: Path, width: int) -> None:
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    sys.stdout.write(
        f"\033]1337;File=inline=1;width={width};preserveAspectRatio=1:{data}\a\n"
    )


def display_image_ansi(image_path: Path, width: int) -> None:
    if Image is None:
        return

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        max_width = max(1, min(width, shutil.get_terminal_size((width, 40)).columns))
        rows = max(1, int((img.height / img.width) * max_width * 0.5))
        img = img.resize((max_width, rows * 2))

        pixels = img.load()
        for y in range(0, img.height, 2):
            parts = []
            for x in range(img.width):
                tr, tg, tb = pixels[x, y]
                br, bg, bb = pixels[x, min(y + 1, img.height - 1)]
                parts.append(f"\033[38;2;{tr};{tg};{tb}m\033[48;2;{br};{bg};{bb}m▀")
            sys.stdout.write("".join(parts) + "\033[0m\n")


def display_terminal_image(image_path: Optional[Path], mode: str, width: int) -> None:
    if image_path is None or mode == "none":
        return

    if mode == "path":
        print(str(image_path))
        return

    selected = mode
    if mode == "auto":
        if os.environ.get("KITTY_WINDOW_ID"):
            selected = "kitty"
        elif os.environ.get("TERM_PROGRAM") == "iTerm.app":
            selected = "iterm"
        else:
            selected = "ansi"

    try:
        if selected == "kitty":
            display_image_kitty(image_path)
        elif selected == "iterm":
            display_image_iterm(image_path, width)
        else:
            display_image_ansi(image_path, width)
    except Exception:
        return


MAP_NAME_ALIASES = {
    "las_vegas": "us-nv-las-vegas-strip",
    "boston": "us-ma-boston",
    "pittsburgh": "us-pa-pittsburgh-hazelwood",
    "pitssburgh": "us-pa-pittsburgh-hazelwood",
    "singapore": "sg-one-north",
}


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def db_map_name(db_path: Path) -> Optional[str]:
    info = db_log_info(db_path)
    map_version = info.get("map_version")
    if isinstance(map_version, str) and map_version:
        return map_version
    location = info.get("location")
    if isinstance(location, str):
        return MAP_NAME_ALIASES.get(location, location)
    return None


def resolved_map_root(map_root: str) -> Path:
    root = Path(map_root)
    candidates = [
        root,
        root / "nuplan-maps-v1.0",
        root / "maps" / "nuplan-maps-v1.0",
        root.parent,
        root.parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "nuplan-maps-v1.0.json").exists():
            return candidate
    return root


def safe_obj_id(obj: Any) -> Optional[str]:
    value = getattr(obj, "id", None)
    return str(value) if value is not None else None


def safe_distance(obj: Any, point: Any) -> Optional[float]:
    polygon = getattr(obj, "polygon", None)
    if polygon is None:
        return None
    try:
        return float(polygon.distance(point))
    except Exception:
        return None


def safe_contains(obj: Any, point: Any) -> bool:
    polygon = getattr(obj, "polygon", None)
    if polygon is None:
        return False
    try:
        return bool(polygon.contains(point) or polygon.distance(point) < 0.25)
    except Exception:
        return False


def heading_change_deg(obj: Any) -> Optional[float]:
    baseline_path = getattr(obj, "baseline_path", None)
    path = getattr(baseline_path, "discrete_path", None)
    if not path or len(path) < 2:
        return None
    h0 = getattr(path[0], "heading", None)
    h1 = getattr(path[-1], "heading", None)
    if h0 is None or h1 is None:
        return None
    return math.degrees(normalize_angle(float(h1) - float(h0)))


def turn_label(obj: Any) -> str:
    change = heading_change_deg(obj)
    if change is None:
        return "unknown"
    if change > 20.0:
        return "left"
    if change < -20.0:
        return "right"
    return "straight"


class NuPlanCompositionProvider:
    def __init__(self, db_path: Path, map_root: str, radius_m: float):
        self.db_path = db_path
        self.map_name = db_map_name(db_path)
        self.radius_m = radius_m
        self.unavailable_reason: Optional[str] = None
        self.map_api = None

        try:
            from nuplan.common.actor_state.state_representation import StateSE2
            from nuplan.common.maps.maps_datatypes import SemanticMapLayer
            from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
            from shapely.geometry import Point as ShapelyPoint
        except Exception as exc:
            self.unavailable_reason = f"nuplan_map_dependencies_missing:{type(exc).__name__}:{exc}"
            return

        if not self.map_name:
            self.unavailable_reason = "map_name_missing_in_db_log"
            return

        root = resolved_map_root(map_root)
        try:
            self.map_api = get_maps_api(str(root.resolve()), "nuplan-maps-v1.0", self.map_name)
        except Exception as exc:
            self.unavailable_reason = f"map_load_failed:{self.map_name}:{root}:{type(exc).__name__}:{exc}"
            return

        self.StateSE2 = StateSE2
        self.SemanticMapLayer = SemanticMapLayer
        self.ShapelyPoint = ShapelyPoint

    def describe(self, x: float, y: float, heading: float) -> Dict[str, Any]:
        if self.map_api is None:
            return {"composition_unavailable": self.unavailable_reason or "map_api_unavailable"}

        layer_names = ["LANE", "LANE_CONNECTOR", "CROSSWALK", "STOP_LINE", "INTERSECTION"]
        layers = [getattr(self.SemanticMapLayer, name) for name in layer_names if hasattr(self.SemanticMapLayer, name)]
        state = self.StateSE2(x, y, heading)
        point = self.ShapelyPoint(x, y)

        try:
            proximal = self.map_api.get_proximal_map_objects(state, self.radius_m, layers)
        except Exception as exc:
            return {"composition_unavailable": f"map_query_failed:{type(exc).__name__}:{exc}"}

        by_name: Dict[str, List[Any]] = {}
        for name in layer_names:
            layer = getattr(self.SemanticMapLayer, name, None)
            by_name[name] = list(proximal.get(layer, [])) if layer is not None else []

        lanes = by_name.get("LANE", [])
        connectors = by_name.get("LANE_CONNECTOR", [])
        drivable = lanes + connectors

        ego_lane = None
        if drivable:
            containing = [obj for obj in drivable if safe_contains(obj, point)]
            ego_lane = (
                min(containing, key=lambda obj: safe_distance(obj, point) or 0.0)
                if containing
                else min(drivable, key=lambda obj: safe_distance(obj, point) if safe_distance(obj, point) is not None else float("inf"))
            )

        left_lane_ids: List[str] = []
        right_lane_ids: List[str] = []
        if ego_lane is not None:
            for lane in lanes:
                if lane is ego_lane:
                    continue
                try:
                    if lane.is_same_roadblock(ego_lane):
                        if lane.is_left_of(ego_lane):
                            lane_id = safe_obj_id(lane)
                            if lane_id:
                                left_lane_ids.append(lane_id)
                        elif lane.is_right_of(ego_lane):
                            lane_id = safe_obj_id(lane)
                            if lane_id:
                                right_lane_ids.append(lane_id)
                except Exception:
                    continue

        turn_buckets = {"left": [], "right": [], "straight": [], "unknown": []}
        for obj in drivable:
            obj_id = safe_obj_id(obj)
            if not obj_id:
                continue
            turn_buckets.setdefault(turn_label(obj), []).append(obj_id)

        ego_lane_id = safe_obj_id(ego_lane) if ego_lane is not None else None
        ego_lane_distance = safe_distance(ego_lane, point) if ego_lane is not None else None

        return {
            "source": "nuplan_hd_map",
            "map": self.map_name,
            "ego_lane_id": ego_lane_id,
            "ego_lane_distance_m": round(ego_lane_distance, 3) if ego_lane_distance is not None else None,
            "ego_lane_turn": turn_label(ego_lane) if ego_lane is not None else None,
            "nearby_lanes": len(lanes),
            "nearby_lane_connectors": len(connectors),
            "nearby_crosswalks": len(by_name.get("CROSSWALK", [])),
            "nearby_stop_lines": len(by_name.get("STOP_LINE", [])),
            "nearby_intersections": len(by_name.get("INTERSECTION", [])),
            "left_lane_ids": left_lane_ids[:5],
            "right_lane_ids": right_lane_ids[:5],
            "turning_lane_ids": {key: value[:8] for key, value in turn_buckets.items() if value},
        }


def print_composition(provider: Optional[NuPlanCompositionProvider], x: float, y: float, heading: float) -> None:
    if provider is None:
        return
    print(json.dumps(provider.describe(x, y, heading), sort_keys=True, separators=(",", ":")))


def convert_manual(args: argparse.Namespace) -> None:
    x = float(args.x)
    y = float(args.y)

    if args.epsg:
        latlon = epsg_xy_to_latlon(x, y, int(args.epsg))
    else:
        city = normalize_city(args.city)
        if not city and (args.origin_lat is None or args.origin_lon is None):
            raise SystemExit("--city or --origin-lat/--origin-lon is required when --epsg is not used")
        latlon = local_xy_to_latlon(
            x,
            y,
            city or "custom",
            origin_lat=args.origin_lat,
            origin_lon=args.origin_lon,
        )

    print_conversion("manual", x, y, latlon)


def resolve_db_path(db_arg: str, raw_db_root: Path) -> Path:
    path = Path(db_arg)
    if path.exists():
        return path
    candidate = raw_db_root / db_arg
    if candidate.exists():
        return candidate
    matches = list(raw_db_root.rglob(path.name)) if raw_db_root.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find DB '{db_arg}' under {raw_db_root}")


def db_log_info(db_path: Path) -> Dict[str, object]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT location, map_version, logfile, date, timestamp FROM log LIMIT 1"
        ).fetchone()
    if not row:
        return {}
    return {
        "location": row[0],
        "map_version": row[1],
        "logfile": row[2],
        "date": row[3],
        "timestamp": row[4],
    }


def local_datetime_for_db_sample(db_path: Path, sample_timestamp: Optional[int]) -> Optional[datetime.datetime]:
    if sample_timestamp is None:
        return None
    try:
        return nuplan_timestamp_us_to_local(int(sample_timestamp), timezone_for_db(db_path))
    except (TypeError, OverflowError, OSError, ValueError):
        return None


def format_db_sample_metadata(db_path: Path, sample_timestamp: Optional[int] = None) -> str:
    info = db_log_info(db_path)
    city = normalize_city(str(info.get("location")) if info.get("location") else None)
    city_name = city or str(info.get("location") or "unknown")

    dt_local = local_datetime_for_db_sample(db_path, sample_timestamp)
    if dt_local is not None:
        date_value = dt_local.date().isoformat()
        time_value = dt_local.strftime("%H:%M:%S")
    else:
        date_value = None
        time_value = None

    if date_value is None:
        match = re.match(r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})", db_path.stem)
        if match:
            date_value = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            time_value = f"{match.group(4)}:{match.group(5)}:{match.group(6)}"

    if date_value is None:
        date_value = "unknown"
    if time_value is None:
        time_value = "unknown"

    return f"city={city_name} date={date_value} time={time_value}"


def weather_for_latlon(lat: float, lon: float, date_value: Optional[str], time_value: Optional[str] = None) -> str:
    if requests is None:
        return "weather=unavailable"
    if not date_value or date_value == "unknown":
        return "weather=unknown"

    try:
        sample_date = datetime.datetime.strptime(date_value, "%Y-%m-%d").date()
    except ValueError:
        return "weather=unknown"

    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Rime fog",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Heavy drizzle",
        61: "Light rain",
        63: "Rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Snow",
        75: "Heavy snow",
        80: "Light rain showers",
        81: "Rain showers",
        82: "Heavy rain showers",
        85: "Light snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Heavy thunderstorm with hail",
    }

    try:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": float(lat),
                "longitude": float(lon),
                "start_date": sample_date.isoformat(),
                "end_date": sample_date.isoformat(),
                "daily": "weather_code",
                "timezone": "auto",
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return "weather=unavailable"

    daily = payload.get("daily") or {}
    codes = daily.get("weather_code") or []
    if not codes:
        return "weather=unknown"

    try:
        code = int(codes[0])
    except (TypeError, ValueError):
        return "weather=unknown"

    return f"weather={weather_codes.get(code, f'Unknown ({code})')}"


def convert_db_samples(args: argparse.Namespace) -> None:
    db_path = resolve_db_path(args.db, Path(args.raw_db_root))
    samples = (
        iter_db_image_poses(db_path, args.limit or 5, args.camera, args.scene)
        if args.image
        else iter_db_ego_poses(db_path, args.limit or 5, args.scene)
    )
    composition_provider = (
        NuPlanCompositionProvider(db_path, args.map_root, args.composition_radius)
        if args.composition
        else None
    )

    for i, (timestamp, x, y, z, epsg, heading, scene_name, image_filename) in enumerate(samples):
        label = f"ego_pose[{i}]"
        if scene_name:
            label = f"{scene_name}:{label}"
        latlon = epsg_xy_to_latlon(x, y, epsg)
        dt_local = local_datetime_for_db_sample(db_path, timestamp)
        date_value = dt_local.date().isoformat() if dt_local is not None else "unknown"
        time_value = dt_local.strftime("%H:%M:%S") if dt_local is not None else None
        weather = weather_for_latlon(latlon.latitude, latlon.longitude, date_value, time_value)
        metadata = format_db_sample_metadata(db_path, timestamp)
        print_conversion(label, x, y, latlon, extra=f"{metadata} {weather} z={z:.3f} epsg={epsg} timestamp={timestamp}")
        print_composition(composition_provider, x, y, heading)
        if args.image:
            image_path = resolve_image_path(image_filename, db_path, args.image_root)
            if image_path is not None:
                display_terminal_image(image_path, args.image_mode, args.image_width)
            else:
                candidate = potential_image_path(image_filename, db_path, args.image_root)
                if candidate is not None:
                    print(str(candidate))


def convert_graph_samples(args: argparse.Namespace) -> None:
    paths = graph_paths(args)
    if not paths:
        raise SystemExit("No graph files found. Check --eval-root, --split, or --graph-dir.")

    for graph_path in paths:
        with graph_path.open("r") as f:
            graph = json.load(f)

        metadata = graph.get("metadata", {})
        raw_city = metadata.get("city") or args.city
        city = normalize_city(raw_city)
        scene_relative = is_scene_relative(graph)
        anchor = None if args.no_db else find_raw_anchor(metadata, Path(args.raw_db_root))

        for node_type, node_id, features in iter_sample_nodes(graph, args.node_types, args.nodes_per_graph):
            rel_x = float(features["x"])
            rel_y = float(features["y"])
            label = f"{node_type}:{node_id}"

            try:
                if anchor and anchor.epsg is not None:
                    abs_x = anchor.x + rel_x if scene_relative else rel_x
                    abs_y = anchor.y + rel_y if scene_relative else rel_y
                    latlon = epsg_xy_to_latlon(abs_x, abs_y, anchor.epsg)
                    print_conversion(label, rel_x, rel_y, latlon, extra=f"abs_x={abs_x:.3f} abs_y={abs_y:.3f}")
                else:
                    if not city:
                        continue
                    latlon = local_xy_to_latlon(
                        rel_x,
                        rel_y,
                        city,
                        origin_lat=args.origin_lat,
                        origin_lon=args.origin_lon,
                    )
                    print_conversion(label, rel_x, rel_y, latlon)
            except Exception:
                continue


def iter_db_ego_poses(
    db_path: Path, limit: int, scene_name: Optional[str] = None
) -> Iterable[Tuple[int, float, float, float, int, float, Optional[str], Optional[str]]]:
    quat_expr = "ep.qw, ep.qx, ep.qy, ep.qz"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        if scene_name:
            rows = conn.execute(
                f"""
                SELECT ep.timestamp, ep.x, ep.y, COALESCE(ep.z, 0.0), ep.epsg, {quat_expr}, s.name, NULL
                FROM lidar_pc AS lp
                JOIN scene AS s ON lp.scene_token = s.token
                JOIN ego_pose AS ep ON lp.ego_pose_token = ep.token
                WHERE s.name = ?
                ORDER BY lp.timestamp
                LIMIT ?
                """,
                (scene_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT timestamp, x, y, COALESCE(z, 0.0), epsg, qw, qx, qy, qz, NULL, NULL
                FROM ego_pose
                ORDER BY timestamp
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    for timestamp, x, y, z, epsg, qw, qx, qy, qz, scene, image_filename in rows:
        heading = yaw_from_quaternion(float(qw), float(qx), float(qy), float(qz))
        yield int(timestamp), float(x), float(y), float(z), int(epsg), heading, scene, image_filename


def iter_db_image_poses(
    db_path: Path, limit: int, camera: str, scene_name: Optional[str] = None
) -> Iterable[Tuple[int, float, float, float, int, float, Optional[str], Optional[str]]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        if scene_name:
            bounds = conn.execute(
                """
                SELECT MIN(lp.timestamp), MAX(lp.timestamp)
                FROM lidar_pc AS lp
                JOIN scene AS s ON lp.scene_token = s.token
                WHERE s.name = ?
                """,
                (scene_name,),
            ).fetchone()
            if not bounds or bounds[0] is None or bounds[1] is None:
                rows = []
            else:
                rows = conn.execute(
                    """
                    SELECT im.timestamp, ep.x, ep.y, COALESCE(ep.z, 0.0), ep.epsg,
                           ep.qw, ep.qx, ep.qy, ep.qz, ?, im.filename_jpg
                    FROM image AS im
                    JOIN camera AS c ON im.camera_token = c.token
                    JOIN ego_pose AS ep ON im.ego_pose_token = ep.token
                    WHERE c.channel = ?
                      AND im.timestamp BETWEEN ? AND ?
                    ORDER BY im.timestamp
                    LIMIT ?
                    """,
                    (scene_name, camera, int(bounds[0]), int(bounds[1]), limit),
                ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT im.timestamp, ep.x, ep.y, COALESCE(ep.z, 0.0), ep.epsg,
                       ep.qw, ep.qx, ep.qy, ep.qz, NULL, im.filename_jpg
                FROM image AS im
                JOIN camera AS c ON im.camera_token = c.token
                JOIN ego_pose AS ep ON im.ego_pose_token = ep.token
                WHERE c.channel = ?
                ORDER BY im.timestamp
                LIMIT ?
                """,
                (camera, limit),
            ).fetchall()

    for timestamp, x, y, z, epsg, qw, qx, qy, qz, scene, image_filename in rows:
        heading = yaw_from_quaternion(float(qw), float(qx), float(qy), float(qz))
        yield int(timestamp), float(x), float(y), float(z), int(epsg), heading, scene, image_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert/sample NuPlan x/y coordinates as map-ready lat/lon.")
    parser.add_argument("--eval-root", default="data/NuPlan/evaluation_data")
    parser.add_argument("--split", default="noisy_true")
    parser.add_argument("--graph-dir", default=None, help="Optional direct path to a directory containing *_graph.json files.")
    parser.add_argument("--graph", nargs="*", help="Specific graph filenames or paths to sample.")
    parser.add_argument("--limit", type=int, default=3, help="Number of graphs to sample by default.")
    parser.add_argument("--nodes-per-graph", type=int, default=4)
    parser.add_argument(
        "--node-types",
        nargs="+",
        default=["ego", "vehicle", "pedestrian", "object"],
        help="Node types to print from each graph.",
    )

    parser.add_argument("--raw-db-root", default="nuplan_actual/data/cache/mini")
    parser.add_argument("--db", default=None, help="Raw NuPlan .db file name/path to sample ego_pose coordinates from.")
    parser.add_argument("--scene", default=None, help="Optional scene name, e.g. scene-0001, when sampling --db.")
    parser.add_argument("--image", action="store_true", help="After each [lat,lon], render the matching camera image if the JPG exists locally.")
    parser.add_argument("--camera", default="CAM_F0", help="Camera channel to display with --image.")
    parser.add_argument("--image-root", nargs="*", default=None, help="Root directory containing NuPlan JPG paths from image.filename_jpg.")
    parser.add_argument("--image-mode", choices=["auto", "ansi", "kitty", "iterm", "path", "none"], default="auto")
    parser.add_argument("--image-width", type=int, default=80, help="Terminal image width for ansi/iterm modes.")
    parser.add_argument("--composition", action="store_true", help="Print NuPlan HD-map road composition after each coordinate.")
    parser.add_argument("--map-root", default="nuplan_actual/nuplan-maps-v1.0", help="Directory containing nuplan-maps-v1.0.json and map folders.")
    parser.add_argument("--composition-radius", type=float, default=30.0, help="Map query radius in meters for --composition.")
    parser.add_argument("--no-db", action="store_true", help="Skip raw DB lookup even if DB files exist.")

    parser.add_argument("--city", default=None, help="City for manual or fallback conversion.")
    parser.add_argument("--origin-lat", type=float, default=None, help="Override local origin latitude.")
    parser.add_argument("--origin-lon", type=float, default=None, help="Override local origin longitude.")

    parser.add_argument("--x", type=float, default=None, help="Manual x/easting coordinate to convert.")
    parser.add_argument("--y", type=float, default=None, help="Manual y/northing coordinate to convert.")
    parser.add_argument("--epsg", type=int, default=None, help="Manual projected EPSG code for --x/--y.")

    args = parser.parse_args()
    if (args.x is None) ^ (args.y is None):
        parser.error("--x and --y must be provided together")
    return args


def main() -> None:
    args = parse_args()
    if args.x is not None and args.y is not None:
        convert_manual(args)
    elif args.db:
        convert_db_samples(args)
    else:
        convert_graph_samples(args)


if __name__ == "__main__":
    main()
