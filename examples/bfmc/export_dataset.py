"""Export pure BFMC 33-waypoint multi-task dataset with 3-way zero-leakage split (train/val/test)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

CLS_NAMES = (
    "Stop", "Pedestrian", "Car", "Priority", "Crosswalk", "Oneway", "Stop-Line",
    "Parking", "Highway-Entry", "Roundabout", "Highway-Exit", "Traffic-Light", "No-Entry",
)
CMD_NAMES = ("FOLLOW_LANE", "TURN_LEFT", "TURN_RIGHT", "STRAIGHT")

VAL_SESSIONS = ("Video_19700101_085820_output", "Video_20260227_135534_output", "Video_140422")
TEST_SESSIONS = ("dataCollect2", "video1", "video4")
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _compute_dhash(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((small[:, 1:] > small[:, :-1]).flatten())


def parse_session_and_frame(name: str, w: int, h: int) -> Tuple[str, int]:
    """Classify recording session and frame number; tag non-BFMC domains for exclusion."""
    s = re.sub(r"^(?:[0-9a-f]{8}_c\d_)+", "", name)
    s = re.sub(r"^(?:[0-9a-f]{8}_)+", "", s)
    s = re.sub(r"^(?:copy_\d+_)+", "", s)
    s = re.sub(r"^(?:[0-9a-f]{8}_)+", "", s)
    m_fn = re.search(r"frame_(\d+)", s)
    fnum = int(m_fn.group(1)) if m_fn else -1

    if (w, h) == (1640, 590):
        return "NON_BFMC_culane", fnum
    if (w, h) in ((1280, 720), (960, 540)):
        return "NON_BFMC_udacity", fnum
    if "solidYellowLeft" in s or "solidWhiteRight" in s:
        return "NON_BFMC_udacity_lane", fnum
    for token, sess in (
        ("Video_19700101_080303", "Video_19700101_080303"),
        ("Video_19700101_085820", "Video_19700101_085820_output"),
        ("Video_20260227_135534", "Video_20260227_135534_output"),
        ("Video_20260227_133239", "Video_20260227_133239_output"),
        ("Video_20260119_140111", "Video_20260119_140111_output"),
        ("Video_140422", "Video_140422"),
        ("dataCollect2", "dataCollect2"),
        ("dataCollect0", "dataCollect0"),
        ("dataCollect_frame", "dataCollect_misc"),
        ("part_00", "part_00x_session"),
        ("video4_frame", "video4"),
        ("video3_frame", "video3"),
        ("video2_frame", "video2"),
        ("video1_frame", "video1"),
        ("values_", "values_session"),
    ):
        if token in s:
            return sess, fnum
    if (w, h) == (720, 480):
        return "bfmc_720x480_rf", fnum
    return "bfmc_rf_640", fnum


def _fetch_minio_bytes(client: Any, bucket: str, uri: str, filename: str) -> bytes | None:
    cands = [uri.split("/", 3)[-1]] if uri.startswith("minio://") else []
    cands.append(filename)
    for obj in cands:
        try:
            resp = client.get_object(bucket, obj)
            try:
                data = resp.read()
                if data:
                    return data
            finally:
                resp.close()
                resp.release_conn()
        except Exception:
            pass
    if uri and Path(uri).is_file():
        return Path(uri).read_bytes()
    return None


def _min_dhash_dist(hashes_a: np.ndarray, h_vec: np.ndarray) -> int:
    if len(hashes_a) == 0:
        return 64
    return int(_POPCOUNT[np.bitwise_xor(hashes_a, h_vec)].sum(axis=1).min())


def _assign_splits(
    records: List[Dict[str, Any]],
    sha_to_dhash: Dict[str, np.ndarray],
    dhash_gap: int = 6,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Assign zero-leakage train/val/test splits by session, temporal buffer, and dHash guard."""
    v0803 = sorted(
        r["frame_num"] for r in records if r["session"] == "Video_19700101_080303" and r["frame_num"] >= 0
    )
    n0 = len(v0803)
    # Chunk 0 (start, rich Oneway/Ped/Stop/HW-Exit) -> val; Chunk 5 (mid, rich Crosswalk/Priority/RA) -> test
    val_0803_hi = v0803[int(n0 * 0.10)] if n0 else -1
    test_0803_lo = v0803[int(n0 * 0.50)] if n0 else 10**9
    test_0803_hi = v0803[int(n0 * 0.60)] if n0 else -1
    buf = 240  # ~8s temporal buffer dropped around every temporal split boundary

    rf720 = sorted(r["frame_num"] for r in records if r["session"] == "bfmc_720x480_rf" and r["frame_num"] >= 0)
    rf720_val_lo = rf720[int(len(rf720) * 0.72)] if rf720 else 10**9
    rf720_test_lo = rf720[int(len(rf720) * 0.86)] if rf720 else 10**9

    rf640_shas = sorted({r["img_sha"] for r in records if r["session"] == "bfmc_rf_640"})
    rf640_val_set = set(rf640_shas[: int(len(rf640_shas) * 0.08)])
    rf640_test_set = set(rf640_shas[int(len(rf640_shas) * 0.08) : int(len(rf640_shas) * 0.14)])

    sha_split: Dict[str, str] = {}
    for r in records:
        sha, sess, fn = r["img_sha"], r["session"], r["frame_num"]
        if sha in sha_split:
            continue
        if sess in VAL_SESSIONS:
            sha_split[sha] = "val"
        elif sess in TEST_SESSIONS:
            sha_split[sha] = "test"
        elif sess == "Video_19700101_080303":
            if fn <= val_0803_hi:
                sha_split[sha] = "val"
            elif val_0803_hi < fn <= val_0803_hi + buf:
                sha_split[sha] = "gap"
            elif test_0803_lo - buf <= fn < test_0803_lo:
                sha_split[sha] = "gap"
            elif test_0803_lo <= fn <= test_0803_hi:
                sha_split[sha] = "test"
            elif test_0803_hi < fn <= test_0803_hi + buf:
                sha_split[sha] = "gap"
            else:
                sha_split[sha] = "train"
        elif sess == "bfmc_720x480_rf":
            if fn >= rf720_test_lo:
                sha_split[sha] = "test"
            elif fn >= rf720_val_lo:
                sha_split[sha] = "val"
            else:
                sha_split[sha] = "train"
        elif sess == "bfmc_rf_640":
            if sha in rf640_val_set:
                sha_split[sha] = "val"
            elif sha in rf640_test_set:
                sha_split[sha] = "test"
            else:
                sha_split[sha] = "train"
        else:
            sha_split[sha] = "train"

    # Hard perceptual dHash leakage guard: drop any val/test image within Hamming <= dhash_gap of train
    train_shas = [s for s, sp in sha_split.items() if sp == "train"]
    train_h = np.array([sha_to_dhash[s] for s in train_shas], dtype=np.uint8) if train_shas else np.zeros((0, 8), dtype=np.uint8)
    dropped_dhash = 0
    kept_val_shas: List[str] = []
    for s in [s for s, sp in sha_split.items() if sp == "val"]:
        if _min_dhash_dist(train_h, sha_to_dhash[s]) <= dhash_gap:
            sha_split[s] = "dhash_drop"
            dropped_dhash += 1
        else:
            kept_val_shas.append(s)
    val_h = np.array([sha_to_dhash[s] for s in kept_val_shas], dtype=np.uint8) if kept_val_shas else np.zeros((0, 8), dtype=np.uint8)
    for s in [s for s, sp in sha_split.items() if sp == "test"]:
        if _min_dhash_dist(train_h, sha_to_dhash[s]) <= dhash_gap or _min_dhash_dist(val_h, sha_to_dhash[s]) <= dhash_gap:
            sha_split[s] = "dhash_drop"
            dropped_dhash += 1

    tr, va, te = [], [], []
    for r in records:
        sp = sha_split[r["img_sha"]]
        if sp == "train":
            tr.append(r)
        elif sp == "val":
            va.append(r)
        elif sp == "test":
            te.append(r)

    s_tr, s_va, s_te = {r["img_sha"] for r in tr}, {r["img_sha"] for r in va}, {r["img_sha"] for r in te}
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "DATA LEAKAGE: shared SHA across splits!"
    gap_cnt = sum(1 for sp in sha_split.values() if sp == "gap")
    return tr, va, te, {"temporal_gap_imgs": gap_cnt, "dhash_dropped_imgs": dropped_dhash}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def export_bfmc_dataset(
    db_path: Path,
    out_dir: Path,
    project_id: int = 2,
    expected_waypoints: int = 33,
    minio_endpoint: str = "localhost:9000",
    minio_access: str = "minioadmin",
    minio_secret: str = "minioadmin",
    minio_bucket: str = "neuropilot-data",
    unlabeled_src: Path | None = None,
    dhash_gap: int = 6,
) -> Dict[str, Any]:
    """Export pure BFMC 33-waypoint frames with strict 3-way zero-leakage split and portable paths."""
    from minio import Minio

    client = Minio(minio_endpoint, access_key=minio_access, secret_key=minio_secret, secure=False)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT image_name, image_path, data FROM samples WHERE project_id = ? ORDER BY image_name",
        (project_id,),
    ).fetchall()
    conn.close()

    records: List[Dict[str, Any]] = []
    sha_to_dhash: Dict[str, np.ndarray] = {}
    dropped_non_bfmc = dropped_no_wp = fixed_reversed = 0

    for idx, r in enumerate(rows):
        d = json.loads(r["data"]) if r["data"] else {}
        raw_wps = d.get("waypoints") or []
        if len(raw_wps) != expected_waypoints:
            dropped_no_wp += 1
            continue
        raw = _fetch_minio_bytes(client, minio_bucket, r["image_path"] or "", r["image_name"])
        if raw is None:
            continue
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        session, fnum = parse_session_and_frame(r["image_name"], w, h)
        if session.startswith("NON_BFMC"):
            dropped_non_bfmc += 1
            continue

        pix_sha = hashlib.sha256(img.tobytes()).hexdigest()[:16]
        rel_path = f"images/{pix_sha}.jpg"
        if pix_sha not in sha_to_dhash:
            img_file = out_dir / rel_path
            if not img_file.exists():
                img_file.write_bytes(raw)
            sha_to_dhash[pix_sha] = _compute_dhash(img)

        wps = [[round(float(p["x"]), 6), round(float(p["y"]), 6)] for p in raw_wps]
        if wps[-1][1] - wps[0][1] > 0.05:
            wps.reverse()
            fixed_reversed += 1

        bboxes = [
            [int(b["category"]), round(float(b["cx"]), 6), round(float(b["cy"]), 6),
             round(float(b["w"]), 6), round(float(b["h"]), 6)]
            for b in (d.get("bboxes") or [])
            if 0 <= int(b["category"]) < len(CLS_NAMES)
        ]
        records.append({
            "id": idx,
            "orig_name": r["image_name"],
            "img_sha": pix_sha,
            "img_path": rel_path,
            "width": w,
            "height": h,
            "command": int(d.get("command", 0) or 0),
            "waypoints": wps,
            "bboxes": bboxes,
            "session": session,
            "frame_num": fnum,
        })

    train_recs, val_recs, test_recs, guard = _assign_splits(records, sha_to_dhash, dhash_gap=dhash_gap)

    # Remove unused image files belonging to dropped gap/dHash frames so export folder is 100% clean
    used_shas: Set[str] = {r["img_sha"] for r in (train_recs + val_recs + test_recs)}
    for f in img_dir.glob("*.jpg"):
        if f.stem not in used_shas:
            f.unlink()

    for name, recs in (("train", train_recs), ("val", val_recs), ("test", test_recs)):
        (out_dir / f"{name}.json").write_text(json.dumps(recs))
        (out_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")

    unlabeled_count = 0
    if unlabeled_src and unlabeled_src.is_file():
        u_paths = [p for p in json.loads(unlabeled_src.read_text()) if Path(p).is_file()]
        (out_dir / "unlabeled.json").write_text(json.dumps(u_paths))
        unlabeled_count = len(u_paths)

    def _stats(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        cmds = Counter(CMD_NAMES[r["command"]] for r in recs)
        cls_c = Counter(CLS_NAMES[b[0]] for r in recs for b in r["bboxes"])
        return {
            "frames": len(recs),
            "unique_images": len({r["img_sha"] for r in recs}),
            "total_boxes": sum(len(r["bboxes"]) for r in recs),
            "commands": {c: cmds.get(c, 0) for c in CMD_NAMES},
            "classes": dict(sorted(cls_c.items())),
            "sessions": dict(Counter(r["session"] for r in recs).most_common()),
        }

    meta = {
        "classes": list(CLS_NAMES),
        "commands": list(CMD_NAMES),
        "num_waypoints": expected_waypoints,
        "dropped_non_bfmc": dropped_non_bfmc,
        "dropped_no_wp": dropped_no_wp,
        "fixed_reversed": fixed_reversed,
        "dhash_min_cross_split_gap": dhash_gap + 1,
        **guard,
        "unlabeled_count": unlabeled_count,
        "sha256": {f"{s}.json": _sha256_file(out_dir / f"{s}.json") for s in ("train", "val", "test")},
        "train": _stats(train_recs),
        "val": _stats(val_recs),
        "test": _stats(test_recs),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Export clean 3-way zero-leakage BFMC 33-waypoint dataset.")
    p.add_argument("--db", default="data/labeler.db", help="Path to NeuroPilot Labeler SQLite DB")
    p.add_argument("--out-dir", default="datasets/bfmc_33wp", help="Output dataset directory")
    p.add_argument("--project-id", type=int, default=2)
    p.add_argument("--num-waypoints", type=int, default=33)
    p.add_argument("--dhash-gap", type=int, default=6)
    p.add_argument("--unlabeled-src", default="", help="Optional path to unlabeled.json for REPA")
    args = p.parse_args()
    meta = export_bfmc_dataset(
        db_path=Path(args.db),
        out_dir=Path(args.out_dir),
        project_id=args.project_id,
        expected_waypoints=args.num_waypoints,
        unlabeled_src=Path(args.unlabeled_src) if args.unlabeled_src else None,
        dhash_gap=args.dhash_gap,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
