"""
Extend a size-generalization split to additional (larger) test widths.

The v3 split (`mnist_sizegen_splits.json`) already consumes **all** 70,000 MNIST
images: 51,000 train + 1,000 val + 9 widths x 2,000 test. There are therefore no
free images left to give a new width a test set that is disjoint from every other
width. This script frees images by shrinking the per-width test sets, without
touching train/val:

  * `train` and `val` are copied verbatim from the base split (same images), so
    checkpoints trained on the base split remain valid — they only need to be
    re-evaluated on the new per-width test sets.
  * The pooled test images of the base split (9 x 2,000 = 18,000) are re-shuffled
    and re-partitioned into `--test-size` images per width, covering both the
    existing widths and the new ones. With `--test-size 1000` that supports up to
    18 widths.

The result is written under a **new** filename (`--out-split`), so the base split
and any training run reading it are left untouched.

For the new widths the directory is created (with the empty
`{prefix}_png_{train,test}/{digit}/` tree) and the split JSON is written *before*
any INR is fitted — `generate_inrs.py --only-from-split <that JSON>` then fits
exactly the ~1k INRs that will be evaluated, instead of all 70k.

Usage
-----
python scripts/extend_sizegen_splits.py \
    --dataset mnist --init-types sp mup24 \
    --train-width 24 --existing-widths 24 32 48 64 80 96 128 192 256 \
    --new-widths 384 512 768 1024 \
    --base-split mnist_sizegen_splits.json \
    --out-split mnist_sizegen_splits_v4.json --test-size 1000
"""
import argparse
import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

ImageId = tuple[str, str, str]  # (png split dir, digit, file stem)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mnist", "fmnist"], default="mnist")
    p.add_argument(
        "--init-types",
        type=str,
        nargs="+",
        default=["sp", "mup24"],
        help="Directory suffixes to write the extended split for (e.g. sp mup24).",
    )
    p.add_argument("--train-width", type=int, default=24)
    p.add_argument(
        "--existing-widths",
        type=int,
        nargs="+",
        default=[24, 32, 48, 64, 80, 96, 128, 192, 256],
        help="Widths already present on disk (all 70k INRs fitted).",
    )
    p.add_argument(
        "--new-widths",
        type=int,
        nargs="+",
        default=[384, 512, 768, 1024],
        help="New widths; directories are created and only their test INRs need fitting.",
    )
    p.add_argument("--base-split", type=str, default="mnist_sizegen_splits.json")
    p.add_argument("--out-split", type=str, default="mnist_sizegen_splits_v4.json")
    p.add_argument("--test-size", type=int, default=1000, help="Test images per width (disjoint across widths)")
    p.add_argument("--seed", type=int, default=20260819, help="Seed for re-partitioning the test pool")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="Report the partition without writing anything")
    return p.parse_args()


def image_id(path: str) -> ImageId:
    """Identity of the source image an INR file was fitted to.

    `generate_inrs.py` walks the dataset in a fixed order and names files by a
    per-(split, digit) counter, so the triple below refers to the same source
    image at every width.
    """
    parts = Path(path).parts
    return (parts[-3], parts[-2], Path(path).stem)


def ids_of(entry: dict) -> list[ImageId]:
    return [image_id(p) for p in entry["path"]]


def load_base_split(data_root: Path, width: int, suffix: str, split_name: str) -> dict:
    path = data_root / f"w{width}_{suffix}" / split_name
    if not path.exists():
        raise FileNotFoundError(f"Base split not found: {path}")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    if args.data_root is None:
        root = Path(os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data")))
        args.data_root = root / ("fmnist_inrs" if args.dataset == "fmnist" else "mnist_inrs")

    ref_suffix = args.init_types[0]
    all_widths = sorted(set(args.existing_widths) | set(args.new_widths))
    if args.train_width not in all_widths:
        raise ValueError(f"--train-width {args.train_width} must be one of the widths being written")

    # ---- 1. Read the base split and pool its test images -------------------
    base = {w: load_base_split(args.data_root, w, ref_suffix, args.base_split) for w in args.existing_widths}

    train_ids = ids_of(base[args.train_width]["train"])
    val_ids = ids_of(base[args.train_width]["val"])
    fixed = set(train_ids) | set(val_ids)
    if len(fixed) != len(train_ids) + len(val_ids):
        raise ValueError("Base split has overlapping train/val images")

    pool: list[ImageId] = []
    for w in args.existing_widths:
        for key in ("train", "val"):
            if w != args.train_width and base[w][key]["path"]:
                raise ValueError(f"Base split for w{w} unexpectedly has a non-empty '{key}' set")
        pool.extend(ids_of(base[w]["test"]))

    if len(set(pool)) != len(pool):
        raise ValueError("Base split reuses the same image at two test widths")
    leaked = fixed & set(pool)
    if leaked:
        raise ValueError(f"{len(leaked)} base-split test images also appear in train/val")

    print(f"Base split '{args.base_split}' (reference suffix _{ref_suffix}):")
    print(f"  train {len(train_ids):,}  val {len(val_ids):,}  pooled test {len(pool):,}")

    # All init types must agree on the assignment, otherwise one manifest cannot
    # serve both parameterizations.
    for suffix in args.init_types[1:]:
        other = load_base_split(args.data_root, args.train_width, suffix, args.base_split)
        if set(ids_of(other["train"])) != set(train_ids) or set(ids_of(other["val"])) != set(val_ids):
            raise ValueError(f"Base split for _{suffix} assigns different train/val images than _{ref_suffix}")

    # ---- 2. Re-partition the pool across all widths ------------------------
    needed = len(all_widths) * args.test_size
    if needed > len(pool):
        raise ValueError(
            f"Need {needed:,} test images ({len(all_widths)} widths x {args.test_size}) "
            f"but the base test pool only holds {len(pool):,}. "
            f"Lower --test-size or drop a width."
        )

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    test_ids = {}
    cursor = 0
    for w in all_widths:
        test_ids[w] = pool[cursor:cursor + args.test_size]
        cursor += args.test_size
    print(f"  allocated {cursor:,} / {len(pool):,} pooled test images "
          f"({args.test_size:,} per width x {len(all_widths)} widths, {len(pool) - cursor:,} unused)")

    # Disjointness, stated as an assertion rather than assumed.
    seen: set[ImageId] = set(fixed)
    for w in all_widths:
        s = set(test_ids[w])
        assert len(s) == args.test_size, f"w{w}: duplicate images in its own test set"
        assert not (s & seen), f"w{w}: test images overlap another split"
        seen |= s

    # ---- 3. Write one split JSON per (width, init type) ---------------------
    prefix = "fmnist" if args.dataset == "fmnist" else "mnist"
    for suffix in args.init_types:
        print(f"\n_{suffix}:")
        for w in all_widths:
            w_path = args.data_root / f"w{w}_{suffix}"
            is_new = w in args.new_widths

            if not w_path.exists():
                if not is_new:
                    print(f"  [WARN] w{w:4d}: {w_path} missing, skipping")
                    continue
                if not args.dry_run:
                    for split_dir in (f"{prefix}_png_train", f"{prefix}_png_test"):
                        for digit in range(10):
                            (w_path / split_dir / str(digit)).mkdir(parents=True, exist_ok=True)

            split_data = {
                "train": _entry(train_ids, w_path) if w == args.train_width else _empty(),
                "val": _entry(val_ids, w_path) if w == args.train_width else _empty(),
                "test": _entry(test_ids[w], w_path),
            }

            n = {k: len(v["path"]) for k, v in split_data.items()}
            tag = "NEW — fit required" if is_new else "fitted"
            if not args.dry_run:
                with open(w_path / args.out_split, "w") as f:
                    json.dump(split_data, f)
            print(f"  w{w:4d}: {n['train']:6,} train / {n['val']:5,} val / {n['test']:5,} test  [{tag}]")

    if args.dry_run:
        print("\n[dry run] nothing written.")
        return

    print(f"\nWrote '{args.out_split}' for {len(all_widths)} widths x {len(args.init_types)} init types.")
    print("Next: fit only the test INRs of the new widths, e.g.")
    for w in args.new_widths[:1]:
        print(f"  python scripts/generate_inrs.py --widths {w} --only-from-split "
              f"{args.data_root / f'w{w}_{ref_suffix}' / args.out_split}")


def _empty() -> dict:
    return {"path": [], "label": []}


def _entry(ids: list[ImageId], w_path: Path) -> dict:
    """Point a list of image identities at the INR files of one width."""
    paths, labels = [], []
    for split_dir, digit, stem in ids:
        paths.append(str((w_path / split_dir / digit / f"{stem}.pth").resolve()))
        labels.append(digit)
    return {"path": paths, "label": labels}


if __name__ == "__main__":
    main()
