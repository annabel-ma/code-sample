#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import hashlib
import tempfile
from pathlib import Path

import torch

from utils import ( 
    extract_disjoint_sets,
    prepare_bispec,
    fashion_mnist,
    kmnist,
    mnist,
    usps,
    test_ot_once,
)


NUM = 2500

CACHE_ROOT = Path(
    os.environ.get(
        "DT_CACHE_DIR",
        "/n/holylabs/LABS/dam_lab/Lab/annabelma/experiments/cache"
    )
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

CACHE_VERSION = "v4"
FORCE_REBUILD = os.environ.get("FORCE_REBUILD") == "1"

def _key_to_path(tag: str, params: dict) -> Path:
    payload = {"tag": tag, "version": CACHE_VERSION, **params}
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_ROOT / f"{tag}__{h}.pt"


def load_or_build(tag: str, params: dict, builder):
    """Load cached object if present, else build and cache atomically."""
    path = _key_to_path(tag, params)
    if (not FORCE_REBUILD) and path.exists():
        return torch.load(path, map_location="cpu")
    obj = builder()
    with tempfile.NamedTemporaryFile(dir=str(CACHE_ROOT), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        torch.save(obj, tmp_path)
    tmp_path.replace(path)
    return obj


def _disjoint_cached(name: str, ds, *, N: int, random: bool, angle=None):
    params = {"N": int(N), "random": bool(random), "angle": None if angle is None else int(angle)}
    return load_or_build(
        f"{name}_disjoint",
        params,
        lambda: extract_disjoint_sets(ds, N=N, random=random, angle=angle if angle is not None else 60),
    )


def _bispec_from_split(tag: str, data_list):
    return load_or_build(tag, {"split": True}, lambda: prepare_bispec(data_list))


def _bispec_from_tv(tag: str, ds, *, angle=None, N=None, random=False):
    params = {
        "source": getattr(ds, "__class__", type("X",(object,),{})).__name__,
        "angle": None if angle is None else int(angle),
        "N": None if N is None else int(N),
        "random": bool(random),
    }
    return load_or_build(tag, params, lambda: prepare_bispec(ds, angle=angle, N=N, random=random))


def build_dataset_by_key(key: str):

    if key in ("fashion_unrotated", "fashion_rotated",
               "fashion_baseline_labels", "fashion_baseline_tests"):
        fash_sets = _disjoint_cached("fashion_random", fashion_mnist, N=NUM, random=True, angle=None)
        fash_label, fash_test = fash_sets['labelset'], fash_sets['testset']
        fash_base_sets = _disjoint_cached("fashion_baseline", fashion_mnist, N=NUM, random=False, angle=0)
        fash_base_label, fash_base_test = fash_base_sets['labelset'], fash_base_sets['testset']

        if key == "fashion_unrotated":
            return _bispec_from_split("fashion_unrotated", fash_label)
        if key == "fashion_rotated":
            return _bispec_from_split("fashion_rotated", fash_test)
        if key == "fashion_baseline_labels":
            return _bispec_from_split("fashion_baseline_labels", fash_base_label)
        if key == "fashion_baseline_tests":
            return _bispec_from_split("fashion_baseline_tests", fash_base_test)

    if key in ("kmnist_unrotated", "kmnist_rotated",
               "kmnist_baseline_labels", "kmnist_baseline_tests"):
        km_sets = _disjoint_cached("kmnist_random", kmnist, N=NUM, random=True, angle=None)
        km_label, km_test = km_sets['labelset'], km_sets['testset']
        km_base_sets = _disjoint_cached("kmnist_baseline", kmnist, N=NUM, random=False, angle=0)
        km_base_label, km_base_test = km_base_sets['labelset'], km_base_sets['testset']

        if key == "kmnist_unrotated":
            return _bispec_from_split("kmnist_unrotated", km_label)
        if key == "kmnist_rotated":
            return _bispec_from_split("kmnist_rotated", km_test)
        if key == "kmnist_baseline_labels":
            return _bispec_from_split("kmnist_baseline_labels", km_base_label)
        if key == "kmnist_baseline_tests":
            return _bispec_from_split("kmnist_baseline_tests", km_base_test)

    if key in ("mnist_unrotated", "mnist_rotated",
               "mnist_baseline_labels", "mnist_baseline_tests"):
        m_sets = _disjoint_cached("mnist_random", mnist, N=NUM, random=True, angle=None)
        m_label, m_test = m_sets['labelset'], m_sets['testset']
        m_base_sets = _disjoint_cached("mnist_baseline", mnist, N=NUM, random=False, angle=0)
        m_base_label, m_base_test = m_base_sets['labelset'], m_base_sets['testset']

        if key == "mnist_unrotated":
            return _bispec_from_split("mnist_unrotated", m_label)
        if key == "mnist_rotated":
            return _bispec_from_split("mnist_rotated", m_test)
        if key == "mnist_baseline_labels":
            return _bispec_from_split("mnist_baseline_labels", m_base_label)
        if key == "mnist_baseline_tests":
            return _bispec_from_split("mnist_baseline_tests", m_base_test)

    if key == "mnist_usps_0":
        # Your original: prepare_bispec(mnist, angle=0, N=542)
        return _bispec_from_tv("mnist_usps_0", mnist, angle=0, N=542, random=False)

    if key == "usps_0":
        # Your original: prepare_bispec(usps, angle=0, N=542) 
        return _bispec_from_tv("usps_0", usps, angle=0, N=542, random=False)

    if key == "mnist_usps_rotated":
        # Your original: prepare_bispec(usps, N=542, random=True)
        return _bispec_from_tv("mnist_usps_rotated", usps, angle=None, N=542, random=True)

    raise ValueError(f"Unknown dataset key: {key}")

