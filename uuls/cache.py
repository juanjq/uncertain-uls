"""Results keyed on the parameters that produced them.

Two ways to use it:

    result = cached("background_lambda", params, lambda: expensive(**params))
        Returns the stored result if `params` was seen before, otherwise computes,
        stores and returns it. The result must be picklable.

    folder = cache_dir("simulst_dl3", params)
    if not is_done(folder):
        ...write files into folder...
        mark_done(folder)
        For steps that produce files (DL3 folders, FITS maps) instead of a Python object.

The key is the SHA-1 of the parameters written as canonical JSON. Paths in the parameters
are replaced by (path, size, modification time) of the file, or of every file under a
folder, so editing an input invalidates the entry. Each entry folder holds `params.json`,
readable by eye, next to the stored result.

Nothing here knows about the physics. Changing the code of a step does not invalidate its
entries: add a `"version"` item to `params` and bump it when the output would change.
"""
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from . import paths

_DONE = ".done"


def _file_signature(p: Path):
    st = p.stat()
    return [str(p), st.st_size, int(st.st_mtime)]


def _normalise(obj):
    """Turn `obj` into plain JSON types, deterministically."""
    if isinstance(obj, Path):
        p = obj.expanduser().resolve()
        if p.is_file():
            return {"file": _file_signature(p)}
        if p.is_dir():
            return {"dir": [_file_signature(f) for f in sorted(p.rglob("*")) if f.is_file()]}
        return {"path": str(p)}
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {"ndarray_sha1": hashlib.sha1(np.ascontiguousarray(obj).tobytes()).hexdigest(),
                "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "unit") and hasattr(obj, "value"):  # astropy Quantity
        return {"value": _normalise(np.asarray(obj.value).tolist()), "unit": str(obj.unit)}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise TypeError(f"cache: cannot use a {type(obj).__name__} as a parameter; pass plain values or paths")


def key(params: dict) -> str:
    """Hex key of a parameter dict (first 12 characters of the SHA-1)."""
    text = json.dumps(_normalise(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def cache_dir(name: str, params: dict, root: Path | None = None) -> Path:
    """Entry folder for step `name` with `params`. Created, with params.json, if missing."""
    folder = Path(root or paths.CACHE) / name / key(params)
    folder.mkdir(parents=True, exist_ok=True)
    pfile = folder / "params.json"
    if not pfile.exists():
        pfile.write_text(json.dumps(_normalise(params), indent=2, sort_keys=True))
    return folder


def is_done(folder: Path) -> bool:
    return (Path(folder) / _DONE).exists()


def mark_done(folder: Path) -> None:
    (Path(folder) / _DONE).touch()


def cached(name: str, params: dict, compute, root: Path | None = None, verbose: bool = True):
    """Return compute()'s result for `params`, computing it only the first time."""
    folder = cache_dir(name, params, root)
    rfile = folder / "result.pkl"
    if is_done(folder) and rfile.exists():
        if verbose:
            print(f"[cache] {name}/{folder.name}: loaded")
        with open(rfile, "rb") as f:
            return pickle.load(f)
    result = compute()
    tmp = rfile.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(rfile)
    mark_done(folder)
    if verbose:
        print(f"[cache] {name}/{folder.name}: computed and stored")
    return result


def clear(name: str, params: dict | None = None, root: Path | None = None) -> None:
    """Delete one entry (`params` given) or every entry of step `name`."""
    base = Path(root or paths.CACHE) / name
    target = base / key(params) if params is not None else base
    if target.exists():
        shutil.rmtree(target)
