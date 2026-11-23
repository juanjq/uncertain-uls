import numpy as np
import pytest

from uuls import cache


def test_same_params_compute_once(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"x": np.arange(3)}

    p = {"a": 1, "b": [1.5, "s"]}
    r1 = cache.cached("step", p, compute, root=tmp_path, verbose=False)
    r2 = cache.cached("step", p, compute, root=tmp_path, verbose=False)
    assert len(calls) == 1
    np.testing.assert_array_equal(r1["x"], r2["x"])


def test_key_ignores_dict_order_and_sees_values():
    assert cache.key({"a": 1, "b": 2}) == cache.key({"b": 2, "a": 1})
    assert cache.key({"a": 1}) != cache.key({"a": 2})


def test_editing_an_input_file_changes_the_key(tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("one")
    k1 = cache.key({"f": f})
    f.write_text("two, longer")
    assert cache.key({"f": f}) != k1


def test_files_entry_and_clear(tmp_path):
    d = cache.cache_dir("dl3", {"seed": 1}, root=tmp_path)
    assert (d / "params.json").exists() and not cache.is_done(d)
    cache.mark_done(d)
    assert cache.is_done(d)
    cache.clear("dl3", {"seed": 1}, root=tmp_path)
    assert not d.exists()


def test_unsupported_param_type_raises():
    with pytest.raises(TypeError):
        cache.key({"obj": object()})
