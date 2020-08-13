import math
import numpy as np
import pandas as pd
from frameguard.frameguard import FrameGuard, FrameGuardError

rg = np.random.default_rng(0)
sz = 20


def generate_string():
    charset = list(map(chr, range(48, 58))) + list(map(chr, range(97, 123)))
    s1 = "".join(rg.choice(charset, 4))
    s2 = "".join(rg.choice(charset, 4))
    return s1 + "-" + s2


data = {
    "power": np.arange(sz),
    "energy": rg.random((sz,)),
    "ice": rg.choice(np.array(["alpha", "beta", "gamma", "zeta"]), (sz,)),
    "plasma": [generate_string() for i in range(sz)]
}
df = pd.DataFrame(data)


test_schema = {
    "features": {
        "power": {
            "d_type": "int64",
            "all_unique": True,
            "allow_null": False
        },
        "energy": {
            "d_type": "float64",
            "minimum": 0,
            "maximum": 1,
            "allow_null": False
        },
        "ice": {
            "d_type": "object",
            "levels": ["alpha", "beta", "gamma", "zeta"],
            "allow_null": False
        },
        "plasma": {
            "d_type": "object",
            "pattern": r"\w{4}-\w{4}",
            "allow_null": False
        }
    }
}

test_batch_0 = pd.DataFrame({
    "power": [20, 21, 22],
    "energy": [0.63696169, 0.26978671, 0.04097352],
    "ice": ["alpha", "gamma", "gamma"],
    "plasma": ["umi9-b120", "6tnw-ilyq", "mjkx-9to0"]
})

test_batch_1 = pd.DataFrame({
    "power": ["AA", "AB", "AC"],
    "energy": [0.63696169, 0.26978671, 0.04097352],
    "ice": ["alpha", "gamma", "gamma"],
    "plasma": ["umi9-b120", "6tnw-ilyq", "mjkx-9to0"]
})

test_batch_2 = pd.DataFrame({
    "power": [20, 21, 22],
    "energy": [1.63696169, 1.26978671, 1.04097352],
    "ice": ["alpha", "gamma", "gamma"],
    "plasma": ["umi9-b120", "6tnw-ilyq", "mjkx-9to0"]
})

test_batch_3 = pd.DataFrame({
    "power": [20, 21, 22],
    "energy": [0.63696169, 0.26978671, 0.04097352],
    "ice": ["omega", "epsilon", "psi"],
    "plasma": ["umi9-b120", "6tnw-ilyq", "mjkx-9to0"]
})

test_batch_4 = pd.DataFrame({
    "power": [20, 21, 22],
    "energy": [0.63696169, 0.26978671, 0.04097352],
    "ice": ["alpha", "gamma", "gamma"],
    "plasma": ["u i9-b1 0", "6t w-ily ", "mjkx7-9to0a"]
})

test_batch_5 = pd.DataFrame({
    "power": [20, 21, None],
    "energy": [0.63696169, 0.26978671, 0.04097352],
    "ice": ["alpha", "gamma", "gamma"],
    "plasma": ["umi9-b120", "6tnw-ilyq", "mjkx-9to0"]
})

test_batch_6 = pd.DataFrame({
    "power": [20, 21, 22],
    "energy": [-0.63696169, -0.26978671, -0.04097352],
    "ice": ["alpha", "gamma", "gamma"],
    "plasma": ["umi9-b120", "6tnw-ilyq", "mjkx-9to0"]
})

test_batch_7 = pd.DataFrame({
    "power": [20, 20, 20],
    "energy": [0.63696169, 0.26978671, 0.04097352],
    "ice": ["alpha", "gamma", "gamma"],
    "plasma": ["umi9-b120", "6tnw-ilyq", "mjkx-9to0"]
})


def test_detect_schema():
    fg = FrameGuard(df, auto_detect=True)
    spec = fg._schema["features"]
    assert spec["power"]["d_type"] == "int64"
    assert spec["energy"]["d_type"] == "float64"
    assert spec["ice"]["d_type"] == "object"
    assert spec["plasma"]["d_type"] == "object"


def test_update_schema():
    fg = FrameGuard(df)
    fg.update_schema(
        ["power"], d_type="int64", all_unique=True, allow_null=False
    )
    fg.update_schema(
        ["energy"], d_type="float64", minimum=0, maximum=1
    )
    fg.update_schema(
        ["ice"], d_type="object", levels=["alpha", "beta", "gamma", "zeta"]
    )
    fg.update_schema(
        ["plasma"], d_type="object", pattern=r"\w{4}-\w{4}"
    )
    assert fg._schema == test_schema


def test_append_remove():
    fg = FrameGuard(df, schema=test_schema)
    fg.append(test_batch_0)
    assert len(fg._df) == 23

    for batch in [
        test_batch_1,
        test_batch_2,
        test_batch_3,
        test_batch_4,
        test_batch_5,
        test_batch_6,
        test_batch_7
    ]:
        try:
            fg.append(batch)
        except FrameGuardError:
            assert len(fg._df) == 23

    fg.remove([20, 21, 22], reset_index=True)
    assert len(fg._df) == 20
