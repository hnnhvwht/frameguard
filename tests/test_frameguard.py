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
    "power": np.arange(sz, dtype=np.int64),
    "energy": rg.random((sz,)),
    "ice": rg.choice(np.array(["alpha", "beta", "gamma", "zeta"]), (sz,)),
    "plasma": [generate_string() for i in range(sz)]
}
df = pd.DataFrame(data)


test_schema = {
    "features": {
        "power": {
            "data_type": "int64",
            "all_unique": True,
            "allow_null": False
        },
        "energy": {
            "data_type": "float64",
            "min": 0,
            "max": 1,
            "allow_null": False
        },
        "ice": {
            "data_type": "object",
            "levels": ["alpha", "beta", "gamma", "zeta"],
            "allow_null": False
        },
        "plasma": {
            "data_type": "object",
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
    assert spec["power"]["data_type"] == "int64"
    assert spec["energy"]["data_type"] == "float64"
    assert spec["ice"]["data_type"] == "object"
    assert spec["plasma"]["data_type"] == "object"


def test_add_constraint():
    fg = FrameGuard(df)

    fg.add_constraint(
        ["power"],
        data_type="int64",
        all_unique=True,
        allow_null=False
    )

    fg.add_constraint(
        ["energy"],
        data_type="float64",
        min=0,
        max=1
    )

    fg.add_constraint(
        ["ice"],
        data_type="object",
        levels=["alpha", "beta", "gamma", "zeta"]
    )
    
    fg.add_constraint(
        ["plasma"],
        data_type="object",
        pattern=r"\w{4}-\w{4}"
    )

    assert fg._schema == test_schema


def test_append():
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
            continue


def test_remove():
    fg = FrameGuard(df, schema=test_schema)
    fg.remove([17, 18, 19], reset_index=True)
    assert len(fg._df) == 17


def test_load_schema():
    fg = FrameGuard(df)
    fg.load_schema(test_schema)
    assert len(fg._schema["features"].keys()) == 4


def test_import_schema():
    fg = FrameGuard(df)
    fg.import_schema("./tests/test_schema.yml")
    assert len(fg._schema["features"].keys()) == 4
