<p align="center">
  <img src="https://raw.githubusercontent.com/hnnhvwht/frameguard/master/docs/img/frameguard-logo.png">
</p>

<h2 align="center">Validated Pandas DataFrames</h2>

FrameGuard is a wrapper class around the Pandas DataFrame that stores and manages schema to ensure the integrity of the underlying data. The FrameGuard API allows you to append instances to the underlying DataFrame&mdash;but only if they have been successfully validated against the schema. FrameGuard checks for:
- data type equality,
- boundary conditions (minimum/maximum) on numerical features,
- set membership for categorical features,
- regex pattern matching,
- and more!

FrameGuard is presently in the alpha stage and more features and tests are being developed actively. Please send bug reports and feature requests to the [author](mailto:hannah.white@tutanota.com) or post them as [issues](https://github.com/hnnhvwht/frameguard/issues).

## Quick Start

### Installation

Install FrameGuard, e.g., from PyPI:

```bash
$ python -m pip install frameguard
```

FrameGuard depends on `numpy`, `pandas` and `pyyaml`.

### Usage

In this example, we'll use the iris flower dataset:

```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
target = pd.DataFrame(iris["target"], columns=["species"])
df = pd.concat([df, target], axis=1)
```

We begin by importing and instantiating the `FrameGuard` class:

```python
from frameguard.frameguard import FrameGuard
fg = FrameGuard(df, auto_detect=True, categories=["species"])
```

```
Building schema...
=============================================================================
Schema for feature 'sepal length (cm)':
{'data_type': 'float64', 'allow_null': False}
=============================================================================
Schema for feature 'sepal width (cm)':
{'data_type': 'float64', 'allow_null': False}
=============================================================================
Schema for feature 'petal length (cm)':
{'data_type': 'float64', 'allow_null': False}
=============================================================================
Schema for feature 'petal width (cm)':
{'data_type': 'float64', 'allow_null': False}
=============================================================================
Schema for feature 'species':
{'data_type': 'int32', 'levels': array([0, 1, 2]), 'allow_null': False}
=============================================================================
Done! Created constraints for 5 features.
```

We instructed FrameGuard to generate the schema automatically, indicating that the `"species"` column represents a categorical variable.

So far, so good. Let's see what happens when we try to append bad data:

```python
batch = pd.DataFrame({
    "sepal length (cm)": [4.8, 5.2, 4.7],
    "sepal width (cm)": [3.3, 3.4, 3.0],
    "petal length (cm)": [1.4, 1.2, 1.3],
    "petal width (cm)": [0.2, 0.2, 0.3],
    "species": [0, 0, 3] # Bad target label
})
fg.append(batch)
```

```
---------------------------------------------------------------------------
[...]
ValidationError: Incorrect type for 'species' in batch.

During handling of the above exception, another exception occurred:
[...]
FrameGuardError: Batch does not satisfy schema. Operation cancelled...
```

Thus, the integrity of the underlying DataFrame is assured.

Presently, automatic schema detection is perhaps too simple for most real-world use cases. FrameGuard allows you to add and update constraints manually:

```python
fg = FrameGuard(df)
fg.add_constraint(
    features=[
      "sepal length (cm)",
      "sepal width (cm)",
      "petal length (cm)",
      "petal width (cm)"
    ],
    data_type="float64",
    allow_null=False
)
fg.add_constraint(
    features=["species"],
    data_type="int32",
    levels=[0, 1, 2],
    allow_null=False
)
```

Modifications to schemata will not be accepted if they do not match the data:

```python
fg.add_constraint(
    features=["species"],
    data_type="str",
    levels=["setosa", "versicolor", "virginica"],
    allow_null=False
)
```

```
SchemaWarning: Type mismatch for 'species'. Skipping...
```

When we're satisfied, we can export our schema in JSON or YAML form. By default, schema are exported to the current working directory in YAML format:

```python
fg.export_schema()
```

```
Schema exported successfully to schema-2020-11-21-162209.yml.
```

This is what the output looks like:

```yaml
features:
  petal length (cm):
    allow_null: false
    data_type: float64
  petal width (cm):
    allow_null: false
    data_type: float64
  sepal length (cm):
    allow_null: false
    data_type: float64
  sepal width (cm):
    allow_null: false
    data_type: float64
  species:
    allow_null: false
    data_type: int32
    levels:
    - 0
    - 1
    - 2
```

Just as well, we may import a schema after initialization. The DataFrame will be checked automatically against the schema provided that the schema was loaded correctly:

```python
fg = FrameGuard(df)
fg.import_schema("schema-2020-11-21-162209.yml")
```

```
Schema loaded successfully!
Validating DataFrame...

Checking feature 'sepal length (cm)'...
	Done checking feature 'sepal length (cm)'.
	Found 0 integrity violation(s).

Checking feature 'sepal width (cm)'...
	Done checking feature 'sepal width (cm)'.
	Found 0 integrity violation(s).

Checking feature 'petal length (cm)'...
	Done checking feature 'petal length (cm)'.
	Found 0 integrity violation(s).

Checking feature 'petal width (cm)'...
	Done checking feature 'petal width (cm)'.
	Found 0 integrity violation(s).

Checking feature 'species'...
	Done checking feature 'species'.
	Found 0 integrity violation(s).

Done validating DataFrame. Found 0 integrity violation(s).
```

Alternatively, if you have a schema in the form of a mapping, YAML or JSON object in memory, you could load it using the `load_schema()` method.

### Constraints

Presently, the following constraints are supported:
- `"data_type"` &ndash; the data type (NumPy types only);
- `"min"` &ndash; the minimum value for numerical features;
- `"max"` &ndash; the maximum value for numerical features;
- `"levels"` &ndash; the allowed levels for categorical features;
- `"pattern"` &ndash; a pattern for matching regular expressions;
- `"all_unique"` &ndash; whether duplicated values are permitted; and
- `"allow_null"` &ndash; whether null values are allowed.

## Planned Updates

- Write more tests and complete documentation
- Improve automatic detection of schema
- Add support for datetime detection and formatting
- Add support for conformity of numeric features to statistical distributions

## Authors

FrameGuard is written and maintained by [Hannah White](mailto:hannah.white@tutanota.com).

## Acknowledgements

The FrameGuard logo is set in Google's [Roboto Bold 700 Italic](https://fonts.google.com/specimen/Roboto).
