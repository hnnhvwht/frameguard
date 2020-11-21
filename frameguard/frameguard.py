from datetime import datetime
import collections
import json
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import yaml


class FrameGuardError(Exception):
    pass


class ValidationError(FrameGuardError):
    pass


class SchemaWarning(UserWarning):
    pass


class FrameGuard:
    """
    Stores a validated pandas.DataFrame and an associated schema in dictionary
    format.

    Parameters
    ----------
    df : pandas.DataFrame
        The underlying DataFrame for which to create a schema
    auto_detect : bool, optional default: False
        Whether to automatically detect feature constraints upon initialization

    Attributes
    ----------
    self._df : pandas.DataFrame
    self._schema : dict
    """
    _constraint_types = ("data_type", "min", "max", "levels", "pattern", 
        "all_unique", "allow_null")

    def __init__(self, df, schema=None, auto_detect=False, categories=None):
        self._df = df
        if schema is not None:
            self._schema = schema
            self.validate()
        elif schema is None and auto_detect:
            self._detect_schema(categories=categories)
        else:
            self._schema = dict()
            self._schema["features"] = dict()

    def _detect_schema(self, categories):
        print("Building schema...")
        print(77 * "=")
        self._schema = dict()
        self._schema["features"] = dict()
        for feature in self._df.columns:
            if feature not in self._schema["features"].keys():
                self._schema["features"][feature] = dict()
            spec = self._schema["features"][feature]
            spec["data_type"] = str(self._df[feature].dtype)
            if hasattr(self._df[feature], "cat"):
                spec["levels"] = self._df[feature].unique()
            elif categories is not None and feature in categories:
                spec["levels"] = self._df[feature].unique()
            if self._df[feature].isna().sum() == 0:
                spec["allow_null"] = False
            print(f"Schema for feature '{feature}':\n{spec}")
            print(77 * "=")
        n_features = len(self._schema["features"])
        print(f"Done! Created constraints for {n_features} features.")

    def _validate_batch(self, batch):
        for feature in self._df.columns:
            spec = self._schema["features"][feature]
            if not batch[feature].dtype == np.dtype(spec["data_type"]):
                raise ValidationError(
                    f"Incorrect type for '{feature}' in batch."
                )

            if "min" in spec.keys():
                if not all(batch[feature] >= spec["min"]):
                    raise ValidationError(
                        f"At least one value is smaller than the allowed "
                        f"min for '{feature}'."
                    )

            if "max" in spec.keys():
                if not all(batch[feature] <= spec["max"]):
                    raise ValidationError(
                        f"At least one value is greater than the allowed "
                        f"max for '{feature}'."
                    )

            if "levels" in spec.keys():
                if not all(batch[feature].apply(
                    lambda x: x in spec["levels"])
                ):
                    raise ValidationError(
                        f"Unexpected level in categorical feature '{feature}'."
                    )

            if "pattern" in spec.keys():
                regex = re.compile(spec["pattern"])
                if not all(batch[feature].apply(
                    lambda x: regex.fullmatch(x))
                ):
                    raise ValidationError(
                        f"Can't match pattern to '{feature}' in batch."
                    )

            if "all_unique" in spec.keys() and spec["all_unique"]:
                result = pd.concat([self._df[feature], batch[feature]])
                if any(result.duplicated()):
                    raise ValidationError(
                        f"Duplicate value(s) detected in unique feature "
                        f"'{feature}'."
                )

            if "allow_null" in spec.keys() and not spec["allow_null"]:
                if any(batch[feature].isna()):
                    raise ValidationError(
                        f"Null value(s) detected in non-null feature "
                        f"'{feature}'."
                )

    @staticmethod
    def _validate_resources(schema):
        if "features" not in schema.keys():
            raise FrameGuardError("Failed to find features resource.")

        for feature in schema["features"].keys():
            for constraint in schema["features"][feature]:
                if constraint not in FrameGuard._constraint_types:
                    raise FrameGuardError(
                        f"Bad schema: {constraint} is not a valid "
                        f"FrameGuard constraint."
                    )

    def add_constraint(
        self, features, data_type="auto", min=None, max=None, levels=None,
        pattern=None, all_unique=False, allow_null=False
    ):
        r"""
        Add constraint on one or more features.

        Parameters
        ----------
        features : array_like
            Feature(s) for which to create constraints
        data_type : str {"auto"} or numpy.dtype, optional, default: "auto"
            NumPy type that the feature should have
        min : numeric, optional, default: None
            Minimum value (lower bound) for numeric features
        max : numeric, optional, default: None
            Maximum value (upper bound) for numeric features
        levels : array_like, optional, default: None
            Permitted levels for a categorical feature
        pattern : str, optional, default: None
            Regular expression for pattern matching for string features
            Example: "^\+\d{2} \d \d{4} \d{4}$" to match international numbers
        all_unique : bool, optional, default: False
            Whether all values must be unique
        allow_null : bool, optional, default: False
            Whether NA values are permitted
        """
        if not isinstance(features, collections.abc.Sequence):
            raise FrameGuardError(
                f"Passed feature(s) are not properly formed. "
                f"A {type(features)} was passed. "
                f"Please pass a sequence of features."
            )

        for feature in features:
            if feature not in self._df.columns:
                warnings.warn(
                    f"Feature '{feature}' not found in DataFrame. "
                    f"Skipping...",
                    SchemaWarning
                )
                continue

            if feature not in self._schema["features"].keys():
                self._schema["features"][feature] = dict()

            spec = self._schema["features"][feature]

            if data_type == "auto":
                spec["data_type"] = str(self._df[feature].dtype)
            else:
                if isinstance(data_type, str):
                    try:
                        _ = np.dtype(data_type)
                    except TypeError:
                        warnings.warn(
                            f"Failed to parse type for '{feature}'. "
                            f"Skipping...",
                            SchemaWarning
                        )
                        continue
                if isinstance(data_type, np.dtype):
                    data_type = str(data_type)
                if not np.dtype(data_type) == self._df[feature].dtype:
                    warnings.warn(
                        f"Type mismatch for '{feature}'. Skipping...",
                        SchemaWarning
                    )
                    continue
                spec["data_type"] = data_type

            if min is not None:
                if not pd.api.types.is_numeric_dtype(spec["data_type"]):
                    warnings.warn(
                        f"Cannot impose minimum value on non-numeric feature "
                        f"'{feature}'. Skipping...",
                        SchemaWarning
                    )
                    continue
                if not all(self._df[feature] >= min):
                    warnings.warn(
                        f"At least one value in '{feature}' is smaller than "
                        f"the allowed minimum value of {min}. Skipping...",
                        SchemaWarning
                    )
                    continue
                spec["min"] = min

            if max is not None:
                if not pd.api.types.is_numeric_dtype(spec["data_type"]):
                    warnings.warn(
                        f"Cannot impose maximum value on non-numeric feature "
                        f"'{feature}'. Skipping...",
                        SchemaWarning
                    )
                    continue
                if not all(self._df[feature] <= max):
                    warnings.warn(
                        f"At least one value in '{feature}' is greater than "
                        f"the allowed maximum value of {max}. Skipping...",
                        SchemaWarning
                    )
                    continue
                if min is not None:
                    if not min < max:
                        warnings.warn(
                            f"The minimum value for '{feature}' is not "
                            f"smaller than the maximum value. Skipping...",
                            SchemaWarning
                        )
                        continue
                spec["max"] = max

            if levels is not None:
                if (spec["data_type"] not in
                    ("int32", "int64", "object", "str")):
                    warnings.warn(
                        f"Cannot impose levels on feature '{feature}' of type "
                        f"{spec['data_type']}. Skipping...",
                        SchemaWarning
                    )
                    continue
                if not all(self._df[feature].apply(lambda x: x in levels)):
                    warnings.warn(
                        f"Unexpected level in categorical feature "
                        f"'{feature}'. Skipping...",
                        SchemaWarning
                    )
                    continue
                if not isinstance(levels, list):
                    levels = list(levels)
                spec["levels"] = levels

            if pattern is not None:
                try:
                    regex = re.compile(pattern)
                except re.error:
                    warnings.warn(
                        f"Failed to compile regular expression for {feature}. "
                        f"Skipping...",
                        SchemaWarning
                    )
                    continue
                try:
                    self._df[feature].apply(lambda x: regex.match(x))
                except TypeError:
                    warnings.warn(
                        f"Can't match pattern for incorrectly typed feature "
                        f"'{feature}'. Skipping...",
                        SchemaWarning
                    )
                    continue
                spec["pattern"] = pattern

            if all_unique:
                if any(self._df[feature].duplicated()):
                    warnings.warn(
                        f"Duplicate value(s) detected in unique feature "
                        f"'{feature}'. Skipping...",
                        SchemaWarning
                    )
                    continue
                spec["all_unique"] = True

            if not allow_null:
                if any(self._df[feature].isna()):
                    warnings.warn(
                        f"Null value(s) detected in non-null feature "
                        f"'{feature}'. Skipping...",
                        SchemaWarning
                    )
                    continue
                spec["allow_null"] = False

    def validate(self):
        """
        Check the DataFrame against the schema for integrity violations.
        """
        if not hasattr(self, "_schema"):
            raise SchemaWarning("No schema found.")
        
        if not len(self._schema["features"]) > 0:
            raise SchemaWarning("The stored schema is empty!")

        total_errors = 0
        print("Validating DataFrame...", end="\n\n")
        for feature in self._df.columns:
            feature_errors = 0
            print(f"Checking feature '{feature}'...")
            spec = self._schema["features"][feature]

            if not self._df[feature].dtype == np.dtype(spec["data_type"]):
                print(
                    f"\tTYPE: Found {str(self._df[feature].dtype)}, "
                    f"expected {spec['data_type']}"
                )
                feature_errors += 1

            if "min" in spec.keys():
                mask = self._df[feature] >= spec["min"]
                if not all(mask):
                    idx_str = str(self._df.loc[mask is False].index.values)
                    print(
                        f"\tMINIMUM: The value(s) at {idx_str} are smaller "
                        f"than the allowed minimum, {spec['min']}."
                    )
                    feature_errors += 1

            if "max" in spec.keys():
                mask = self._df[feature] <= spec["max"]
                if not all(mask):
                    idx_str = str(self._df.loc[mask is False].index.values)
                    print(
                        f"\tMAXIMUM: The value(s) at {idx_str} are greater "
                        f"than the allowed maximum, {spec['max']}."
                    )
                    feature_errors += 1

            if "levels" in spec.keys():
                mask = self._df[feature].apply(
                    lambda x: x in spec["levels"]
                )
                if not all(mask):
                    idx_str = str(self._df.loc[mask is False].index.values)
                    print(
                        f"\tLEVELS: Unexpected level(s) at indices {idx_str} "
                        f"for categorical feature."
                    )
                    feature_errors += 1

            if "pattern" in spec.keys():
                regex = re.compile(spec["pattern"])
                mask = self._df[feature].apply(
                    lambda x: isinstance(regex.fullmatch(x), re.Match)
                )
                if not all(mask):
                    idx_str = str(self._df.loc[mask is False].index.values)
                    print(
                        f"\tPATTERN: Pattern mismatch at {idx_str}."
                    )
                    feature_errors += 1

            if "all_unique" in spec.keys() and spec["all_unique"]:
                mask = self._df[feature].duplicated()
                if any(mask):
                    idx_str = str(self._df.loc[mask is True].index.values)
                    print(
                        f"\tALL UNIQUE: Duplicated value(s) at {idx_str}."
                    )
                    feature_errors += 1

            if "allow_null" in spec.keys() and not spec["allow_null"]:
                mask = self._df[feature].isna()
                if any(mask):
                    idx_str = str(self._df.loc[mask is True].index.values)
                    print(
                        f"\tALLOW NULL: Null value(s) at {idx_str}."
                    )
                    feature_errors += 1

            print(f"\tDone checking feature '{feature}'.\n"
                  f"\tFound {feature_errors} integrity violation(s).")

            if feature_errors > 0:
                total_errors += feature_errors
            print()

        print(f"Done validating DataFrame. Found {total_errors} integrity "
              f"violation(s).")

    def access(self):
        """
        Access a copy of the underlying DataFrame.
        """
        return self._df.copy()

    def append(self, batch):
        """
        Validate and append a batch of instances to the underlying DataFrame.

        Parameters
        ----------
        batch : pandas.DataFrame
        """
        try:
            self._validate_batch(batch)
        except ValidationError:
            raise FrameGuardError(
                "Batch does not satisfy schema. Operation cancelled..."
            )
        try:
            self._df = pd.concat([self._df, batch], ignore_index=True)
        except TypeError:
            raise FrameGuardError(
                "Could not append batch due to a type mismatch. "
                "Expected a Pandas DataFrame."
        )
        print(
            f"Append operation successful. "
            f"Added {len(batch)} instances."
        )

    def remove(self, index, reset_index=False):
        """
        Drop instances by index.

        Parameters
        ----------
        index : scalar or array_like
            Index identifying the instances to drop
        reset_index
            Whether to reset the index and discard the old index
        """
        self._df.drop(index=index, inplace=True)
        if reset_index:
            self._df.reset_index(drop=True, inplace=True)

    def export_schema(self, path="./", fmt="yml"):
        """
        Export the schema.

        Parameters
        ----------
        path : str, optional, default: "./
            The directory to which to dump the schema
        fmt : str {"yml", "yaml", "json"}, optional, default: "yml"
            The desired schema format
        """
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        file_name = ("schema-"
                     + datetime.now().strftime("%Y-%m-%d-%H%M%S")
                     + "." + fmt)
        fp = p / file_name
        with fp.open("w") as stream:
            if fmt in ("yml", "yaml"):
                yaml.dump(self._schema, stream)
            elif fmt == "json":
                json.dump(self._schema, stream)
        print(f"Schema exported successfully to {fp}.")

    def load_schema(self, obj, validate=True):
        """
        Load a schema from a mapping or a JSON or YSML object.

        Parameters
        ----------
        obj : mapping, str
            The object to load as a schema
        validate : bool, optional, default: True
            Whether to validate the schema after importing it
        """
        try:
            schema = yaml.safe_load(obj)
        except AttributeError:
            if isinstance(obj, collections.abc.Mapping):
                schema = obj
            else:
                raise FrameGuardError(
                    "Failed to parse object. "
                    "Please provide a YAML or JSON object or a mapping."
                )
        
        FrameGuard._validate_resources(schema)
        self._schema = schema
        print("Schema loaded successfully!")
        if validate:
            self.validate()

    def import_schema(self, path, validate=True):
        """
        Import a schema in YAML or JSON form.

        Parameters
        ----------
        path : str
            The file path from which to load the schema
        validate : bool, optional, default: True
            Whether to validate the schema after importing it
        """
        extension = path.split(".")[-1]
        if extension not in ("yml", "yaml", "json"):
            raise FrameGuardError("Unrecognized file extension.")

        p = pathlib.Path(path)
        if not p.exists():
            raise FrameGuardError("Cannot find file at given path.")

        with p.open() as stream:
            if extension in ("yml", "yaml"):
                schema = yaml.safe_load(stream)
            elif extension == "json":
                schema = json.load(stream)

            FrameGuard._validate_resources(schema)
            self._schema = schema
            print("Schema loaded successfully!")
            if validate:
                self.validate()
