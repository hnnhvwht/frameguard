from datetime import datetime
import json
import numpy as np
import pandas as pd
import pathlib
import re
import warnings
import yaml


class FrameGuardError(Exception):
    pass


class FrameGuardWarning(UserWarning):
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

    @staticmethod
    def _check_features(features):
        """
        Check whether the features argument to `self.update_schema` is an
        iterable that is not a dictionary or string.
        """
        if (hasattr(type(features), "__len__") and
                not isinstance(features, (dict, str))):
            return True

    def _detect_schema(self, categories):
        print("Building schema...")
        print(77 * "=")
        self._schema = dict()
        self._schema["features"] = dict()
        for feature in self._df.columns:
            if feature not in self._schema["features"].keys():
                self._schema["features"][feature] = dict()
            spec = self._schema["features"][feature]
            spec["d_type"] = str(self._df[feature].dtype)
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
            assert batch[feature].dtype == np.dtype(spec["d_type"]), (
                f"Incorrect type for '{feature}' in batch."
            )

            if "minimum" in spec.keys():
                assert all(batch[feature] >= spec["minimum"]), (
                    f"At least one value is smaller than the allowed minimum "
                    f"for '{feature}'."
                )

            if "maximum" in spec.keys():
                assert all(batch[feature] <= spec["maximum"]), (
                    f"At least one value is greater than the allowed maximum "
                    f"for '{feature}'."
                )

            if "levels" in spec.keys():
                assert all(batch[feature].apply(
                    lambda x: x in spec["levels"])
                ), f"Unexpected level in categorical feature '{feature}'."

            if "pattern" in spec.keys():
                regex = re.compile(spec["pattern"])
                assert all(batch[feature].apply(
                    lambda x: regex.fullmatch(x))
                ), f"Can't match pattern to '{feature}' in batch."

            if "all_unique" in spec.keys() and spec["all_unique"]:
                result = pd.concat([self._df[feature], batch[feature]])
                assert not any(result.duplicated()), (
                    f"Duplicate value(s) detected in unique feature "
                    f"'{feature}'."
                )

            if "allow_null" in spec.keys() and not spec["allow_null"]:
                assert not any(batch[feature].isna()), (
                    f"Null value(s) detected in non-null feature '{feature}'."
                )

    def update_schema(
        self, features, documentation=None, d_type="auto",
        minimum=None, maximum=None, levels=None,
        pattern=None, all_unique=False, allow_null=False
    ):
        r"""
        Add constraints on one or more features.

        Parameters
        ----------
        features : array_like
            Feature(s) for which to create constraints
        documentation : str, optional, default: None
            Feature documentation
        d_type : str {"auto"} or numpy.dtype, optional, default: "auto"
            NumPy type that the feature should have
        minimum : numeric, optional, default: None
            Minimum value (lower bound) for numeric features
        maximum : numeric, optional, default: None
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
        if not FrameGuard._check_features(features):
            raise FrameGuardError(
                f"Passed feature(s) are not properly formed. "
                f"A {type(features)} was passed. "
                f"Please pass a list, tuple, set, numpy.ndarray, ..."
            )

        for feature in features:
            try:
                assert feature in self._df.columns
            except AssertionError:
                warnings.warn(
                    f"Feature '{feature}' not found in DataFrame. Skipping...",
                    FrameGuardWarning
                )
                continue

            if feature not in self._schema["features"].keys():
                self._schema["features"][feature] = dict()

            spec = self._schema["features"][feature]

            if documentation is not None:
                spec["documentation"] = documentation

            if d_type == "auto":
                spec["d_type"] = str(self._df[feature].dtype)
            else:
                if isinstance(d_type, str):
                    try:
                        _ = np.dtype(d_type)
                    except TypeError:
                        warnings.warn(
                            f"Failed to parse type for '{feature}'. "
                            f"Skipping...",
                            FrameGuardWarning
                        )
                        continue
                if isinstance(d_type, np.dtype):
                    d_type = str(d_type)
                try:
                    assert np.dtype(d_type) == self._df[feature].dtype
                except AssertionError:
                    warnings.warn(
                        f"Type mismatch for '{feature}'. Skipping...",
                        FrameGuardWarning
                    )
                    continue
                spec["d_type"] = d_type

            if minimum is not None:
                try:
                    assert pd.api.types.is_numeric_dtype(spec["d_type"])
                except AssertionError:
                    warnings.warn(
                        f"Cannot impose minimum value on non-numeric feature "
                        f"'{feature}'. Skipping...",
                        FrameGuardWarning
                    )
                    continue
                try:
                    assert all(self._df[feature] >= minimum)
                except AssertionError:
                    warnings.warn(
                        f"At least one value in '{feature}' is smaller than "
                        f"the allowed minimum value of {minimum}. Skipping...",
                        FrameGuardWarning
                    )
                    continue
                spec["minimum"] = minimum

            if maximum is not None:
                try:
                    assert pd.api.types.is_numeric_dtype(spec["d_type"])
                except AssertionError:
                    warnings.warn(
                        f"Cannot impose maximum value on non-numeric feature "
                        f"'{feature}'. Skipping...",
                        FrameGuardWarning
                    )
                    continue
                try:
                    assert all(self._df[feature] <= maximum)
                except AssertionError:
                    warnings.warn(
                        f"At least one value in '{feature}' is greater than "
                        f"the allowed maximum value of {maximum}. Skipping...",
                        FrameGuardWarning
                    )
                    continue
                if minimum is not None:
                    try:
                        assert minimum < maximum
                    except AssertionError:
                        warnings.warn(
                            f"The minimum value for '{feature}' is not smaller "
                            f"than the maximum value. Skipping...",
                            FrameGuardWarning
                        )
                        continue
                spec["maximum"] = maximum

            if levels is not None:
                try:
                    assert spec["d_type"] in ("int64", "object", "str")
                except AssertionError:
                    warnings.warn(
                        f"Cannot impose levels on non-string feature "
                        f"'{feature}' of type {spec['d_type']}. Skipping...",
                        FrameGuardWarning
                    )
                    continue
                try:
                    assert all(self._df[feature].apply(lambda x: x in levels))
                except AssertionError:
                    warnings.warn(
                        f"Unexpected level in categorical feature '{feature}'. "
                        f"Skipping...",
                        FrameGuardWarning
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
                        FrameGuardWarning
                    )
                    continue
                try:
                    self._df[feature].apply(lambda x: regex.match(x))
                except TypeError:
                    warnings.warn(
                        f"Can't match pattern for incorrectly typed feature "
                        f"'{feature}'. Skipping...",
                        FrameGuardWarning
                    )
                    continue
                spec["pattern"] = pattern

            if all_unique:
                try:
                    assert not any(self._df[feature].duplicated())
                except AssertionError:
                    warnings.warn(
                        f"Duplicate value(s) detected in unique feature "
                        f"'{feature}'. Skipping...",
                        FrameGuardWarning
                    )
                    continue
                spec["all_unique"] = True

            if not allow_null:
                try:
                    assert not any(self._df[feature].isna())
                except AssertionError:
                    warnings.warn(
                        f"Null value(s) detected in non-null feature "
                        f"'{feature}'. Skipping...",
                        FrameGuardWarning
                    )
                    continue
                spec["allow_null"] = False

    def validate(self):
        """
        Check the DataFrame against the schema for integrity violations.
        """
        assert hasattr(self, "_schema"), "No schema found."
        assert len(self._schema["features"]) > 0, "The stored schema is empty!"

        total_errors = 0
        print("Validating DataFrame...", end="\n\n")
        for feature in self._df.columns:
            feature_errors = 0
            print(f"Checking feature '{feature}'...")
            spec = self._schema["features"][feature]

            try:
                assert(self._df[feature].dtype == np.dtype(spec["d_type"]))
            except AssertionError:
                print(
                    f"\tTYPE: Found {str(self._df[feature].dtype)}, "
                    f"expected {spec['d_type']}"
                )
                feature_errors += 1

            if "minimum" in spec.keys():
                mask = self._df[feature] >= spec["minimum"]
                try:
                    assert all(mask)
                except AssertionError:
                    idx_str = str(self._df.loc[mask is False].index.values)
                    print(
                        f"\tMINIMUM: The value(s) at {idx_str} are smaller "
                        f"than the allowed minimum, {spec['minimum']}."
                    )
                    feature_errors += 1

            if "maximum" in spec.keys():
                mask = self._df[feature] <= spec["maximum"]
                try:
                    assert all(mask)
                except AssertionError:
                    idx_str = str(self._df.loc[mask is False].index.values)
                    print(
                        f"\tMAXIMUM: The value(s) at {idx_str} are greater "
                        f"than the allowed maximum, {spec['maximum']}."
                    )
                    feature_errors += 1

            if "levels" in spec.keys():
                mask = self._df[feature].apply(
                    lambda x: x in spec["levels"]
                )
                try:
                    assert all(mask)
                except AssertionError:
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
                try:
                    assert all(mask)
                except AssertionError:
                    idx_str = str(self._df.loc[mask is False].index.values)
                    print(
                        f"\tPATTERN: Pattern mismatch at {idx_str}."
                    )
                    feature_errors += 1

            if "all_unique" in spec.keys() and spec["all_unique"]:
                mask = self._df[feature].duplicated()
                try:
                    assert not all(mask)
                except AssertionError:
                    idx_str = str(self._df.loc[mask is False].index.values)
                    print(
                        f"\tALL UNIQUE: Duplicated value(s) at {idx_str}."
                    )
                    feature_errors += 1

            if "allow_null" in spec.keys() and not spec["allow_null"]:
                mask = self._df[feature].isna()
                try:
                    assert not all(mask)
                except AssertionError:
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
        assert isinstance(batch, pd.DataFrame), (
            f"Expected Pandas DataFrame for append operation, "
            f"got {type(batch)} instead."
        )
        try:
            self._validate_batch(batch)
        except AssertionError:
            raise FrameGuardError(
                "Batch does not satisfy schema. Cancelling operation..."
            )
        self._df = pd.concat([self._df, batch], ignore_index=True)
        print(f"Append operation successful."
              f"Added {len(batch)} instances.")

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

    def save_schema(self, path="./", fmt="yaml"):
        """
        Export the schema.

        Parameters
        ----------
        path : str
            The directory to which to dump the schema
        fmt : str {"yaml", "json"}, optional, default: "yaml"
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
            if fmt == "json":
                json.dump(self._schema, stream)
        print(f"Schema exported successfully to {fp}.")

    def load_schema(self, path):
        """
        Import a schema in YAML or JSON form.

        Parameters
        ----------
        path : str
            The file path from which to load the schema
        """
        extension = path.split(".")[-1]
        if extension not in ("yml", "yaml", "json"):
            raise FrameGuardError("Unrecognized file extension.")

        p = pathlib.Path(path)
        try:
            assert p.exists()
        except AssertionError:
            raise FrameGuardError("Cannot find file at given path.")

        with p.open() as stream:
            if extension in ("yml", "yaml"):
                schema = yaml.safe_load(stream)
            if extension == "json":
                schema = json.load(stream)

            try:
                assert "features" in schema.keys()
            except AssertionError:
                raise FrameGuardError("Failed to find features resource.")

            for feature in schema["features"].keys():
                for constraint in schema["features"][feature]:
                    try:
                        assert constraint in (
                            "documentation", "d_type", "minimum", "maximum",
                            "levels", "pattern", "all_unique", "allow_null"
                        )
                    except AssertionError:
                        raise FrameGuardError(
                            f"Bad schema: {constraint} is not a valid "
                            f"FrameGuard constraint."
                        )

            self._schema = schema
            print("Schema loaded successfully!")
            self.validate()
