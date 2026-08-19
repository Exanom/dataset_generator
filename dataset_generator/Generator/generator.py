from ..DatasetDef import DatasetDef, FeatureDrift, ClassFunc
from pandas import DataFrame
from scipy.stats import truncnorm
import random
import numpy as np
import math
import ast
from asteval import Interpreter


class UnsafeExpressionError(Exception):
    """Raised when a user expression contains disallowed syntax."""


class Generator:

    @staticmethod
    def sigmoid_vectorized(indices, p, w):
        x = -4.0 * (indices - p) / w
        x_clipped = np.clip(x, None, 700)
        return 1.0 / (1.0 + np.exp(x_clipped))

    @staticmethod
    def set_global_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def generate_truncated_norm(
        mean: float, std_dev: float, low: float, high: float, n: int
    ) -> np.array:
        a = (low - mean) / std_dev  # lower z-score
        b = (high - mean) / std_dev  # upper z-score

        dist = truncnorm(a, b, loc=mean, scale=std_dev)
        ret = dist.rvs(size=n)
        return ret

    @staticmethod
    def generate_categorical(labels: list[str], probs: list[float], n: int) -> np.array:
        arr = random.choices(labels, probs, k=n)
        return np.array(arr)

    @staticmethod
    def generate_constant(value: float | str, n: int) -> np.array:
        arr = np.full(n, fill_value=value)
        return arr

    @staticmethod
    def generate_random(min_val: float, max_val: float, n: int) -> np.array:
        arr = np.random.uniform(min_val, max_val, size=(n))
        return arr

    @staticmethod
    def generate(dataset: DatasetDef) -> dict[str, any]:
        res = {}
        for i in range(dataset.repetitions):
            name = f"{dataset.name}_s{dataset.seeds[i]}"
            print(f"GENERATING {name}")
            Generator.set_global_seed(dataset.seeds[i])
            column_names = []
            columns_vals = []
            for feature in dataset.features:
                feature_data = Generator.generate_feature_vector(
                    feature.data_dist, feature.type, dataset.samples
                )
                column_names.append(feature.name)
                columns_vals.append(feature_data)

            df = DataFrame(dict(zip(column_names, columns_vals)))
            Y = Generator.generate_labels_vector(
                dataset.class_func, df, dataset.samples
            )
            df["class"] = Y
            dataset_tmp = dataset.model_copy(deep=True)
            dataset_tmp.repetitions = 1
            dataset_tmp.seeds = [dataset.seeds[i]]
            tmp = {"df": df, "meta": dataset_tmp}
            res[name] = tmp
        return res

    def generate_feature_vector(
        feature_data: FeatureDrift, feature_type: str, samples: int
    ) -> np.array:
        # All relevant drift points, structure:
        # 0, start1, end1, start2, end2, start3, end3, samples
        drift_points = [0]

        # All drift start and end points(this structure helps with generating data streams), structure:
        #   start_points            end_points
        #           0                           end1
        #       start1                       end2
        #       start2                       end3
        #       start3                       samples
        drift_start_ponts = [0]
        drift_end_points = []

        for drift in feature_data.drift_defs:
            start = drift.center - math.ceil(drift.window / 2)
            end = drift.center + math.ceil(drift.window / 2)
            drift_points.append(start)
            drift_points.append(end)

            drift_start_ponts.append(start)
            drift_end_points.append(end)
        drift_points.append(samples)
        drift_end_points.append(samples)

        # Generating data sources of different distributions. Must contain enough samples until drift involving them has ended.
        # Legend: - data points, * drift center, | drift point
        #
        #                                 0              start1               end1                   start2                             end2                           samples
        # dataset definition:    |------------|-------*-------|-------------------|------------*------------|------------------------|
        #  Distribution 1:        -----------------------------
        #  Distribution 2:                           -------------------------------------------------------------
        #  Distribution 3:                                                                               ---------------------------------------------------
        data_sources = []
        for i, dist in enumerate(feature_data.distributions):
            num_of_samples = drift_end_points[i] - drift_start_ponts[i]
            if dist.type == "normal":
                source = Generator.generate_truncated_norm(
                    dist.dist_mean,
                    dist.dist_std,
                    dist.min_val,
                    dist.max_val,
                    num_of_samples,
                )
            elif dist.type == "categorical":
                source = Generator.generate_categorical(
                    dist.literals, dist.probabilities, num_of_samples
                )
            elif dist.type == "constant":
                source = Generator.generate_constant(dist.value, num_of_samples)
            elif dist.type == "uniform":
                source = Generator.generate_random(
                    dist.min_val, dist.max_val, num_of_samples
                )

            if feature_type == "int":
                source = np.round(source)
            data_sources.append(source)

        drift_state = False
        result = np.array([])
        source_index = 0

        # Populating the final feature vector with mixed distributions
        #
        #                                 0              start1               end1                   start2                             end2                           samples
        # dataset definition:    |------------|-------*-------|-------------------|------------*------------|------------------------|
        #                            D1 -------------                  D2 --------------------                                D3 ------------------------
        #                                        D1+D2 ----------------                D2+D3 --------------------------
        #
        #
        for i in range(len(drift_points) - 1):
            start = drift_points[i]
            end = drift_points[i + 1]
            size = end - start

            if drift_state:
                weights = Generator.sigmoid_vectorized(
                    np.arange(start, end),
                    feature_data.drift_defs[source_index - 1].center,
                    feature_data.drift_defs[source_index - 1].window,
                )
                tmp = np.where(
                    weights >= np.random.uniform(0, 1, size=weights.shape),
                    data_sources[source_index - 1][-size:],  # last n samples
                    data_sources[source_index][:size],  # first  n samples
                )
                result = np.concat([result, tmp])

            else:
                prev_start = 0 if i < 2 else drift_points[i] - drift_points[i - 1]
                result = np.concat(
                    [
                        result,
                        data_sources[source_index][prev_start : prev_start + size],
                    ]
                )

                source_index += 1
            drift_state = not drift_state
        return result

    @staticmethod
    def validateFunction(function: str) -> None:

        FORBIDDEN_NODES = (
            ast.Import,
            ast.ImportFrom,
            ast.Lambda,
            ast.FunctionDef,
            ast.ClassDef,
            ast.With,
            ast.Global,
            ast.Nonlocal,
            ast.Delete,
            ast.Assign,
            ast.AugAssign,
            ast.Attribute,
        )
        try:
            tree = ast.parse(function, mode="eval")
        except SyntaxError as e:
            raise UnsafeExpressionError(f"Invalid syntax: {e}")

        for node in ast.walk(tree):
            if isinstance(node, FORBIDDEN_NODES):
                raise UnsafeExpressionError(
                    f"Disallowed construct in expression: {type(node).__name__}"
                )
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise UnsafeExpressionError(f"Disallowed identifier: {node.id}")

    @staticmethod
    def safeEval(function: str) -> np.array:
        SAFE_SYMBOLS = {
            # math module functions
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "log2": math.log2,
            "log10": math.log10,
            "pi": math.pi,
            "e": math.e,
            "floor": math.floor,
            "ceil": math.ceil,
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            # a few numpy elementwise ops, useful if a "row" carries array-like fields
            "np_clip": np.clip,
            "np_sign": np.sign,
        }
        Generator.validateFunction(function)

        def f(row):
            interpreter = Interpreter(
                usersyms={**SAFE_SYMBOLS, **row.to_dict()},
                use_numpy=False,
                minimal=True,
            )
            result = interpreter(function)
            if interpreter.error:
                raise UnsafeExpressionError(
                    f"Error evaluating {function}: {interpreter.error}"
                )
            return result

        return f

    @staticmethod
    def generate_labels_vector(
        class_functions: ClassFunc, df: DataFrame, samples: int
    ) -> np.array:
        drift_points = [0]

        for drift in class_functions.drift_defs:
            start = drift.center - math.ceil(drift.window / 2)
            end = drift.center + math.ceil(drift.window / 2)
            drift_points.append(start)
            drift_points.append(end)
        drift_points.append(samples)

        drift_state = False
        function_index = 0
        Y = np.array([])
        for i in range(len(drift_points) - 1):
            start = drift_points[i]
            end = drift_points[i + 1]
            if drift_state:
                weights = Generator.sigmoid_vectorized(
                    np.arange(start, end),
                    class_functions.drift_defs[function_index - 1].center,
                    class_functions.drift_defs[function_index - 1].window,
                )
                f_prev = Generator.safeEval(
                    class_functions.functions[function_index - 1]
                )
                f_next = Generator.safeEval(class_functions.functions[function_index])
                try:
                    tmp = np.array(
                        [
                            f_prev(row) if np.random.random() < w else f_next(row)
                            for (_, row), w in zip(df[start:end].iterrows(), weights)
                        ]
                    )
                except Exception as e:
                    raise Exception(
                        f"Invalid classification function(either function {function_index-1} or {function_index}). Error message: {e}"
                    )
                Y = np.concat([Y, tmp])
            else:
                try:
                    tmp = df[start:end].apply(
                        Generator.safeEval(class_functions.functions[function_index]),
                        axis=1,
                    )
                except Exception as e:
                    raise Exception(
                        f"Invalid classification function(function {function_index}). Error message: {e}"
                    )
                Y = np.concat([Y, tmp])

                function_index += 1
            drift_state = not drift_state

        return Y
