from ..DatasetDef import DatasetDef, FeatureDrift, ClassFunc
from pandas import DataFrame
from scipy.stats import truncnorm
import random
import numpy as np
import math


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
    def generate_truncated(
        mean: float, std_dev: float, low: float, high: float, n: int
    ) -> np.array:
        a = (low - mean) / std_dev  # lower z-score
        b = (high - mean) / std_dev  # upper z-score

        dist = truncnorm(a, b, loc=mean, scale=std_dev)
        ret = dist.rvs(size=n)
        return ret

    @staticmethod
    def genetate_categorical(labels: list[str], probs: list[float], n: int) -> np.array:
        arr = random.choices(labels, probs, k=n)
        return np.array(arr)

    @staticmethod
    def generate(dataset: DatasetDef) -> dict[str, any]:
        res = {}
        for i in range(dataset.repetitions):
            name = f"{dataset.name}_s{dataset.seeds[i]}"
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
            if dist.type == "continuous":
                source = Generator.generate_truncated(
                    dist.dist_mean,
                    dist.dist_std,
                    dist.min_val,
                    dist.max_val,
                    num_of_samples,
                )
                if feature_type == "int":
                    source = np.round(source)
                data_sources.append(source)
            else:
                source = Generator.genetate_categorical(
                    dist.literals, dist.probabilities, num_of_samples
                )
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
                prev_start = 0 if i < 2 else drift_points[i - 1]
                result = np.concat(
                    [
                        result,
                        data_sources[source_index][
                            start - prev_start : size + prev_start
                        ],
                    ]
                )

                source_index += 1
            drift_state = not drift_state
        return result

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
                f_prev = eval(class_functions.functions[function_index - 1])
                f_next = eval(class_functions.functions[function_index])
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
                if not np.all(np.isin(tmp, [0, 1])):
                    raise Exception(
                        f"Invalid classification function(function {function_index}). All function must return class of 0 or 1."
                    )
                Y = np.concat([Y, tmp])
            else:
                try:
                    tmp = df[start:end].apply(
                        eval(class_functions.functions[function_index]), axis=1
                    )
                except Exception as e:
                    raise Exception(
                        f"Invalid classification function(function {function_index}). Error message: {e}"
                    )
                if not np.all(np.isin(tmp, [0, 1])):
                    raise Exception(
                        f"Invalid classification function(function {function_index}). All function must return class of 0 or 1."
                    )
                Y = np.concat([Y, tmp])

                function_index += 1
            drift_state = not drift_state

        return Y
