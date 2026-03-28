from pathlib import Path
from ..DatasetDef import DatasetDef


def generate_arff_strings(datasets: dict[str, DatasetDef]) -> dict[str, str]:
    res = {}
    for name, dataset in datasets.items():
        arff_str = ""
        arff_str += f"@RELATION {name}\n\n"
        meta: DatasetDef = dataset["meta"]

        arff_str += (
            "% for numeric: (mean ; std ; min-max) -> <drift_center ; drift_width>\n"
        )
        arff_str += "% for categorical: [prob1, prob2, prob3, ...] -> <drift_center ; drift_width>\n\n"

        for feature in meta.features:
            if feature.type == "str":
                dist_info = ""
                values = set()
                for i, dist in enumerate(feature.data_dist.distributions):
                    vals = dist.literals
                    values.update(vals)
                    tmp = f"{dist.probabilities} "
                    if len(feature.data_dist.drift_defs) > i:
                        drift = feature.data_dist.drift_defs[i]
                        tmp += f"-> <{drift.center} ; {drift.window}> -> "
                    dist_info += tmp
                arff_str += f"%{dist_info} \n"
                arff_str += f"@ATTRIBUTE {feature.name} {values}\n\n"
            else:
                dist_info = ""
                for i in range(len(feature.data_dist.distributions)):
                    dist = feature.data_dist.distributions[i]
                    tmp = f"({dist.dist_mean} ; {dist.dist_std} ; {dist.min_val} - {dist.max_val}) "
                    if len(feature.data_dist.drift_defs) > i:
                        drift = feature.data_dist.drift_defs[i]
                        tmp += f"-> <{drift.center} ; {drift.window}> -> "
                    dist_info += tmp
                arff_str += f"%{dist_info} \n"
                arff_str += f"@ATTRIBUTE {feature.name} {'integer' if feature.type=='int' else 'real'}\n\n"

        for i, dist in enumerate(meta.class_func.functions):
            arff_str += f"%{dist}\n"
            if len(meta.class_func.drift_defs) > i:
                drift = meta.class_func.drift_defs[i]
                arff_str += f"%<{drift.center} ; {drift.window}> \n"
        arff_str += f"@ATTRIBUTE class integer\n"

        df = dataset["df"]
        arff_str += "\n@DATA\n"
        for _, row in df.iterrows():
            arff_str += ",".join(str(v) for v in row) + "\n"
        res[name] = arff_str
    return res


def export_to_arff(datasets: dict[str, any], out_path: str = "results") -> None:
    arff_strs = generate_arff_strings(datasets)
    for name in datasets:
        Path(out_path).mkdir(exist_ok=True, parents=True)
        filepath = Path(f"{out_path}/{name}.arff")
        with open(filepath, "w") as f:
            f.write(arff_strs[name])
