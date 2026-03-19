from ..DatasetDef import DatasetDef
import json


def load_from_file(file_path: str) -> list[DatasetDef]:
    with open(file_path) as f:
        raw = json.load(f)
    return _parse_datasets(raw)


def load_from_text(json_text: str) -> list[DatasetDef]:
    raw = json.loads(json_text)
    return _parse_datasets(raw)


def _parse_datasets(json_data: any) -> list[DatasetDef]:
    datasets = []
    for dataset in json_data:
        config = DatasetDef.model_validate(dataset)
        datasets.append(config)
    return datasets
