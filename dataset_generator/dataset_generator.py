from .Loaders import load_from_file, load_from_text
from .Generator import Generator
from .Exporter import export_to_arff, generate_arff_strings
from pathlib import Path


class DatasetGenerator:
    @staticmethod
    def generate(
        source: str, outpath: str = None, ret_arff: bool = False
    ) -> dict[str, any] | list[str] | None:
        if Path(source).is_file():
            dataset_defs = load_from_file(source)
        else:
            dataset_defs = load_from_text(source)

        datasets = {}
        for dataset_def in dataset_defs:
            generated = Generator.generate(dataset_def)
            datasets = datasets | generated

        if outpath:
            export_to_arff(datasets, outpath)
            return

        if not ret_arff:
            return datasets
        else:
            return generate_arff_strings(datasets)
