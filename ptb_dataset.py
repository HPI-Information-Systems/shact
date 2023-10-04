#adapted from https://huggingface.co/datasets/MultiCoNER/multiconer_v2
"""PTB version from https://github.com/sustcsonglin/pointer-net-for-nested
The official implementation of ACL2022: Bottom-Up Constituency Parsing and Nested Named Entity Recognition with Pointer Networks
@misc{yang2021bottomup,
      title={Bottom-Up Constituency Parsing and Nested Named Entity Recognition with Pointer Networks}, 
      author={Songlin Yang and Kewei Tu},
      year={2021},
      eprint={2110.05419},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

import datasets
from datasets import BuilderConfig, ClassLabel, Sequence
from pathlib import Path

#load .env file
from dotenv import dotenv_values
env_config = dotenv_values(".env")

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
"""

_DESCRIPTION = """\
PTB
"""
_BASE_URL = Path(env_config['EXTRA_DATASETS']) / "constituency_parsing" / "ptb"
_TRAINING_FILE = "02-21.10way.clean.txt"
_DEV_FILE = "22.auto.clean.txt"
_TEST_FILE = "23.auto.clean.txt"


class MultiCoNER2(datasets.GeneratorBasedBuilder):
    """PTB dataset."""

    BUILDER_CONFIGS = [
        BuilderConfig(name="PTB", version=datasets.Version("1.0.0"),
                          description=_DESCRIPTION),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tree": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/sustcsonglin/pointer-net-for-nested",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": _BASE_URL / _TRAINING_FILE,
            "validation": _BASE_URL / _DEV_FILE,
            "test": _BASE_URL / _TEST_FILE,
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["validation"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)

        with open(filepath) as f:

            for i,line in enumerate(f):
                yield i, {
                    "id": str(i),
                    "tree": line.strip(),
                }

