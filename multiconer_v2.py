#adapted from https://huggingface.co/datasets/MultiCoNER/multiconer_v2
#a pull request was created https://huggingface.co/datasets/MultiCoNER/multiconer_v2/discussions/2. If it is merged, this file can be deleted and the dataset can be loaded directly from the library
# coding=utf-8
"""SemEval 2023 Task 2: MultiCoNER II: Multilingual Complex Named Entity Recognition"""

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{multiconer2-report,
    title={{SemEval-2023 Task 2: Fine-grained Multilingual Named Entity Recognition (MultiCoNER 2)}},
    author={Fetahu, Besnik and Kar, Sudipta and Chen, Zhiyu and Rokhlenko, Oleg and Malmasi, Shervin},
    booktitle={Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)},
    year={2023},
    publisher={Association for Computational Linguistics},
}

@article{multiconer2-data,
    title={{MultiCoNER v2: a Large Multilingual dataset for Fine-grained and Noisy Named Entity Recognition}},
    author={Fetahu, Besnik and Chen, Zhiyu and Kar, Sudipta and Rokhlenko, Oleg and Malmasi, Shervin},
    year={2023},
}
"""

_DESCRIPTION = """\
Complex named entities (NE), like the titles of creative works, are not simple nouns and pose challenges for NER systems (Ashwini and Choi, 2014). They can take the form of any linguistic constituent, like an imperative clause (“Dial M for Murder”), and do not look like traditional NEs (Persons, Locations, etc.). This syntactic ambiguity makes it challenging to recognize them based on context. We organized the MultiCoNER task (Malmasi et al., 2022) at SemEval-2022 to address these challenges in 11 languages, receiving a very positive community response with 34 system papers. Results confirmed the challenges of processing complex and long-tail NEs: even the largest pre-trained Transformers did not achieve top performance without external knowledge. The top systems infused transformers with knowledge bases and gazetteers. However, such solutions are brittle against out of knowledge-base entities and noisy scenarios like the presence of spelling mistakes and typos. We propose MultiCoNER II which represents novel challenges through new tasks that emphasize the shortcomings of the current top models.

MultiCoNER II features complex NER in these languages:

1. English
2. Spanish
3. Hindi
4. Bangla
5. Chinese
6. Swedish
7. Farsi
8. French
9. Italian
10. Portugese
11. Ukranian
12. German

For more details see https://multiconer.github.io/

## References
* Sandeep Ashwini and Jinho D. Choi. 2014. Targetable named entity recognition in social media. CoRR, abs/1408.0782.
* Shervin Malmasi, Anjie Fang, Besnik Fetahu, Sudipta Kar, Oleg Rokhlenko. 2022. SemEval-2022 Task 11: Multilingual Complex Named Entity Recognition (MultiCoNER).

"""
_URL = "https://huggingface.co/datasets/MultiCoNER/multiconer_v2/resolve/main"

code_vs_lang_map = {"en": "English",
                    "es": "Spanish",
                    "pt": "Portuguese",
                    "uk": "Ukrainian",
                    "sv": "Swedish",
                    "fr": "French",
                    "fa": "Farsi",
                    "de": "German",
                    "zh": "Chinese",
                    "hi": "Hindi",
                    "bn": "Bangla",
                    "it": "Italian",
                    "multi": "Multilingual"}

label_vs_code_map = {"Bangla (BN)": 'bn',
                     "Chinese (ZH)": 'zh',
                     "English (EN)": 'en',
                     "Spanish (ES)": 'es',
                     "Swedish (SV)": 'sv',
                     "French (FR)": 'fr',
                     "Farsi (FA)": 'fa',
                     "German (DE)": 'de',
                     "Portuguese (PT)": 'pt',
                     "Hindi (HI)": 'hi',
                     "Italian (IT)": 'it',
                     "Ukrainian (UK)": 'uk',
                     "Multilingual (MULTI)": 'multi'}


class MultiCoNER2Config(datasets.BuilderConfig):
    """BuilderConfig for MultiCoNER2"""

    def __init__(self, **kwargs):
        """BuilderConfig for MultiCoNER2.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MultiCoNER2Config, self).__init__(**kwargs)


class MultiCoNER2(datasets.GeneratorBasedBuilder):
    """MultiCoNER2 dataset."""

    BUILDER_CONFIGS = [
        MultiCoNER2Config(name="Bangla (BN)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Bangla dataset"),
        MultiCoNER2Config(name="Chinese (ZH)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Chinese dataset"),
        MultiCoNER2Config(name="English (EN)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 English dataset"),
        MultiCoNER2Config(name="Farsi (FA)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Farsi dataset"),
        MultiCoNER2Config(name="French (FR)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 French dataset"),
        MultiCoNER2Config(name="German (DE)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 German dataset"),
        MultiCoNER2Config(name="Hindi (HI)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Hindi dataset"),
        MultiCoNER2Config(name="Italian (IT)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Italian dataset"),
        MultiCoNER2Config(name="Portuguese (PT)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Portuguese dataset"),
        MultiCoNER2Config(name="Spanish (ES)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Spanish dataset"),
        MultiCoNER2Config(name="Swedish (SV)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Swedish dataset"),
        MultiCoNER2Config(name="Ukrainian (UK)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Ukrainian dataset"),
        MultiCoNER2Config(name="Multilingual (MULTI)", version=datasets.Version("1.0.0"),
                          description="MultiCoNER2 Multilingual dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sample_id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.Value("string")),
                    "ner_tags_index": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'O',
                                'B-Facility',
                                'I-Facility',
                                'B-OtherLOC',
                                'I-OtherLOC',
                                'B-HumanSettlement',
                                'I-HumanSettlement',
                                'B-Station',
                                'I-Station',
                                'B-VisualWork',
                                'I-VisualWork',
                                'B-MusicalWork',
                                'I-MusicalWork',
                                'B-WrittenWork',
                                'I-WrittenWork',
                                'B-ArtWork',
                                'I-ArtWork',
                                'B-Software',
                                'I-Software',
                                'B-OtherCW',
                                'I-OtherCW',
                                'B-MusicalGRP',
                                'I-MusicalGRP',
                                'B-PublicCorp',
                                'I-PublicCorp',
                                'B-PrivateCorp',
                                'I-PrivateCorp',
                                'B-OtherCorp',
                                'I-OtherCorp',
                                'B-AerospaceManufacturer',
                                'I-AerospaceManufacturer',
                                'B-SportsGRP',
                                'I-SportsGRP',
                                'B-CarManufacturer',
                                'I-CarManufacturer',
                                'B-TechCORP',
                                'I-TechCORP',
                                'B-ORG',
                                'I-ORG',
                                'B-Scientist',
                                'I-Scientist',
                                'B-Artist',
                                'I-Artist',
                                'B-Athlete',
                                'I-Athlete',
                                'B-Politician',
                                'I-Politician',
                                'B-Cleric',
                                'I-Cleric',
                                'B-SportsManager',
                                'I-SportsManager',
                                'B-OtherPER',
                                'I-OtherPER',
                                'B-Clothing',
                                'I-Clothing',
                                'B-Vehicle',
                                'I-Vehicle',
                                'B-Food',
                                'I-Food',
                                'B-Drink',
                                'I-Drink',
                                'B-OtherPROD',
                                'I-OtherPROD',
                                'B-Medication/Vaccine',
                                'I-Medication/Vaccine',
                                'B-MedicalProcedure',
                                'I-MedicalProcedure',
                                'B-AnatomicalStructure',
                                'I-AnatomicalStructure',
                                'B-Symptom',
                                'I-Symptom',
                                'B-Disease',
                                'I-Disease'
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://multiconer.github.io",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}/{label_vs_code_map[self.config.name].upper()}-{code_vs_lang_map[label_vs_code_map[self.config.name]]}/{label_vs_code_map[self.config.name]}_train.conll",
            "dev": f"{_URL}/{label_vs_code_map[self.config.name].upper()}-{code_vs_lang_map[label_vs_code_map[self.config.name]]}/{label_vs_code_map[self.config.name]}_dev.conll",
            "test": f"{_URL}/{label_vs_code_map[self.config.name].upper()}-{code_vs_lang_map[label_vs_code_map[self.config.name]]}/{label_vs_code_map[self.config.name]}_test.conll",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)

        separator=' _ _ '

        with open(filepath) as f:
            guid = -1
            s_id = None
            tokens = []
            ner_tags = []

            for line in f:
                if line.strip().startswith("# id"):
                    s_id = line.split()[2].strip()
                    guid += 1
                    tokens = []
                    ner_tags = []
                elif separator in line:
                    # Separator is " _ _ "
                    splits = line.split(separator)
                    tokens.append(splits[0].strip())
                    ner_tags.append(splits[1].strip())
                elif len(line.strip()) == 0:
                    if s_id and len(tokens) >= 1 and len(tokens) == len(ner_tags):
                        yield guid, {
                            "id": guid,
                            "sample_id": s_id,
                            "tokens": tokens,
                            "ner_tags_index": ner_tags,
                            "ner_tags": ner_tags,
                        }
                        s_id = None
                        tokens = []
                        ner_tags = []
                    else:
                        continue

            if s_id and len(tokens) >= 1 and len(tokens) == len(ner_tags):
                yield guid, {
                    "id": guid,
                    "sample_id": s_id,
                    "tokens": tokens,
                    "ner_tags_index": ner_tags,
                    "ner_tags": ner_tags,
                }

