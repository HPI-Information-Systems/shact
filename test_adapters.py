import unittest
import data_adapters
from datasets import load_dataset

class TestAdapters(unittest.TestCase):
    #write a test for each dataset in conll2003, genia, ontonotes, conll2000
    #each test loads the dataset, converts it to the normalized format and checks if it contains the right columns (id, tokens, spans)
    def test_conll2003_ner(self):
        dataset=load_dataset("conll2003")
        dataset=data_adapters.convert_conll03_ner_dataset(dataset)
        data_adapters.assert_columns(dataset)

    def test_conll2000_chunk(self):
        dataset=load_dataset("conll2000")
        dataset=data_adapters.convert_conll00_chunk_dataset(dataset)
        data_adapters.assert_columns(dataset)

    def test_genia(self):
        dataset=load_dataset("Rosenberg/genia")
        dataset=data_adapters.convert_genia_dataset(dataset)
        data_adapters.assert_columns(dataset)

    def test_ontonotes_en_ner(self):
        dataset=load_dataset("conll2012_ontonotesv5", "english_v12")
        dataset=data_adapters.convert_ontonotes_en_ner(dataset)
        data_adapters.assert_columns(dataset)

    def test_ontonotes_parse_trees(self):
        dataset=load_dataset("conll2012_ontonotesv5", "english_v12")
        dataset=data_adapters.convert_ontonotes_parse_trees(dataset)
        data_adapters.assert_columns(dataset)


if __name__ == '__main__':
    unittest.main()
