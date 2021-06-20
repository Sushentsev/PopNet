from typing import List

import spacy
from spacy.tokens import Token


class SpacyRuTokenizer:
    def __init__(self):
        self.__spacy_ru = spacy.load("ru_core_news_md")

    def tokenize(self, text: str) -> List[Token]:
        return self.__spacy_ru.tokenizer(text)
