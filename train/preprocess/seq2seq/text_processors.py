from typing import List

from spacy.tokens import Token


class SpacyTextPreprocessor:
    def preprocess(self, text: List[Token]) -> List[str]:
        return [token.text.lower().strip() for token in text if token.is_alpha()]
