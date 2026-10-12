from __future__ import annotations

from cap_cop.preprocessing.clinical import full_structured_clinical_preprocessor


def build_preprocessor():
    return full_structured_clinical_preprocessor()
