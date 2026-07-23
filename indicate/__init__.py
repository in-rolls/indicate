from .hindi2english import HindiToEnglish
from .indic_utils import detect_indic_script, detect_language_from_script
from .llm_indic import IndicLLMTransliterator
from .punjabi2english import PunjabiToEnglish

hindi2english = HindiToEnglish.transliterate
punjabi2english = PunjabiToEnglish.transliterate

__all__ = [
    "IndicLLMTransliterator",
    "detect_indic_script",
    "detect_language_from_script",
    "hindi2english",
    "punjabi2english",
]
