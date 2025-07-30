# nlp/translator.py

from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # Ensures stable language detection results

class SmartTranslator:
    def __init__(self):
        pass

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception:
            return "unknown"

    def translate_to_english(self, text):
        lang = self.detect_language(text)
        if lang != "en":
            try:
                translated = GoogleTranslator(source='auto', target='en').translate(text)
                return translated
            except Exception as e:
                print(f"[SmartTranslator] Translation failed: {e}")
                return text
        else:
            return text
