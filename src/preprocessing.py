import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = ENGLISH_STOP_WORDS


def clean_text(text, model_rn=True):
    """
    Función básica para limpiar el texto de las reseñas de Amazon.
    :param model_rn: bool, si es True (por defecto), NO elimina stopwords (ideal para Redes Neuronales).
    si es False, elimina stopwords (ideal para modelos Boosting/TF-IDF).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convertir a minúsculas
    text = text.lower()
    
    # 2. Eliminar etiquetas HTML
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 4. Eliminar stopwords (solo si NO es para modelo RN)
    if not model_rn:
        tokens = text.split()
        tokens = [
            tok for tok in tokens
            if tok.strip(string.punctuation) not in STOP_WORDS
        ]
        text = " ".join(tokens)
    
    # 5. Sustituir puntuación y caracteres especiales por espacio
    # Así no se pegan palabras (great!product → great product)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 6. Remover espacios extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def parse_fasttext_line(line):
    """
    Parsea una línea del dataset de Amazon Reviews en formato fastText.
    Ejemplo: __label__2 Great product! -> (1, "Great product!") (Positivo)
            __label__1 Bad product! -> (0, "Bad product!") (Negativo)
    """
    if line.startswith("__label__1"):
        label = 0
        text = line.replace("__label__1", "", 1).strip()
    elif line.startswith("__label__2"):
        label = 1
        text = line.replace("__label__2", "", 1).strip()
    else:
        label = None
        text = line.strip()
    return label, text
