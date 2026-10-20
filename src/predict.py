import os
import pickle
import numpy as np
import tensorflow as tf
from src.preprocessing import clean_text


# --- Clase Predictora Unificada ---
class SentimentPredictor:
    """
    Clase unificada para cargar modelos entrenados y realizar inferencias.
    Soporta: Sklearn/Boosting (.pkl), TensorFlow SavedModel (directorio) y TensorFlow .keras.
    """
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model = None
        self.model_type = None  # 'tf_textcnn', 'tf_lstm', 'sklearn_or_boosting'
        self.vectorizer = None
        self.word_to_idx = None

    def load_model(self, model_name="best_model"):
        """
        Detecta PRIMERO qué archivo físico de modelo existe (best_model.keras,
        best_model/ como SavedModel, o best_model.pkl) y SOLO DESPUÉS carga
        el artefacto de vectorización que corresponde a ese tipo de modelo
        (vocab.pkl para redes con embeddings secuenciales, vectorizer.pkl
        -TF-IDF- para modelos clásicos de sklearn/boosting).
        """
        # 1. Metadata explícita de tipo de modelo, si existe
        meta_path = os.path.join(self.models_dir, "model_type.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.model_type = pickle.load(f)

        # 2. Detectar el archivo físico del modelo
        tf_savedmodel_path = os.path.join(self.models_dir, model_name)
        keras_path = os.path.join(self.models_dir, f"{model_name}.keras")
        pkl_path = os.path.join(self.models_dir, f"{model_name}.pkl")

        is_tf_savedmodel = os.path.isdir(tf_savedmodel_path)
        is_tf_keras = os.path.isfile(keras_path)
        is_sklearn = os.path.isfile(pkl_path)

        # 3. Cargar SOLO el artefacto de vectorización que corresponde
        if is_tf_savedmodel or is_tf_keras:
            # Modelo de red neuronal -> usa vocabulario para secuencias
            vocab_path = os.path.join(self.models_dir, "vocab.pkl")
            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as f:
                    self.word_to_idx = pickle.load(f)

            if self.model_type is None:
                # Si no hay metadata, asumimos LSTM por defecto para modelos TF
                self.model_type = 'tf_lstm'

        elif is_sklearn:
            # Modelo clásico -> usa TF-IDF
            vectorizer_path = os.path.join(self.models_dir, "vectorizer.pkl")
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)

            if self.model_type is None:
                self.model_type = 'sklearn_or_boosting'

        # 4. Cargar el modelo en sí, según lo detectado
        if is_tf_savedmodel:
            self.model = tf.keras.models.load_model(tf_savedmodel_path)
            print(f"Modelo TensorFlow SavedModel ({self.model_type}) cargado correctamente.")
            return True

        if is_tf_keras:
            self.model = tf.keras.models.load_model(keras_path)
            print(f"Modelo TensorFlow .keras ({self.model_type}) cargado correctamente.")
            return True

        if is_sklearn:
            with open(pkl_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Modelo Boosting/Sklearn ({self.model_type}) cargado correctamente.")
            return True

        print("Advertencia: No se encontró ningún modelo entrenado. Ejecutando en modo simulación.")
        return False

    def predict(self, text):
        """
        Realiza la limpieza del texto de entrada, aplica las transformaciones
        necesarias y ejecuta la predicción real sobre el modelo cargado.
        """
        is_rn = self.model_type != 'sklearn_or_boosting'
        cleaned = clean_text(text, model_rn=is_rn)
        if not cleaned:
            return {"sentiment": "Neutro / Inválido", "confidence": 0.0}

        # Modo Simulado (si no se ha entrenado/guardado ningún modelo aún)
        if self.model_type is None:
            import random
            sentiment = "Positivo" if random.random() > 0.5 else "Negativo"
            confidence = random.uniform(0.6, 0.99)
            return {
                "cleaned_text": cleaned,
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "simulated": True
            }

        # Modo Inferencia Real
        try:
            if self.model_type == 'sklearn_or_boosting':
                # Transformación TF-IDF
                x = self.vectorizer.transform([cleaned])
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(x)[0]
                    prob_pos = probs[1]
                    sentiment = "Positivo" if prob_pos >= 0.5 else "Negativo"
                    confidence = prob_pos if sentiment == "Positivo" else probs[0]
                else:
                    pred = self.model.predict(x)[0]
                    sentiment = "Positivo" if pred == 1 else "Negativo"
                    confidence = 1.0

            elif self.model_type == 'tf_textcnn':
                tokens = cleaned.split()
                seq = [self.word_to_idx.get(tok, 1) for tok in tokens]
                max_len = 100
                if len(seq) < max_len:
                    seq = seq + [0] * (max_len - len(seq))
                else:
                    seq = seq[:max_len]

                seq_array = np.array([seq])
                prob = float(self.model.predict(seq_array, verbose=0).flatten()[0])
                sentiment = "Positivo" if prob >= 0.5 else "Negativo"
                confidence = prob if sentiment == "Positivo" else (1.0 - prob)

            elif self.model_type == 'tf_lstm':
                tokens = cleaned.split()
                seq = [self.word_to_idx.get(tok, 1) for tok in tokens]
                max_len = 100
                if len(seq) < max_len:
                    seq = seq + [0] * (max_len - len(seq))
                else:
                    seq = seq[:max_len]

                seq_array = np.array([seq])
                prob = float(self.model.predict(seq_array, verbose=0).flatten()[0])
                sentiment = "Positivo" if prob >= 0.5 else "Negativo"
                confidence = prob if sentiment == "Positivo" else (1.0 - prob)

            return {
                "cleaned_text": cleaned,
                "sentiment": sentiment,
                "confidence": round(float(confidence), 4),
                "simulated": False
            }

        except Exception as e:
            return {"error": f"Error durante la inferencia: {str(e)}"}