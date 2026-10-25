"""
Ruta de inferencia de producción: análisis de sentimiento de reseñas (WeLoveReviews).

Lee   data/raw/reviews.csv
Usa   nlptown/bert-base-multilingual-uncased-sentiment (versión fijada por commit)
Escribe data/processed/reviews_with_sentiment.csv

Uso (desde la raíz del repositorio):
    python src/app.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

# Silenciamos barras de progreso y avisos de descarga para tener una salida limpia.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import pipeline  # noqa: E402  (se importa tras configurar el entorno)
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_with_sentiment.csv"

# Modelo fijado: nombre + commit concreto del repositorio de Hugging Face.
# Así el resultado no cambia si el autor publica una versión nueva en "main".
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"

BATCH_SIZE = 32
REQUIRED_COLUMNS = {"review_id", "rating", "review_text"}


# ---------------------------------------------------------------------------
# Lógica de negocio
# ---------------------------------------------------------------------------

def stars_to_band(stars: int) -> str:
    """Mapea una puntuación de 1 a 5 estrellas a su banda de sentimiento."""
    if stars <= 2:
        return "Negativo"
    if stars == 3:
        return "Neutral"
    return "Positivo"


def load_reviews(path: Path) -> pd.DataFrame:
    """Carga el CSV de reseñas y valida que tenga las columnas esperadas."""
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {sorted(missing)}")
    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza mínima decidida en la EDA:
    - Recorta espacios del texto.
    - Descarta reseñas sin texto o con puntuación fuera de 1-5 (en este dataset no hay).
    - Mantiene los textos duplicados: tienen review_id distinto y cuentan como reseñas
      independientes dentro de las 500 que se entregan al cliente.
    """
    clean = df.copy()
    clean["review_text"] = clean["review_text"].astype(str).str.strip()
    clean = clean[clean["review_text"].ne("") & clean["rating"].between(1, 5)]
    clean["human_band"] = clean["rating"].map(stars_to_band)
    return clean.reset_index(drop=True)


def load_sentiment_model():
    """Carga el pipeline UNA sola vez. Los pesos van a la caché de HF, no al repo."""
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        revision=MODEL_REVISION,
        device=-1,  # CPU
    )


def predict(classifier, texts: list[str]) -> pd.DataFrame:
    """Inferencia por lotes reutilizando el mismo clasificador para todas las reseñas."""
    outputs = classifier(texts, batch_size=BATCH_SIZE, truncation=True)
    predictions = pd.DataFrame(outputs)  # columnas: label ("4 stars"), score
    predictions["pred_stars"] = predictions["label"].str.extract(r"(\d)").astype(int)
    predictions["sentiment_band"] = predictions["pred_stars"].map(stars_to_band)
    predictions["pred_score"] = predictions["score"].round(4)
    return predictions[["pred_stars", "sentiment_band", "pred_score"]]


def enrich(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Une reseñas y predicciones y marca los falsos negativos."""
    enriched = pd.concat([df.reset_index(drop=True), predictions], axis=1)
    enriched["is_false_negative"] = (enriched["rating"] >= 4) & (enriched["pred_stars"] <= 2)
    enriched["model_name"] = MODEL_NAME
    enriched["model_revision"] = MODEL_REVISION
    return enriched


def print_summary(enriched: pd.DataFrame) -> None:
    """Resumen corto para la consola."""
    order = ["Positivo", "Neutral", "Negativo"]
    model_pct = enriched["sentiment_band"].value_counts(normalize=True).reindex(order, fill_value=0) * 100
    human_pct = enriched["human_band"].value_counts(normalize=True).reindex(order, fill_value=0) * 100
    print("\nDistribución de sentimiento (%):")
    print(pd.DataFrame({"modelo": model_pct.round(1), "puntuación humana": human_pct.round(1)}).to_string())
    print(f"\nPromedio de estrellas humano: {enriched['rating'].mean():.2f}")
    print(f"Promedio de estrellas del modelo: {enriched['pred_stars'].mean():.2f}")
    print(f"Falsos negativos (rating 4-5, modelo 1-2): {int(enriched['is_false_negative'].sum())}")


def main() -> None:
    reviews = clean_reviews(load_reviews(RAW_PATH))
    print(f"Reseñas a procesar: {len(reviews)}")

    start = time.perf_counter()
    classifier = load_sentiment_model()
    predictions = predict(classifier, reviews["review_text"].tolist())
    print(f"Inferencia completada en {time.perf_counter() - start:.1f} s")

    enriched = enrich(reviews, predictions)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"Archivo escrito: {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(enriched)} filas)")

    print_summary(enriched)


if __name__ == "__main__":
    main()
