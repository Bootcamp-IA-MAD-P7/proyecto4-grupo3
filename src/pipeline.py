import logging

from src.config import MERGED_DATA_PATH, FIGURES_DIR
from src.preprocess import run_preprocessing
from src.train import run_training

logger = logging.getLogger(__name__)


def main():
    border = "=" * 60
    logger.info(border)
    logger.info("  PROYECTO DE REGRESION - INCENDIOS FORESTALES")
    logger.info(border)

    logger.info("")
    logger.info(border)
    logger.info("  FASE 1: PREPROCESAMIENTO")
    logger.info(border)
    (X_train, X_test, y_train, y_test,
     preprocessor, target_transformer, te_mappings) = run_preprocessing(MERGED_DATA_PATH)

    logger.info("")
    logger.info(border)
    logger.info("  FASE 2: ENTRENAMIENTO Y EVALUACION")
    logger.info(border)
    feature_names = preprocessor.get_feature_names_out()
    run_training(
        X_train, X_test, y_train, y_test,
        preprocessor, target_transformer, feature_names,
    )

    logger.info("")
    logger.info(border)
    logger.info("  PROYECTO COMPLETADO")
    logger.info(border)
    logger.info("  Visualizaciones guardadas en: %s", FIGURES_DIR)
    logger.info("  Modelo guardado en: models/")


if __name__ == "__main__":
    main()
