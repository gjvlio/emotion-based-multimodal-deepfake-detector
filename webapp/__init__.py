"""DeepSentinel web service — FastAPI app serving the latest trained detector.

The model is hot-reloaded automatically whenever the training pipeline writes a
newer checkpoint, so the web app is always "equipped" with the latest weights.
"""
