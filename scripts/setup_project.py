from pathlib import Path

folders = [
    "app/api",
    "app/ingestion",
    "app/processors",
    "app/embeddings",
    "app/vectordb",
    "app/retrieval",
    "app/core",
    "data/text",
    "data/audio",
    "data/video",
    "data/processed",
    "tests",
    "scripts",
    "docs",
    "chroma_db"
]

files = [
    "app/main.py",

    "app/api/upload.py",
    "app/api/search.py",
    "app/api/health.py",

    "app/ingestion/text_ingestor.py",
    "app/ingestion/audio_ingestor.py",
    "app/ingestion/video_ingestor.py",

    "app/processors/transcription.py",
    "app/processors/video_processor.py",
    "app/processors/image_captioner.py",
    "app/processors/text_processor.py",

    "app/embeddings/embedding_service.py",
    "app/embeddings/models.py",

    "app/vectordb/chroma_client.py",
    "app/vectordb/repository.py",

    "app/retrieval/search_service.py",
    "app/retrieval/ranking.py",

    "app/core/config.py",
    "app/core/logging.py",
    "app/core/constants.py",

    "tests/test_ingestion.py",
    "tests/test_embeddings.py",
    "tests/test_search.py",

    "scripts/seed_data.py",
    "scripts/reset_db.py",

    "docs/architecture.md",
    "docs/report.md",

    ".env",
]
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

for file in files:
    Path(file).touch(exist_ok=True)

for py_file in Path("app").rglob("*.py"):
    py_file.write_text("# TODO\n")

print("Project structure created successfully!")