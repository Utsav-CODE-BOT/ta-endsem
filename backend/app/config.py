import os


class Settings:
    PROJECT_NAME: str = "VoiceMiner"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./backend/voiceminer.db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    SENTIMENT_MODEL: str = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    DEFAULT_CLUSTERS: int = 6
    SIMILARITY_THRESHOLD: float = 0.35
    CORS_ORIGINS: list = ["*"]


settings = Settings()
