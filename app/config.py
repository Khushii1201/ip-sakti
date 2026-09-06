import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RETRIEVAL_TOP_K: int = int(os.environ.get("RETRIEVAL_TOP_K", "8"))
    # Below this, the bot abstains instead of guessing. Tune this against real
    # queries once you have a corpus - don't ship the default blind.
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.55"))


settings = Settings()
