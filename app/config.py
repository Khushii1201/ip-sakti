import os
import sys

from dotenv import load_dotenv

load_dotenv()

# The original raised a raw KeyError with no context on a missing env var --
# fine for you debugging locally, bad for a teammate running this for the
# first time (they'd see a bare traceback pointing at os.environ, not "you
# forgot to set DATABASE_URL"). Fail just as loud, but say what's actually
# missing and where to get it.
_REQUIRED = {
    "DATABASE_URL": "Supabase connection string (Session mode pooler recommended) - see README setup step 2.",
    "GROQ_API_KEY": "Free API key from console.groq.com - see README setup step 2.",
}


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        hint = _REQUIRED.get(name, "")
        print(f"FATAL: missing required environment variable {name}. {hint}", file=sys.stderr)
        print("Copy .env.example to .env and fill it in before starting the server.", file=sys.stderr)
        raise SystemExit(1)
    return value


class Settings:
    DATABASE_URL: str = _require("DATABASE_URL")  # Supabase postgres connection string
    GROQ_API_KEY: str = _require("GROQ_API_KEY")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RETRIEVAL_TOP_K: int = int(os.environ.get("RETRIEVAL_TOP_K", 8))
    # Below this, the bot abstains instead of guessing. Tune this against real
    # queries once you have a corpus - don't ship the default blind.
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.55))
    # Pool sizing: defaults are conservative for a hackathon demo (a handful
    # of judges hitting /query around the same time), not production load.
    DB_POOL_MIN_SIZE: int = int(os.environ.get("DB_POOL_MIN_SIZE", 2))
    DB_POOL_MAX_SIZE: int = int(os.environ.get("DB_POOL_MAX_SIZE", 10))
    DB_COMMAND_TIMEOUT_SECONDS: float = float(os.environ.get("DB_COMMAND_TIMEOUT_SECONDS", 10))


settings = Settings()
