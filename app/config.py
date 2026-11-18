import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QUADRANT_API_KEY = os.getenv("QUADRANT_API_KEY")
    QUADRANT_URL = os.getenv("QUADRANT_CLUSTER_ENDPOINT")
    QUADRANT_COLLECTION = "enterprise_rag"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GROQ_PLANNER_MODEL = os.getenv("GROQ_PLANNER_MODEL", "openai/gpt-oss-20b")

    LIVERAG_MAX_EXTRACTED_TEXT_BYTES = int(
        os.getenv("LIVERAG_MAX_EXTRACTED_TEXT_BYTES", str(500 * 1024))
    )
    LIVERAG_MAX_CONTENT_BLOCKS = int(
        os.getenv("LIVERAG_MAX_CONTENT_BLOCKS", "1000")
    )
    LIVERAG_EXTRACTION_MAX_FUTURE_SKEW_SECONDS = int(
        os.getenv("LIVERAG_EXTRACTION_MAX_FUTURE_SKEW_SECONDS", "300")
    )
    LIVERAG_EXTRACTION_MAX_AGE_DAYS = int(
        os.getenv("LIVERAG_EXTRACTION_MAX_AGE_DAYS", "30")
    )


settings = Settings()
