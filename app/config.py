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


settings = Settings()
