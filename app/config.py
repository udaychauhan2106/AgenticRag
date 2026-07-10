import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")
    QUADRANT_API_KEY=os.environ.get("QUADRANT_API_KEY")
    QUADRANT_URL=os.environ.get("QUADRANT_CLUSTER_ENDPOINT")
    QUADRANT_COLLECTION="enterprise_rag"

    GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    GROQ_MODEL="llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY=os.environ.get("GROQ_FALLBACK_API_KEY")


settings=Settings()
