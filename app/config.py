import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")
    QUADRANT_API_KEY=os.environ.get("QUADRANT_API_KEY")
    QUADRANT_URL=os.environ.get("QUADRANT_CLUSTER_ENDPOINT")
    QUADRANT_COLLECTION="enterprise_rag"

    GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    GROQ_MODEL="openai/gpt-oss-120b"
    GROQ_PLANNER_MODEL="openai/gpt-oss-20b"
    GROQ_FALLBACK_API_KEY=os.environ.get("GROQ_FALLBACK_API_KEY")

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "rag"     
    GROQ_SLUG_2 = "brag"
    
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-project")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag-project")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings=Settings()
