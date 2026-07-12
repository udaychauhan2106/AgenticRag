import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
)

print("BASE URL:", client.base_url)

try:
    models = client.models.list()
    print([model.id for model in models.data])

except Exception as e:
    response = getattr(e, "response", None)

    print("ERROR TYPE:", type(e).__name__)

    if response is not None:
        print("STATUS:", response.status_code)
        print("REQUEST URL:", response.request.url)
        print("RESPONSE HEADERS:", dict(response.headers))
        print("RESPONSE BODY:", response.text)

    raise