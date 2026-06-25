import os
from google import genai

# Using a dummy key just to see if the SDK forms the URL correctly or fails before network
# Or we can see the exact method it calls
try:
    client = genai.Client(api_key="AIzaSyDummyKey")
    # Try text-embedding-004
    client.models.embed_content(model="embedding-001", contents=["test", "test2"])
except Exception as e:
    print("Exception for text-embedding-004:", e)

