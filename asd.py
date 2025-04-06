from groq import Groq

client = Groq(api_key="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")
models = client.models.list()
print(models)