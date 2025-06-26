from openai import OpenAI

client = OpenAI(
  api_key="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",
  base_url="https://integrate.api.nvidia.com/v1"
)

response = client.embeddings.create(
    input=["What is the capital of France?"],
    model="baai/bge-m3",
    encoding_format="float",
    extra_body={"truncate": "NONE"}
)

print(response.data[0].embedding)
