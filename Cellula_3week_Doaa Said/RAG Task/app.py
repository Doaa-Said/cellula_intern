import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------------
# 1. Load dataset
# --------------------------
dataset = load_dataset("openai_humaneval")

# --------------------------
# 2. Load embedding model
# --------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# 3. Setup Chroma
# --------------------------
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="humanEval")

# --------------------------
# 4. Encode prompts and add to Chroma
# --------------------------
prompt_embeddings = embedding_model.encode(dataset["test"]["prompt"]).tolist()
collection.add(
    documents=list(dataset["test"]["prompt"]),
    embeddings=prompt_embeddings,
    ids=[str(i) for i in dataset["test"]["task_id"]]
)

print("✅ Dataset indexed in ChromaDB!")


model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --------------------------
# 7. New NLP task
# --------------------------
new_task = "Write a Python function to check if a string is a palindrome."

# Encode query & retrieve similar tasks
query_embedding = embedding_model.encode([new_task]).tolist()[0]
results = collection.query(query_embeddings=[query_embedding], n_results=3)

# --------------------------
# 8. Build context for RAG
# --------------------------
context = "\n\n".join(
    [f"Task: {d}" for d in results["documents"][0]]
)

prompt = f"""
You are a helpful coding assistant.

Here are some related tasks (short snippets only):
{context}

Now solve this new task:
{new_task}

Only provide the Python function code. Do NOT include any examples, docstrings, or extra text.
"""
# --------------------------
# 9. Generate free code with HF model
# --------------------------
output = generator(prompt, max_new_tokens=200, temperature=0.3, do_sample=True)

print("\n💡 Generated Code:\n")
print(output[0]["generated_text"])