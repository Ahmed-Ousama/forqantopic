from litellm import embedding, model_cost
import pickle

def get_embeddings(texts: list[str] | str, model_name: str = "text-embedding-3-small") -> list[list[float]]:
    if isinstance(texts, str):
        texts = [texts]
    
    if model_name not in model_cost:
        raise ValueError(f"Embedding model '{model_name}' is not supported by LiteLLM.")
    
    response = embedding(model=model_name, input=texts)
    return [item['embedding'] for item in response['data']]

    
    # with open("embeddings.pkl", "rb") as f:
    #     embeddings = pickle.load(f)
    # return embeddings

def list_supported_embedding_models() -> list[str]:
    return list(model_cost["embedding"].keys())
