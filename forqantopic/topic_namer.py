from litellm import completion
import json

def name_clusters_with_llm(cluster_texts: dict[int, list[str]], model_name: str = "gpt-4o-mini") -> tuple[dict[int, str], list[str]]:
    """
    Returns:
    - cluster_titles: Dict of cluster_id -> inferred topic name from LLM (based on original clusters)
    - llm_labels: List of new topics generated by the LLM after re-clustering or reorganizing the provided text samples
    """
    clusters_text = ""
    for cluster_id, samples in cluster_texts.items():
        clusters_text += f"\nCluster {cluster_id}:\n"
        for text in samples:
            clusters_text += f"- {text.strip()}\n"


    
    
    prompt_phase1 = f"""
    You will be given several clusters of text samples.  
    Each cluster represents a group of semantically similar texts.

    Your task: assign a **short, clear, distinct title (max 4 words)** to each cluster, using the cluster ID as reference.  
    DO NOT merge, split, or drop clusters. Just name them.

    Format:
    {{
    "cluster_titles": {{
        "0": "Weather Forecasts",
        "1": "Pet Ownership"
    }}
    }}

    Clusters:
    {clusters_text}
    """

    prompt_phase2 = f"""
    You will be given several clusters of text samples.  
    Each cluster represents a group of semantically similar texts.


    Your task: reorganize them freely into your own semantic groups.  
    You can merge similar topics, split broad ones, and discard noisy texts.

    Limit: Try to keep the number of topics between 4 and 6 unless clearly necessary.

    Return a list of topic titles only:
    {{
    "llm_labels": [
        "Weather & Climate",
        "Pets & Animals",
        "Cooking & Meals"
    ]
    }}

    Clusters:

    {clusters_text}
    """
    
    
    # Phase 1: Name clusters
    phase1_response = completion(model=model_name, messages=[{"role": "user", "content": prompt_phase1}])
    phase1_content = phase1_response["choices"][0]["message"]["content"].strip()
    
    # Phase 2: Reorganize clusters
    phase2_response = completion(model=model_name, messages=[{"role": "user", "content": prompt_phase2}])
    phase2_content = phase2_response["choices"][0]["message"]["content"].strip()


    
    # Parse the JSON response from the LLM
    try:
        phase1_json_block = phase1_content.split("```json")[-1].split("```", 1)[0].strip()
        phase1_parsed = json.loads(phase1_json_block)
        cluster_titles = {int(i): label for i, label in phase1_parsed["cluster_titles"].items()}
    except Exception as e:
        raise ValueError(f"Failed to parse LLM output.\nContent: {phase1_content}\nError: {e}")
    
    # Parse the JSON response from the LLM
    try:
        phase2_json_block = phase2_content.split("```json")[-1].split("```", 1)[0].strip()
        phase2_parsed = json.loads(phase2_json_block)
        llm_labels = phase2_parsed["llm_labels"]
    except Exception as e:
        raise ValueError(f"Failed to parse LLM output.\nContent: {phase2_content}\nError: {e}")
    
    

    return cluster_titles, llm_labels