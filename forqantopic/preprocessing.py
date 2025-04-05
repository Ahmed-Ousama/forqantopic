def clean_texts(texts: list[str]) -> list[str]:
    cleaned = []

    for text in texts:
        text = text.strip().lower()
        if len(text.split()) <= 1:  # Remove one-word texts
            continue
        if text in [".", "", "..."]:
            continue
        cleaned.append(text)

    # Remove duplicates
    return list(cleaned)
