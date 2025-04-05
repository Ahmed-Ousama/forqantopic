from forqantopic.preprocessing import clean_texts

def test_clean_texts():
    input_texts = [
        "Hello world.",
        "Test",               # one word
        ".",                  # dot only
        "Another valid text.",
        "Hello world.",       # duplicate
        "   ",                # whitespace only
        "Another valid text.",# duplicate
        "JUST ONE",           # valid
        "..."                 # dot dots
    ]

    expected_output = set([
    "hello world.",
    "another valid text.",
    "just one"
])


    cleaned = clean_texts(input_texts)
    assert set(cleaned) == expected_output
