def normalize_sentence(sentence):
    # remove ' and .
    sentence = sentence.replace("'", "")
    sentence = sentence.replace(".", "")
    # change , for space
    sentence = sentence.replace(", ", " ")
    sentence = sentence.replace(",", " ")
    sentence = sentence.replace("/", " ")
    sentence = sentence.lower()
    return sentence
