def normalize_sentence(sentence):
    # remove ' and .
    sentence = sentence.replace("'", "")
    sentence = sentence.replace(".", "")
    # change , for space
    sentence = sentence.replace(", ", " ")
    sentence = sentence.replace(",", " ")
    sentence = sentence.replace("/", " ")
    sentence = sentence.lower()
    sentence = sentence.partition("&")[0]
    sentence = sentence.partition(" and ")[0]
    return sentence
