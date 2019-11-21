import simplejson
from sentence_normalizer import normalize_sentence
from data_sets.classified_titles_list_info import classified_titles_list
from data_sets.values_and_labels_dicts import area_label_value_dict


normalized_sentences = []
classified_sentences = []

for li in classified_titles_list:
    normalized_sentences.append(normalize_sentence(li[0]))
    classified_sentences.append(area_label_value_dict[li[1]])

with open('data_sets/normalized_and_classified_sentences.py', 'w') as file:
    file.write('normalized_sentences = ')
    simplejson.dump(normalized_sentences, file)
    file.write('\n')
    file.write('classified_sentences = ')
    simplejson.dump(classified_sentences, file)
    file.close()

