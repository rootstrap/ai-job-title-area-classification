import re
import simplejson


with open('data_sets/classified_titles.tsv') as f:
    lines = f.readlines()
    lines_without_n = [line.split('\n')[0] for line in lines]
    list_of_lists = [re.split(r'\t+', line) for line in lines_without_n]
    with open('data_sets/classified_titles_list_info.py', 'w') as file:
        file.write('classified_titles_list = ')
        simplejson.dump(list_of_lists, file)
        file.close()
