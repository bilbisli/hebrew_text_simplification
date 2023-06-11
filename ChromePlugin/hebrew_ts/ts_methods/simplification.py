import re
from string import punctuation
from functools import lru_cache

import wordfreq
from tqdm.auto import tqdm

# Cache the result of the function to improve performance
@lru_cache(maxsize=50)
def get_common_words(top_n=27000, lang='he'):
    return wordfreq.top_n_list(lang, top_n)

# Cache the result of the function to improve performance
@lru_cache(maxsize=None)
def get_word_freq(word, lang='he'):
    return wordfreq.word_frequency(word, lang)

# Cache the result of the function to improve performance
@lru_cache(maxsize=50)
def get_top_n_frequency(top_n=27000, lang='he'):
    top_common_words = get_common_words(top_n, lang)
    last_common_word = top_common_words[-1]
    top_n_freq = get_word_freq(last_common_word, lang)
    return top_n_freq

# Check if a given text is a number
def is_number(text):
    return re.match('[+-]?(\d+)?\.?\d+', text) is not None

# Split the text into words and create a mask for named entities
def word_split_ner_mask(text, tokenizer, ner_model, mask_model):
    tokenized_text = tokenizer(text, return_offsets_mapping=True)
    ners = ner_model(text)

    ner_indexes = [enr_ent['index'] for enr_ent in ners]
    ner_mask = [index in ner_indexes for index in range(len(tokenized_text['input_ids']))]

    added_tokens, true_word_split, offset_ner_mask = [], [], []
    for idx, word_id in enumerate(tokenized_text.word_ids()):
        if word_id is not None:
            start, end = tokenized_text.word_to_tokens(word_id)
            if start == end - 1:
                token_range = (start,)
            else:
                token_range = (start, end-1)
            if len(added_tokens) == 0 or added_tokens[-1] != token_range:
                decoded_word = tokenizer.decode(tokenized_text['input_ids'][token_range[0]:token_range[-1]+1], skip_special_tokens=True)
                added_tokens.append(token_range)
                true_word_split.append(decoded_word)
                offset_ner_mask.append(ner_mask[idx])

    return true_word_split, offset_ner_mask

# Mask a word in the text and replace it with a suitable candidate
def mask_and_replace(text_list, index, model, tokenizer, mask_token='[MASK]', score_threshold=0.23, new_line_model_token='<NL>', is_neighbour=False):
    new_text_list = text_list[:]
    new_text_list[index] = mask_token

    sen = tokenizer.convert_tokens_to_string(list(filter(lambda w: w != new_line_model_token, new_text_list)))
    candidates = model(sen)
    new_text_list[index] = text_list[index]

    if candidates[0]['score'] >= score_threshold \
    and (is_neighbour 
         or is_not_word(candidates[0]['token_str']) 
         or get_word_freq(candidates[0]['token_str']) > get_word_freq(text_list[index])):
        new_text_list[index] = candidates[0]['token_str']

    return new_text_list

# Check if a word is not a valid word, such as a number or punctuation
def is_not_word(word, new_line_model_token='<NL>'):
    return is_number(word) or word in list(punctuation) or word == new_line_model_token

# Find the indices of words to mask in the text
def find_mask_index(text_list, mask_exclusion, check_frequency=True, new_line_model_token='<NL>'):
    most_common_word_freq = get_top_n_frequency()
    mask_indices = []
    for i, word in enumerate(text_list):
        if mask_exclusion[i] \
                or is_not_word(word, new_line_model_token=new_line_model_token) \
                or (check_frequency and get_word_freq(word) >= most_common_word_freq):
            mask_indices.append(False)
        else:
            mask_indices.append(True)

    return mask_indices

# Remove spaces between decimal numbers in the text
def unspace_decimal_numbers(text):
    return re.sub(r"(\d+) *(\.) *(\d+)", r"\1\2\3", text)

# Remove unnecessary spaces around quotes in the text
def unspace_quotes(text):
    # Remove spaces before and after single quotes when they are used as quotes ('word')
    text = re.sub(r"'\s*(.*?)\s*'", r"'\1'", text)
    
    # Remove spaces before and after double quotes when they are used as quotes ("word")
    text = re.sub(r'"\s*(.*?)\s*"', r'"\1"', text)

    # Remove spaces surrounding double or single quotes
    text = text.replace(" ' ", "'").replace(' " ', '"')
    
    return text

# Simplify words in the text by masking and replacing them
def simplify_words(text, index_list=None, tokenizer=None, ner_model=None, mask_model=None, score_threshold=0.32, check_frequency=True, neighbours_threshold=0.7, new_line_model_token='<NL>'):
    if index_list is None:
        if not all((tokenizer, ner_model, mask_model)):
            raise ValueError('If index list is not provided, a tokenizer, ner model, and mask model need to be provided')
        text_list, ner_mask = word_split_ner_mask(text, tokenizer, ner_model, mask_model)
        index_list = find_mask_index(text_list, ner_mask, check_frequency=check_frequency, new_line_model_token=new_line_model_token)
    words_bar = tqdm(enumerate(index_list), desc='word masking', leave=False, total=len(index_list), disable=len(index_list)==0)

    for i, mask_word in words_bar:
        if mask_word:
            prev_word = text_list[i]
            text_list = mask_and_replace(text_list, i, model=mask_model, tokenizer=tokenizer, score_threshold=score_threshold)

            if prev_word != text_list[i] and i + 1 < len(index_list) and not ner_mask[i + 1] and not index_list[i + 1] and not is_not_word(text_list[i + 1]):
                text_list = mask_and_replace(text_list, i + 1, model=mask_model, tokenizer=tokenizer, score_threshold=neighbours_threshold, is_neighbour=True)

    tokens_to_text = tokenizer.convert_tokens_to_string(text_list)
    fixed_text = unspace_quotes(unspace_decimal_numbers(tokens_to_text))
    return fixed_text
