import numpy, re


def encode_onehot(index, size):
    onehot = numpy.zeros(size, dtype=numpy.int8)
    onehot[index] = 1
    return onehot


def prettify_string(string, replace_newline_with_space=True):
    pretty_string = str(string).strip()

    if replace_newline_with_space:
        pretty_string = pretty_string.replace('\n', ' ').replace('\r', ' ')

    pretty_string = re.sub(' +', ' ', pretty_string)

    return pretty_string


def remove_size_descriptions(text):
    tailored_text = ''

    for word in text.lower().split(' '):
        if is_alphanumeric(word):
            continue
        elif ',' in word:
            valid_word = True
            for word_part in word.split(','):
                if is_alphanumeric(word_part):
                    valid_word = False
            if valid_word is False:
                continue
        elif '/' in word:
            valid_word = True
            for word_part in word.split('/'):
                if is_alphanumeric(word_part):
                    valid_word = False
            if valid_word is False:
                continue
        elif '~' in word:
            valid_word = True
            for word_part in word.split('~'):
                if is_alphanumeric(word_part):
                    valid_word = False
            if valid_word is False:
                continue

        tailored_text += word
        tailored_text += ' '

    return tailored_text.strip()


def is_alphabet(string):
    is_alphabet = True
    for char in string:
        if not ord('A') <= ord(char) <= ord('z'):
            is_alphabet = False
            break

    return is_alphabet


def is_numeric(string):
    is_numeric = True
    for char in string:
        if not ord('0') <= ord(char) <= ord('9'):
            is_numeric = False
            break

    return is_numeric


def is_alphanumeric(string):
    is_alphanumeric = True
    for char in string:
        if ord('A') <= ord(char) <= ord('z'):
            continue
        elif ord('0') <= ord(char) <= ord('9'):
            continue
        is_alphanumeric = False
        break

    return is_alphanumeric