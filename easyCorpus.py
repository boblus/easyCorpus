# easyCorpus 1.2

import os
import re
import nltk
import jieba
import logging
import pandas as pd
import jieba.posseg as pseg
from collections import Counter

def corporize(direction):
    corpus = {}
    filenames = []
    for filename in os.listdir(direction):
        if filename[-3:] == 'txt':
            filenames.append(filename)
            filenames.sort()
    for filename in filenames:
        file = open(direction + filename, encoding='utf-8')
        text = file.read()
        corpus[filename] = text
    return corpus

def preprocess(corpus, subee, suber):
    output = {}
    for filename in corpus:
        text = corpus[filename].replace(subee, suber)
        output[text] = text
    return output

def tag(text, lan):
    combi, words, tags = [], [], []
    if lan not in ['zh', 'en']:
        raise ValueError('Language not supported. This function supports Chinese (\'zh\') and English (\'en\').')
    if lan == 'zh':
        jieba.setLogLevel(logging.INFO)
        pos = pseg.cut(text)
        for word, flag in pos:
            combi.append(word+'/'+flag)
            words.append(word)
            tags.append(flag)
    if lan == 'en':
        tokenized_text = nltk.word_tokenize(text)
        pos = nltk.pos_tag(tokenized_text)
        for pair in pos:
            combi.append(pair[0]+'/'+pair[1])
            words.append(pair[0])
            tags.append(pair[1])
    return combi, words, tags

def mean_word_length(combi, lan):
    cnt = 0
    length = 0
    if lan not in ['zh', 'en']:
        raise ValueError('Language not supported. This function supports Chinese (\'zh\') and English (\'en\').')
    if lan == 'zh':
        for pair in combi:
            pos_tag = pair.split('/')[1]
            if pos_tag not in ['x', 'w']:
                cnt = cnt + 1
                length = length + len(pair.split('/')[0])
    if lan == 'en':
        for pair in combi:
            pos_tag = pair.split('/')[1]
            if pos_tag not in ['$', ""''"", '(', ')', ',', '--', '.', ':', '``', 'SYM']:
                cnt = cnt + 1
                length = length + len(pair.split('/')[0])
    return length/cnt

def mean_sent_length(sentences, lan):
    length = 0
    for sent in sentences:
        combi, words, tags = tag(sent, lan)
        length = length + len(words)
    return length/len(sentences)

def punct_count(text, lan):
    combi, words, tags = tag(text, lan)
    count_words = Counter(words)
    count_tags = Counter(tags)
    if lan == 'zh':
        period = count_words['。']
        question = count_words['？']
        exclam = count_words['！']
        comma = count_words['，']
        semi = count_words['；']
        punct = count_tags['x'] + count_tags['w']
    if lan == 'en':
        period = count_words['.']
        question = count_words['?']
        exclam = count_words['!']
        comma = count_words[',']
        semi = count_words[';']
        punct = (count_tags[""''""] + count_tags['('] + count_tags[')']
                 + count_tags[','] + count_tags['--'] + count_tags['.']
                 + count_tags[':'] + count_tags['``'])
    return period, question, exclam, comma, semi, punct

def lex_count(corpus, lan):
    output = []
    if lan == 'zh':
        noun_tags = ['noun', 'n', 'ng', 'nr', 'ns', 'nt', 'nz', 'nrt']
        pronoun_tags = ['pronoun', 'r']
        verb_tags = ['verb', 'v', 'vd', 'vg', 'vn']
        adjective_tags = ['adjective', 'a', 'ad', 'ag', 'an']
        adverb_tags = ['adverb', 'd', 'df', 'dg']
        conjunction_tags = ['conjunction', 'c']
        auxiliary_tags = ['auxiliary', 'u', 'ud', 'ug', 'uj', 'ul', 'uv', 'uz']
        content_tags = (['content'] + noun_tags + verb_tags + adjective_tags + adverb_tags + 
                        ['b', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'q',
                         'r', 's', 't', 'z', 'mq', 'tg', 'zg'])
        function_tags = ['function', 'c', 'e', 'o', 'p', 'u', 'y', 'ud', 'ug', 'uj', 'ul', 'uv', 'uz']
    if lan == 'en':
        noun_tags = ['noun', 'NN', 'NNP', 'NNPS', 'NNS']
        pronoun_tags = ['pronoun', 'PRP', 'PRP$', 'WP', 'WP$']
        verb_tags = ['verb', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        adjective_tags = ['adjective', 'J', 'JJR', 'JJS']
        adverb_tags = ['adverb', 'RB', 'RBR', 'RBS', 'WRB']
        conjunction_tags = ['conjunction', 'CC']
        auxiliary_tags = ['auxiliary', 'MD']
        content_tags = ['content'] + noun_tags + verb_tags + adjective_tags + adverb_tags + pronoun_tags + ['CD', 'UH']
        function_tags = ['function'] + conjunction_tags + auxiliary_tags + ['DT', 'EX', 'IN', 'PDT', 'RP', 'TO', 'WDT']
    for filename in corpus:
        combi, words, tags = tag(corpus[filename], lan)
        count = Counter(tags)
        data = {}
        for labels in [noun_tags, pronoun_tags, verb_tags, adjective_tags, adverb_tags,
                       conjunction_tags, auxiliary_tags, content_tags, function_tags]:
            data.setdefault(labels[0], 0)
            for item in labels[1:]:
                data[labels[0]] += count[item]
        period, question, exclam, comma, semi, punct = punct_count(corpus[filename], lan)
        word_count = len(tags) - punct
        output.append([filename, len(words), len(set(words)), len(set(words))/len(words), word_count,
                       mean_word_length(combi, lan), data['content']/word_count, data['function']/word_count,
                       data['noun']/word_count, data['pronoun']/word_count, data['verb']/word_count,
                       data['adjective']/word_count, data['adverb']/word_count, data['conjunction']/word_count,
                       data['auxiliary']/word_count])
    return pd.DataFrame(output, columns=['docname', 'tokens', 'types', 'TTR', 'words',
                                         'MWL', 'content', 'function',
                                         'noun', 'pronoun', 'verb',
                                         'adjective', 'adverb', 'conjunction',
                                         'auxiliary'])

def sent_count(corpus, lan):
    output = []
    for filename in corpus:
        sentences = sent_segment(corpus[filename], lan)
        period, question, exclam, comma, semi, punct = punct_count(corpus[filename], lan)
        output.append([filename, len(sentences), period/len(sentences), question/len(sentences),
                       exclam/len(sentences), mean_sent_length(sentences, lan), punct, period/punct,
                       question/punct, exclam/punct, comma/punct, semi/punct])
    return pd.DataFrame(output, columns=['docname', 'sentences', 'statement', 'interrogative',
                                         'exclamatory', 'MSL', 'punctuation', 'period',
                                         'question', 'exclamation', 'comma', 'semicolon'])

def sent_segment(text, lan):
    if lan not in ['zh', 'en']:
        raise ValueError('Language not supported. This function supports Chinese (\'zh\') and English (\'en\').')
    if lan == 'zh':
        text = re.sub('([。！？\...... \?])([^”’。！？\......])', r'\1\n\2', text)
        text = re.sub('([。！？\...... \?])([”’])', r'\1\2\n', text)
        return text.split('\n')
    if lan == 'en':
        sentences = nltk.sent_tokenize(text)
        return sentences

def pre(tokenized_text, indice, window):
    output = []
    for i in indice:
        avant = []
        if i == 0:
            avant.append('')
        elif i < window:
            for x in range(0, i):
                avant.append(tokenized_text[x])
        else:
            for x in range(i-window, i):
                avant.append(tokenized_text[x])
        output.append(avant)
    return output

def post(tokenized_text, indice, window):
    output = []
    for i in indice:
        apres = []
        if i == len(tokenized_text):
            apres.append('')
        elif i+window >= len(tokenized_text):
            for y in range(i+1, len(tokenized_text)):
                apres.append(tokenized_text[y])
        else:
            for y in range(i+1, i+window+1):
                apres.append(tokenized_text[y])
        output.append(apres)
    return output

def kwic(corpus, keyword, lan, window=4, mode=None, pos=False):
    if lan not in ['zh', 'en']:
        raise ValueError('Language not supported. This function supports Chinese (\'zh\') and English (\'en\').')
    if mode not in ['re', None]:
        raise ValueError('\'mode\' can only be set as \'re\' or None.')
    if lan == 'zh':
        jieba.setLogLevel(logging.INFO)
        keywords = jieba.lcut(keyword)
    if lan == 'en':
        keywords = keyword.split()
    output = []
    for filename in corpus:
        combi, words, tags = tag(corpus[filename], lan)
        if mode == 're' or len(keywords) == 1:
            matches = re.findall(keyword, ' '.join(words))
            for match in set(matches):
                indice = [index for (index, value) in enumerate(words) if value == match]
                if pos == True:
                    avant = pre(combi, indice, window)
                    apres = post(combi, indice, window)
                    word = combi
                else:
                    avant = pre(words, indice, window)
                    apres = post(words, indice, window)
                    word = words
                for i in range(len(avant)):
                    output.append([filename, indice[i], indice[i], ' '.join(avant[i]),
                                   word[indice[i]], ' '.join(apres[i])])
        elif len(keywords) > 1:
            indice = [index for (index, value) in enumerate(words) if value == keywords[-1]]
            avant_words = pre(words, indice, window+len(keywords)-1)
            for i in range(len(avant_words)):
                if ' '.join(avant_words[i][-len(keywords)+1:]) == ' '.join(keywords[:len(keywords)-1]):
                    if pos == True:
                        avant = pre(combi, indice, window+len(keywords)-1)
                        apres = post(combi, indice, window)
                        word = combi
                    else:
                        avant = avant_words
                        apres = post(words, indice, window)
                        word = words
                    output.append([filename, indice[i]-len(keywords)+1, indice[i],
                                   ' '.join(avant[i][:len(avant[i])-len(keywords)+1]),
                                   ' '.join(word[indice[i]-len(keywords)+1:indice[i]+1]),
                                   ' '.join(apres[i])])
    if output == []:
        print('Input not found.')
    return pd.DataFrame(output, columns=['docname', 'from', 'to', 'pre', 'keyword', 'post'])
