# -*- coding: utf-8 -*-

import os
import re
import nltk
import jieba
import logging
import matplotlib
import numpy as np
import pandas as pd
import jieba.posseg as pseg
import matplotlib.pyplot as plt
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
        output[filename] = text
    return output

def tag(text, lan):
    combi, tokens, tags = [], [], []
    if lan not in ['zh', 'en']:
        raise ValueError('Language not supported. This function supports Chinese (\'zh\') and English (\'en\').')
    if lan == 'zh':
        jieba.setLogLevel(logging.INFO)
        pos = pseg.cut(text)
        for word, flag in pos:
            combi.append(word+'/'+flag)
            tokens.append(word)
            tags.append(flag)
    if lan == 'en':
        tokenized_text = nltk.word_tokenize(text)
        pos = nltk.pos_tag(tokenized_text)
        for pair in pos:
            combi.append(pair[0]+'/'+pair[1])
            tokens.append(pair[0])
            tags.append(pair[1])
    return combi, tokens, tags

def STTR(words):
    chunk = []
    start = 0
    for end in range(1000, len(words), 1000):
        chunk.append((start, end))
        start = start + 1000
    chunk.append((start, len(words)))
    TTRs = []
    for pair in chunk:
        if pair[1] - pair[0] != 1000:
            TTRs.append(0)
        else:
            TTRs.append(len(set(words[pair[0]:pair[1]]))/len(words[pair[0]:pair[1]]))
    return np.mean(TTRs)

def mean_word_length(text, lan):
    cnt = 0
    length = 0
    combi, tokens, tags = tag(text, lan)
    if lan not in ['zh', 'en']:
        raise ValueError('Language not supported. This function supports Chinese (\'zh\') and English (\'en\').')
    if lan == 'zh':
        for token in tokens:
            if token not in ['。', '？', '！', '，', '；', '：', '“', '”', '‘', '’',
                             '（', '）', '「', '」', '【', '】', '《', '》', '、',
                             '/', '\\', '-', '——', '……']:
                cnt = cnt + 1
                length = length + len(token)
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
        combi, tokens, tags = tag(sent, lan)
        length = length + len(tokens)
    return length/len(sentences)

def word_count(text, lan):
    output = {}
    combi, tokens, tags = tag(text, lan)
    count = Counter(tags)
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
        adjective_tags = ['adjective', 'JJ', 'JJR', 'JJS']
        adverb_tags = ['adverb', 'RB', 'RBR', 'RBS', 'WRB']
        conjunction_tags = ['conjunction', 'CC']
        auxiliary_tags = ['auxiliary', 'MD']
        content_tags = ['content'] + noun_tags + verb_tags + adjective_tags + adverb_tags + pronoun_tags + ['CD', 'UH']
        function_tags = ['function'] + conjunction_tags + auxiliary_tags + ['DT', 'EX', 'IN', 'PDT', 'RP', 'TO', 'WDT']
    for labels in [noun_tags, pronoun_tags, verb_tags, adjective_tags, adverb_tags,
                   conjunction_tags, auxiliary_tags, content_tags, function_tags]:
        for item in labels[1:]:
            output[labels[0]] = output.get(labels[0], 0) + count[item]
    words = []
    for item in combi:
         if item.split('/')[1] in content_tags + function_tags:
                words.append(item.split('/')[0])        
    output['words'] = words
    return output

def punct_count(text, lan):
    combi, tokens, tags = tag(text, lan)
    count_tokens = Counter(tokens)
    count_tags = Counter(tags)
    if lan == 'zh':
        period = count_tokens['。']
        question = count_tokens['？']
        exclam = count_tokens['！']
        comma = count_tokens['，']
        semi = count_tokens['；']
        total_punct = period + question + exclam + comma + semi
        + count_tokens['：'] + count_tokens['“'] + count_tokens['”']
        + count_tokens['‘'] + count_tokens['’'] + count_tokens['（']
        + count_tokens['）'] + count_tokens['「'] + count_tokens['」']
        + count_tokens['【'] + count_tokens['】'] + count_tokens['《']
        + count_tokens['》'] + count_tokens['、'] + count_tokens['/'] 
        + count_tokens['\\'] + count_tokens['-'] + count_tokens['——']
        + count_tokens['……']        
    if lan == 'en':
        period = count_tokens['.']
        question = count_tokens['?']
        exclam = count_tokens['!']
        comma = count_tokens[',']
        semi = count_tokens[';']
        total_punct = (count_tags[""''""] + count_tags['('] + count_tags[')']
                       + count_tags[','] + count_tags['--'] + count_tags['.']
                       + count_tags[':'] + count_tags['``'])
    return period, question, exclam, comma, semi, total_punct

def lex_count(corpus, lan):
    output = []   
    for filename in corpus:
        combi, tokens, tags = tag(corpus[filename], lan)
        word_data = word_count(corpus[filename], lan)
        period, question, exclam, comma, semi, total_punct = punct_count(corpus[filename], lan)
        total_word = len(word_data['words'])
        output.append([filename, total_word, len(set(word_data['words'])), len(set(word_data['words']))/total_word,
                       STTR(word_data['words']), mean_word_length(corpus[filename], lan), word_data['content']/total_word,
                       word_data['function']/total_word, word_data['noun']/total_word, word_data['pronoun']/total_word,
                       word_data['verb']/total_word, word_data['adjective']/total_word, word_data['adverb']/total_word,
                       word_data['conjunction']/total_word, word_data['auxiliary']/total_word])
    return pd.DataFrame(output, columns=['docname', 'tokens', 'types', 'TTR', 'STTR',
                                         'MWL', 'content', 'function',
                                         'noun', 'pronoun', 'verb',
                                         'adjective', 'adverb', 'conjunction',
                                         'auxiliary'])

def sent_count(corpus, lan):
    output = []
    for filename in corpus:
        sentences = sent_segment(corpus[filename], lan)
        period, question, exclam, comma, semi, total_punct = punct_count(corpus[filename], lan)
        output.append([filename, len(sentences), period/len(sentences), question/len(sentences),
                       exclam/len(sentences), mean_sent_length(sentences, lan), total_punct, period/total_punct,
                       question/total_punct, exclam/total_punct, comma/total_punct, semi/total_punct])
    return pd.DataFrame(output, columns=['docname', 'sentences', 'statement', 'interrogative',
                                         'exclamatory', 'MSL', 'punctuation', 'period',
                                         'question', 'exclamation', 'comma', 'semicolon'])

def seg_count(corpus, lan, start, end):
    if lan != 'zh':
        raise ValueError('Language not supported. This function supports Chinese (\'zh\').')
    else:
        temp = []
        record = {}
        for i in range(start, end):
            temp.append(i)
        for filename in corpus:
            for i in temp:
                record.setdefault(filename, {})
                record[filename].setdefault(i, 0)
            text = re.sub('([，；：。！？])([^”’。！？\…… ）》】」])', r'\1\n\2', corpus[filename])
            text = re.sub('([，；：。！？])([”’ ）》】」])', r'\1\2\n', text)
            for item in text.split('\n'):
                combi, tokens, tags = tag(item, lan)
                period, question, exclam, comma, semi, total_punct = punct_count(item, lan)
                total_word = len(tokens) - total_punct
                try:
                    record[filename][total_word] += 1
                except Exception as e:
                    None
        output = pd.DataFrame()
        output['length'] = temp
        for filename in record:
            temp = []
            for item in record[filename]:
                temp.append(record[filename][item])
            output[filename] = temp
        return output

def sent_segment(text, lan):
    if lan not in ['zh', 'en']:
        raise ValueError('Language not supported. This function supports Chinese (\'zh\') and English (\'en\').')
    if lan == 'zh':
        text = re.sub('([。！？\……])([^”’。！？\…… ）》】」])', r'\1\n\2', text)
        text = re.sub('([。！？\……])([”’ ）》】」])', r'\1\2\n', text)
        output = text.split('\n')
        if '' in output:
            output.remove('')
        return output
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
        combi, tokens, tags = tag(corpus[filename], lan)
        if mode == 're' or len(keywords) == 1:
            matches = re.findall(keyword, ' '.join(tokens))
            for match in set(matches):
                indice = [index for (index, value) in enumerate(tokens) if value == match]
                if pos == True:
                    avant = pre(combi, indice, window)
                    apres = post(combi, indice, window)
                    word = combi
                else:
                    avant = pre(tokens, indice, window)
                    apres = post(tokens, indice, window)
                    word = tokens
                for i in range(len(avant)):
                    output.append([filename, indice[i], indice[i], ' '.join(avant[i]),
                                   word[indice[i]], ' '.join(apres[i])])
        elif len(keywords) > 1:
            indice = [index for (index, value) in enumerate(tokens) if value == keywords[-1]]
            avant_words = pre(tokens, indice, window+len(keywords)-1)
            for i in range(len(avant_words)):
                if ' '.join(avant_words[i][-len(keywords)+1:]) == ' '.join(keywords[:len(keywords)-1]):
                    if pos == True:
                        avant = pre(combi, indice, window+len(keywords)-1)
                        apres = post(combi, indice, window)
                        word = combi
                    else:
                        avant = avant_words
                        apres = post(tokens, indice, window)
                        word = tokens
                    output.append([filename, indice[i]-len(keywords)+1, indice[i],
                                   ' '.join(avant[i][:len(avant[i])-len(keywords)+1]),
                                   ' '.join(word[indice[i]-len(keywords)+1:indice[i]+1]),
                                   ' '.join(apres[i])])
    if output == []:
        print('Input not found.')
    return pd.DataFrame(output, columns=['docname', 'from', 'to', 'pre', 'keyword', 'post'])

def load_stopwords(file_direction):
    file = open(file_direction, encoding='utf-8')
    text = file.readlines()
    return text

def word_frequency(tokens, start, end):
    count = {}
    for token in tokens:
        if (token+'\n') not in stopwords:
            count[token] = count.get(token, 0) + 1
    output = []
    for pair in sorted(count.items(), key=lambda count:count[1], reverse=True)[start:end]:
        output.append([pair[0], pair[1]])
    return pd.DataFrame(output, columns=['word', 'frequency'])

def word_distribution(tokens, keyword, tile):
    if tile not in [1,2,5,10]:
        raise ValueError('The value of tile should be in [1, 2, 5, 10].')
    start= 0
    tiles, times = [], []
    for i in range(0, 10, int(10/tile)):
        end = int(len(tokens) * (i+10/tile) / 10 + 0.5)
        total_token = Counter(tokens[start:end])
        tiles.append(str(10*(i+int(10/tile)))+'%')
        times.append(total_token[keyword])
        start = end
    return tiles, times

def word_distribution_plot(corpus, keyword, lan, tile, fig_width, fig_height):
    
    plt.rcParams['figure.figsize'] = (fig_width, fig_height)
    
    if lan == 'zh':
        plt.rcParams['font.sans-serif']=['SimHei']
        y_label = '词频'
        title = '\'%s\'的分布' % keyword
    if lan == 'en':
        plt.rcParams['font.sans-serif']=['DejaVu Sans']
        y_label = 'Word frequency'
        title = 'The distribution of \'%s\'' % keyword

    cnt = 0
    temp = locals()

    x = np.arange(tile)
    fig, ax = plt.subplots()
    for filename in corpus:
        freq = word_distribution(tag(corpus[filename], lan)[1], keyword, tile)
        temp['rects%s' % cnt] = ax.bar(x-0.85+(cnt+0.5)*0.7/len(corpus), freq[1], width=0.7/len(corpus), label=filename)
        for rect in temp['rects%s' % cnt]:
            height = rect.get_height()
            ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        cnt = cnt + 1

    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(freq[0], fontsize=14)
    ax.legend(fontsize=14)

    plt.savefig('word frequency.png')
    plt.show()

def highlight(df, keyword, color):
    
    def highlight_val(val):
        chrome = color if keyword in str(val) else 'black'
        return 'color: %s' % chrome

    return df.style.applymap(highlight_val)    

class alignment:
    
    def analyze(df, lan):
        st_col = input('Please type in the name of the source text column:')
        tt_col = input('Please type in the name of the target text column:')
        output = []
        for i in range(len(df)):
            hic = sent_segment(str(df[tt_col][i]).rstrip(' '), lan)
            if i == 0 or i == len(df) - 1:
                index = i+1 if i == 0 else i-1
                ref = sent_segment(str(df[tt_col][index]).rstrip(' '), lan)
                if len(hic) > 1:
                    if hic[0] not in ref and hic[-1] not in ref:
                        output.append(['one to many', df[st_col][i], df[tt_col][i]])
                    else:
                        output.append(['many to many', df[st_col][i], df[tt_col][i]])
                else:
                    if str(df[tt_col][i]).rstrip(' ') not in ref:
                        output.append(['one to one', df[st_col][i], df[tt_col][i]])
                    elif df[tt_col][i] == df[tt_col][index]:
                        output.append(['many to one', df[st_col][i], df[tt_col][i]])
                    else:
                        output.append(['many to many', df[st_col][i], df[tt_col][i]])
            else:
                pre = sent_segment(str(df[tt_col][i-1]).rstrip(' '), lan)
                post = sent_segment(str(df[tt_col][i+1]).rstrip(' '), lan)
                if len(hic) > 1:
                    if hic[0] not in pre and hic[-1] not in post:
                        output.append(['one to many', df[st_col][i], df[tt_col][i]])
                    else:
                        output.append(['many to many', df[st_col][i], df[tt_col][i]])
                else:
                    if str(df[tt_col][i]).rstrip(' ') not in pre and str(df[tt_col][i]).rstrip(' ') not in post:
                        output.append(['one to one', df[st_col][i], df[tt_col][i]])
                    elif df[tt_col][i] == df[tt_col][i-1] or df[tt_col][i] == df[tt_col][i+1]:
                        output.append(['many to one', df[st_col][i], df[tt_col][i]])
                    else:
                        output.append(['many to many', df[st_col][i], df[tt_col][i]])
        return pd.DataFrame(output, columns=['type', 'source text', 'target text'])
    
    def summary(df, lan):
        output = []
        analysis = alignment.analyze(df, lan)
        output.append([len(df),
                       len(analysis[analysis['type']=='one to one'])/len(df),
                       len(analysis[analysis['type']=='one to many'])/len(df),
                       len(analysis[analysis['type']=='many to one'])/len(df),
                       len(analysis[analysis['type']=='many to many'])/len(df)])
        return pd.DataFrame(output, columns=['alignments', 'one to one', 'one to many', 'many to one', 'many to many'])
