# easyCorpus

*easyCorpus* is a python module to process corpus data and perform statistical analysis.

# environment
--python 3.7  
--os  
--re  
--nltk  
--jieba  
--numpy  
--pandas  
--logging  
--matplotlib  
--collections  

# main functions

**1. import corpus data (plain text) from a local environment**

* `corporize(direction)`

|  |  |  |
|----|----|----|
|**parameters:**|**direction:**|*str*|
| | |the direction in which the text files (.txt) are stored|
|**return:**|**corpus**|*dict*|

*Example:*
```python
from easyCorpus import corporize

corporize('Corpus/')
```
*Output:*
```
{
    '2016.txt': '公元2016年，公历闰年，共366天，53周。今年大二，要上的课还挺多的。', 
    '2017.txt': '公元2017年，公历平年，共365天，53周。去国外交换！到处玩玩！想去什么国家？', 
    '2018.txt': '公元2018年，公历平年，共365天，53周。眼看就要大四了。还有什么没上的课？', 
    '2019.txt': '公元2019年，公历平年，共365天，52周零1天。毕业了！把行李拿回家是个问题。邮寄？托运？', 
    '2020.txt': '公元2020年，公历闰年，共366天，52周零2天。新的一年有什么期待？'
}
```

**2. segment and tag text**

* `tag(text, lan)`

|  |  |  |
|----|----|----|
|**parameters:**|**text:**|*str*|
| | |the text to be tagged|
| |**lan:**|*{'zh', 'en}*|
| | |language of the text|
|**return:**|**combi, word, tags**|*list*|

*Example:*
```python
from easyCorpus import tag

combi, words, tags = tag(corpus['2020.txt'], lan='zh')

print(combi)

print(words)

print(tags)
```
*Output:*
```
['公元/q', '2020/m', '年/m', '，/x', '公历/n', '闰年/t', '，/x', '共/n', '366/m', '天/n', '，/x', '52/m', '周/nr', '零/n', '2/m', '天/n', '。/x', '新/a', '的/uj', '一年/m', '有/v', '什么/r', '期待/v', '？/x']

['公元', '2020', '年', '，', '公历', '闰年', '，', '共', '366', '天', '，', '52', '周', '零', '2', '天', '。', '新', '的', '一年', '有', '什么', '期待', '？']

['q', 'm', 'm', 'x', 'n', 't', 'x', 'n', 'm', 'n', 'x', 'm', 'nr', 'n', 'm', 'n', 'x', 'a', 'uj', 'm', 'v', 'r', 'v', 'x']
```

**3. statistical analysis in word level**

* `lex_count(corpus, lan)`

|  |  |  |
|----|----|----|
|**parameters:**|**corpus:**|*dict*|
| | |the corpus to be analyzed|
| |**lan:**|*{'zh', 'en'}*|
| | |language of the text in the corpus|
|**return:**|**result**|*pandas.DataFrame*|

*Example:*
```python
from easyCorpus import lex_count

lex_count(corpus, lan='zh')
```
*Output:*
|docname|tokens|types|TTR|words|MWL|content|function|noun|pronoun|verb|adjective|adverb|conjunction|auxiliary|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|2016.txt|26|21|0.807692|20|1.550000|0.900000|0.100000|0.250000|0.000000|0.050000|0.000000|0.10|0.0|0.100000
|2017.txt|27|22|0.814815|20|1.700000|1.000000|0.000000|0.350000|0.050000|0.200000|0.000000|0.05|0.0|0.000000
|2018.txt|26|22|0.846154|20|1.700000|0.900000|0.100000|0.300000|0.050000|0.150000|0.000000|0.05|0.0|0.100000
|2019.txt|33|27|0.818182|25|1.560000|0.920000|0.080000|0.480000|0.000000|0.120000|0.000000|0.00|0.0|0.040000
|2020.txt|24|21|0.875000|19|1.631579|0.947368|0.052632|0.315789|0.052632|0.105263|0.052632|0.00|0.0|0.052632

**4. statistical analysis in sentence level**

* `sent_count(corpus, lan)`

|  |  |  |
|----|----|----|
|**parameters:**|**corpus:**|*dict*|
| | |the corpus to be analyzed|
| |**lan:**|*{'zh', 'en'}*|
| | |language of the text in the corpus|
|**return:**|**result**|*pandas.DataFrame*|

*Example:*
```python
from easyCorpus import sent_count

sent_count(corpus, lan='zh')
```
*Output:*
|docname|sentences|statement|interrogative|exclamatory|MSL|punctuation|period|question|exclamation|comma|semicolon|
|--|--|--|--|--|--|--|--|--|--|--|--|
|2016.txt|2|1.000000|0.000000|0.0|13.000000|6|0.333333|0.000000|0.000000|0.666667|0.0|
|2017.txt|4|0.250000|0.250000|0.5|6.750000|7|0.142857|0.142857|0.285714|0.428571|0.0|
|2018.txt|3|0.666667|0.333333|0.0|8.666667|6|0.333333|0.166667|0.000000|0.500000|0.0|
|2019.txt|5|0.400000|0.400000|0.2|6.600000|8|0.250000|0.250000|0.125000|0.375000|0.0|
|2020.txt|2|0.500000|0.500000|0.0|12.000000|5|0.200000|0.200000|0.000000|0.600000|0.0|

**5. key word in context**

* `kwic(corpus, keyword, lan, window=4, mode=None, pos=False)`

|  |  |  |
|----|----|----|
|**parameters:**|**corpus:**|*dict*|
| | |the corpus to be searched|
| |**keyword:**|*str*|
| | |the word to be searched for|
| |**lan:**|*{'zh', 'en'}*|
| | |language of the text in the corpus|
| |**window:**|*int, default 4*|
| | |the window of the context|
| |**mode:**|*{'re', None}, default None*|
| | |choice of searching method (use regular expression or not)|
| |**pos:**|*bool, default False*|
| | |if True, show the part of speech tags of words|
|**return:**|**result**|*pandas.DataFrame*|

*Example 1: default search*
```python
from easyCorpus import kwic

corpus = corporize('历年政府工作报告/')
kwic(corpus, '伟大的事业', lan='zh', mode=None, pos=False)
```
*Output:*
|docname|from|to|pre|keyword|post|
|--|--|--|--|--|--|
|1955政府工作报告_李富春.txt|32865|32867|过 的 极其 光荣|伟大 的 事业|。 ” 我们 每|
|1986政府工作报告_赵紫阳.txt|5809|5811|经济 建设 服务 的|伟大 的 事业|中 ， 为 人民|

*Example 2: search using regular expression*
```python
kwic(corpus, '[\u4e00-\u9fa5]*工', lan='zh', mode='re', pos=False)[:5]
```
*Output:*
|docname|from|to|pre|keyword|post|
|--|--|--|--|--|--|
|1954政府工作报告_周恩来.txt|4803|4803|但是 中级 形式 的|加工|、 订货 、 包销|
|1954政府工作报告_周恩来.txt|4860|4860|中 ， 接受 国家|加工|、 订货 、 包销|
|1954政府工作报告_周恩来.txt|1467|1467|已 全部 或者 部分|完工|并 投入 生产 的|
|1954政府工作报告_周恩来.txt|2752|2752|虽然 注意 了 一般|技工|的 培养 ， 但是|
|1954政府工作报告_周恩来.txt|2763|2763|注意 技术 人才 和|高级技工|的 培养 ， 对于|

*Example 3: search using regular expression, return a smaller window of context and part of speech tags*
```python
kwic(corpus, '[\u4e00-\u9fa5]*工', window=2, lan='zh', mode='re', pos=True)[:5]
```
*Output:*
|docname|from|to|pre|keyword|post|
|--|--|--|--|--|--|
|1954政府工作报告_周恩来.txt|4982|4982|形式/n 的/uj|加工/vn|、/x 订货/n|
|1954政府工作报告_周恩来.txt|5039|5039|接受/v 国家/n|加工/vn|、/x 订货/n|
|1954政府工作报告_周恩来.txt|1558|1558|或者/c 部分/n|完工/v|并/c 投入/v|
|1954政府工作报告_周恩来.txt|2867|2867|了/ul 一般/a|技工/n|的/uj 培养/v|
|1954政府工作报告_周恩来.txt|2878|2878|人才/n 和/c|高级技工/n|的/uj 培养/v|

**6. word distribution plot**

* `word_distribution_plot(corpus, keyword, lan, tile, fig_width, fig_height)`

|  |  |  |
|----|----|----|
|**parameters:**|**corpus:**|*dict*|
| | |the corpus to be analyzed|
| |**keyword:**|*str*|
| | |the word to be searched for|
| |**lan:**|*{'zh', 'en'}*|
| | |language of the text in the corpus|
| |**tile:**|*{1, 2, 5, 10}*|
| | |the number of tiles to be displayed |
| |**fig_width:**|*int*|
| | |the width of the output figure|
| |**fig_height:**|*int*|
| | |the height of the output figure|
|**return:**|**figure**|*module*|

*Example:*
```python
from easyCorpus import word_distribution, word_distribution_plot

corpus = corporize('二马/')
word_distribution_plot(corpus, 'en', 'China', tile=5, fig_width=16, fig_height=8)
```
*Output:*
![word frequency.png](https://i.loli.net/2020/05/14/UtCAmhDayZxqodw.png)

