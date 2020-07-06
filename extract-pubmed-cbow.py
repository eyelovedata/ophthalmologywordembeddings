import sqlite3, collections
import tensorflow_datasets as tfds

dbfile = 'pubmed.db'
inputfile = 'pubmed_cbow_inputs.txt'
labelfile = 'pubmed_cbow_labels.txt'
vocabularyfile = 'pubmed_cbow_vocabulary.txt'
stepsize = 10000
smallestfreq = 5
cbow_window = 5
edgeword = 'ENDTOKEN'
stopwords = [
    'a', 'all', 'also', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from', 'had', 'has', 'have', 
    'in', 'is', 'it', 'may', 'of', 'on', 'or', 'our', 'than', 'that', 'the', 'there', 'these', 'this', 'to', 
    'was', 'we', 'were', 'which', 'who', 'with'
]

vocab = collections.Counter()
tokenizer = tfds.features.text.Tokenizer()

f = open(inputfile, 'w', encoding='utf-8')
g = open(vocabularyfile, 'w', encoding='utf-8')
h = open(labelfile, 'w', encoding='utf-8')

conn = sqlite3.connect(dbfile)
c = conn.cursor()
c.execute('SELECT abstract FROM Articles ORDER BY RANDOM()')

count = 0
entries = 0
for row in c:
    count = count + 1
    if count % stepsize == 0:
        print('Article:', count)
    abstract = row[0].strip().replace('\n',' ').lower()
    words = tokenizer.tokenize(abstract)
    words = [word for word in words if word not in stopwords]
    for i in range(len(words)):
        indices = [j for j in range(i-cbow_window,i+cbow_window+1) if j != i]
        line = ' '.join([words[j] if (j >= 0 and j < len(words)) else edgeword for j in indices])
        f.write(line + '\n')
        h.write(words[i] + '\n')
        entries = entries + 1

    vocab.update(words)
print("Total articles:", count)
g.write('ENDTOKEN\n')

print('top 10 words')
for word, num in vocab.most_common(10):
    print(word, num)
print('total entries:', entries)

for word, num in vocab.most_common():
    if num < smallestfreq:
        break
    g.write(word + '\n')

f.close()
g.close()
h.close()