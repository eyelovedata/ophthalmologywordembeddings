import collections
import csv
import tensorflow_datasets as tfds

csvfile = 'patientrandomnoteexam.csv'
inputfile = 'emr_cbow_inputs.txt'
labelfile = 'emr_cbow_labels.txt'
vocabularyfile = 'emr_cbow_vocabulary.txt'
stepsize = 1000
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

z = open(csvfile, 'r', encoding='utf-8')
c = csv.reader(z)

count = 0
entries = 0
for row in c:
    if count == 0:
        count += 1
        continue
    else:
        count = count + 1
        if count % stepsize == 0:
            print('Note:', count)
        notetext = row[2].strip().replace('\n',' ').lower()
        words = tokenizer.tokenize(notetext)
        words = [word for word in words if word not in stopwords]
        for i in range(len(words)):
            indices = [j for j in range(i-cbow_window,i+cbow_window+1) if j != i]
            line = ' '.join([words[j] if (j >= 0 and j < len(words)) else edgeword for j in indices])
            f.write(line + '\n')
            h.write(words[i] + '\n')
            entries = entries + 1

        vocab.update(words)
print("Total notes:", count)
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