import tensorflow as tf
import numpy as np
import pickle

pubmed_model_path = 'pubmed_cbow_embeddings.h5'
pubmed_vocab_file = 'pubmed_cbow_vocabulary.txt'
emr_model_path = 'emr_cbow_embeddings.h5'
emr_vocab_file = 'emr_cbow_vocabulary.txt'
glovefile = 'glovedict.pickle'
analogiesfile = 'analogies.csv'
resultsfile = 'analogy-results.txt'

def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print('loading models & vocab')
with open(analogiesfile, 'r', encoding='utf-8') as f:
    analogies = []
    for row in f:
        a, b, c, d, e = row.strip().split(',')
        analogies.append([a, b, c, d, e])

with open(glovefile, 'rb') as f:
    gloveDict = pickle.load(f)

with open(pubmed_vocab_file, 'r', encoding='utf-8') as f:
    pubmedVocab = []
    for word in f:
        pubmedVocab.append(word.strip())

with open(emr_vocab_file, 'r', encoding='utf-8') as f:
    emrVocab = []
    for word in f:
        emrVocab.append(word.strip())

model = tf.keras.models.load_model(pubmed_model_path)
e = model.layers[1]
pubmed_weights = e.get_weights()[0]

model = tf.keras.models.load_model(emr_model_path)
e = model.layers[1]
emr_weights = e.get_weights()[0]

f = open(resultsfile, 'w', encoding='utf-8')
gloveCorrect = 0
pubmedCorrect = 0
emrCorrect = 0

print('checking analogies')
count = 1
step = 20
pubmedCorrectDelta = []
pubmedIncorrectDelta = []
gloveCorrectDelta = []
gloveIncorrectDelta = []
emrCorrectDelta = []
emrIncorrectDelta = []

for analogy in analogies:
    if count % step == 0:
        print(count)
    pubmedVecs = [pubmed_weights[pubmedVocab.index(word)+1] for word in analogy]
    emrVecs = [emr_weights[emrVocab.index(word)+1] for word in analogy]
    gloveVecs = [gloveDict[word] for word in analogy]

    pubmedBase = pubmedVecs[1] - pubmedVecs[0] + pubmedVecs[2]
    emrBase = emrVecs[1] - emrVecs[0] + emrVecs[2]
    gloveBase = gloveVecs[1] - gloveVecs[0] + gloveVecs[2]

    pubmedRightSimilarity = similarity(pubmedBase, pubmedVecs[3])
    pubmedWrongSimilarity = similarity(pubmedBase, pubmedVecs[4])
    gloveRightSimilarity = similarity(gloveBase, gloveVecs[3])
    gloveWrongSimilarity = similarity(gloveBase, gloveVecs[4])
    emrRightSimilarity = similarity(emrBase, emrVecs[3])
    emrWrongSimilarity = similarity(emrBase, emrVecs[4])

    print('\nCorrect analogy: ' + analogy[0] + ' : ' + analogy[1] + ' :: ' + analogy[2] + ' : ' + analogy[3], file=f)
    print('Incorrect analogy: ' + analogy[0] + ' : ' + analogy[1] + ' :: ' + analogy[2] + ' : ' + analogy[4], file=f)
    print('Glove' + (' correct' if gloveRightSimilarity > gloveWrongSimilarity else ' incorrect') + ', right:', gloveRightSimilarity, ' vs. wrong:', gloveWrongSimilarity, file=f)
    print('Pubmed' + (' correct' if pubmedRightSimilarity > pubmedWrongSimilarity else ' incorrect') + ', right:', pubmedRightSimilarity, ' vs. wrong:', pubmedWrongSimilarity, file=f)
    print('EMR' + (' correct' if emrRightSimilarity > emrWrongSimilarity else ' incorrect') + ', right:', emrRightSimilarity, ' vs. wrong:', emrWrongSimilarity, file=f)

    if gloveRightSimilarity > gloveWrongSimilarity:
        gloveCorrect += 1
        gloveCorrectDelta.append(gloveRightSimilarity - gloveWrongSimilarity)
    else:
        gloveIncorrectDelta.append(gloveRightSimilarity - gloveWrongSimilarity)
    if pubmedRightSimilarity > pubmedWrongSimilarity:
        pubmedCorrect += 1
        pubmedCorrectDelta.append(pubmedRightSimilarity - pubmedWrongSimilarity)
    else:
        pubmedIncorrectDelta.append(pubmedRightSimilarity - pubmedWrongSimilarity)
    if emrRightSimilarity > emrWrongSimilarity:
        emrCorrect += 1
        emrCorrectDelta.append(emrRightSimilarity - emrWrongSimilarity)
    else:
        emrIncorrectDelta.append(emrRightSimilarity - emrWrongSimilarity)
    
    count += 1

f.close()
print('Glove accuracy %:', gloveCorrect / len(analogies) * 100.0, ', average deltas:', sum(gloveCorrectDelta)/len(gloveCorrectDelta), sum(gloveIncorrectDelta)/len(gloveIncorrectDelta))
print('Pubmed CBOW accuracy %:', pubmedCorrect / len(analogies) * 100.0, ', average deltas:', sum(pubmedCorrectDelta)/len(pubmedCorrectDelta), sum(pubmedIncorrectDelta)/len(pubmedIncorrectDelta))
print('EMR CBOW accuracy %:', emrCorrect / len(analogies) * 100.0, ', average deltas:', sum(emrCorrectDelta)/len(emrCorrectDelta), sum(emrIncorrectDelta)/len(emrIncorrectDelta))