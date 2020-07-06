import csv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

csv.field_size_limit(1310720)
embedding_dimension = 300
width = 1000
model_info = {
    'EMR CBOW unstructured only': {'model': 'emr_unstructured_lowvaextrinsic.h5', 'vocabulary': 'emr_cbow_vocabulary.txt', 'threshold': 0.5},
    'Pubmed CBOW unstructured only': {'model': 'pubmed_unstructured_lowvaextrinsic.h5', 'vocabulary': 'pubmed_cbow_vocabulary.txt', 'threshold': 0.5},
    'Glove unstructured only': {'model': 'glove_unstructured_lowvaextrinsic.h5', 'vocabulary': 'glove_vocabulary.txt', 'threshold': 0.5},
}
extrinsic_file = 'lowva-extrinsicdata.csv'

print('loading CSV')
with open(extrinsic_file, 'r') as f:
    r = csv.reader(f)
    i = 0
    rawText = []
    outputArray = []
    for row in r:
        i += 1
        if '' in row:
            continue 
        output = np.array([int(row[1])])
        rawText.append(row[2])
        outputArray.append(output)

outputArray = np.array(outputArray)
print('Total Output Array shape:', outputArray.shape)
total_output = tf.data.Dataset.from_tensor_slices(outputArray)

for m in model_info:
    print('\nLoading model:', m)
    print('loading vocabulary')
    with open(model_info[m]['vocabulary'], 'r', errors='surrogateescape') as g:
        vocabulary = []
        for word in g:
            vocabulary.append(word.strip())
        vocabulary_size = len(vocabulary)+2

    tokenizer = tfds.features.text.Tokenizer()
    encoder = tfds.features.text.TokenTextEncoder(vocabulary, tokenizer=tokenizer)
    tokenArray = []
    for note in rawText:
        tokens = encoder.encode(note)
        tokens = tokens[0:width]
        if len(tokens) < width:
            tokens = tokens + [0 for i in range(width-len(tokens))]
        tokenArray.append(tokens)
    
    tokenArray = np.array(tokenArray)
    print('Token Array shape:', tokenArray.shape)

    total_input = tf.data.Dataset.from_tensor_slices(tokenArray)
    total_dataset = tf.data.Dataset.zip((total_input, total_output))
    holdout_dataset = total_dataset.take(300).batch(100)

    print('Setting up Keras model')
    input1 = tf.keras.Input(shape=(width,))
    nl = tf.keras.layers.Embedding(vocabulary_size, embedding_dimension, input_length = width, trainable=False)(input1)
    nl = tf.keras.layers.Dense(512, activation='relu')(nl)
    nl = tf.keras.layers.Dropout(0.50)(nl)

    kernels = [3, 5, 7, 10]
    pooled = []
    for kernel_size in kernels:
        mini_layer = tf.keras.layers.Conv1D(256, kernel_size, activation='relu')(nl)
        mini_pooled = tf.keras.layers.MaxPooling1D(width - kernel_size + 1)(mini_layer)
        pooled.append(mini_pooled)
    nl = tf.keras.layers.Concatenate(axis=1)(pooled)
    nl = tf.keras.layers.Flatten()(nl)
    nl = tf.keras.layers.Dropout(0.50)(nl)

    nl = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(nl)
    nl = tf.keras.layers.Dropout(0.5)(nl)
    nl = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(nl)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(nl)

    model = tf.keras.Model(inputs = input1, outputs = output)

    threshold = model_info[m]['threshold']
    metrics = [tf.keras.metrics.TruePositives(name='tp', thresholds=threshold), 
                tf.keras.metrics.FalsePositives(name='fp', thresholds=threshold), 
                tf.keras.metrics.TrueNegatives(name='tn', thresholds=threshold), 
                tf.keras.metrics.FalseNegatives(name='fn', thresholds=threshold), 
                tf.keras.metrics.Recall(name='sens', thresholds=threshold), 
                tf.keras.metrics.Precision(name='prec', thresholds=threshold), 
                tf.keras.metrics.AUC(name='auroc', curve='ROC'), 
                tf.keras.metrics.AUC(name='auprc', curve='PR'), ]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    model.load_weights(model_info[m]['model'])
    loss, tp, fp, tn, fn, sens, prec, auroc, auprc = model.evaluate(holdout_dataset)
    print('AUROC:', auroc*100)
    print('AUPRC:', auprc*100)
    print('F1:', 2/(1/sens + 1/prec)*100)
    print('Sensitivity/Recall:', sens*100)
    print('Precision:', prec*100)
    print('True Positives:', tp)
    print('False Positives:', fp)
    print('True Negatives:', tn)
    print('False Negatives:', fn)