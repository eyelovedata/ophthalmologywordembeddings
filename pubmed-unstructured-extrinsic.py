import csv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

csv.field_size_limit(1310720)
embedding_dimension = 300
width = 1000
class_weights = [1, 1]
model_path = 'pubmed_cbow_embeddings.h5'
vocabularyfile = 'pubmed_cbow_vocabulary.txt'

def weighted_binary_crossentropy(y_true, y_pred, from_logits=False):
    y_true = tf.keras.backend.cast(y_true, y_pred.dtype)
    obce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    weights = class_weights[0]*y_true+class_weights[1]
    return tf.keras.backend.mean(obce*weights, axis=-1)

print('loading vocabulary')
with open(vocabularyfile, 'r', encoding='utf-8') as g:
    vocabulary = []
    for word in g:
        vocabulary.append(word.strip())
    vocabulary_size = len(vocabulary)+2

tokenizer = tfds.features.text.Tokenizer()
encoder = tfds.features.text.TokenTextEncoder(vocabulary, tokenizer=tokenizer)

print('loading CSV')
filename = 'lowva-extrinsicdata.csv'
with open(filename, 'r') as f:
    r = csv.reader(f)
    i = 0
    tokenArray = []
    structuredArray = []
    outputArray = []
    for row in r:
        i += 1
        if i == 1:
            continue
        if '' in row:
            continue 
        output = np.array([int(row[1])])
        
        tokens = encoder.encode(row[2])
        tokens = tokens[0:width]
        if len(tokens) < width:
            tokens = tokens + [0 for i in range(width-len(tokens))]
        
        outputArray.append(output)
        tokenArray.append(tokens)

tokenArray = np.array(tokenArray)
outputArray = np.array(outputArray)
print(tokenArray.shape, outputArray.shape)

print('loading Pubmed CBOW vectors')
model = tf.keras.models.load_model(model_path)
e = model.layers[1]
embedding_matrix = e.get_weights()[0]
print(embedding_matrix.shape)
del model

total_input = tf.data.Dataset.from_tensor_slices(tokenArray)
total_output = tf.data.Dataset.from_tensor_slices(outputArray)
total_dataset = tf.data.Dataset.zip((total_input, total_output))
train_dataset = total_dataset.skip(900).shuffle(1000).batch(16)
validation_dataset = total_dataset.skip(600).take(300).batch(100)
test_dataset = total_dataset.skip(300).take(300).batch(100)

input1 = tf.keras.Input(shape=(width,))
nl = tf.keras.layers.Embedding(vocabulary_size, embedding_dimension, embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix), input_length = width, trainable=False)(input1)
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
model.summary()

metrics = [tf.keras.metrics.TruePositives(name='tp'), tf.keras.metrics.FalsePositives(name='fp'), tf.keras.metrics.TrueNegatives(name='tn'), tf.keras.metrics.FalseNegatives(name='fn'), tf.keras.metrics.Recall(name='sens'), tf.keras.metrics.Precision(name='prec'), tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.BinaryAccuracy(name='acc')]
model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=metrics)
model.fit(train_dataset, epochs=40, validation_data=validation_dataset, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, min_delta=0.0001, verbose=1, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, verbose=1)])
model.evaluate(test_dataset)
model.save_weights('pubmed_unstructured_lowvaextrinsic.h5')