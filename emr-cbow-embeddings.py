import tensorflow as tf
import tensorflow_datasets as tfds

embedding_dimension = 300

inputfile = 'emr_cbow_inputs.txt'
labelfile = 'emr_cbow_labels.txt'
vocabularyfile = 'emr_cbow_vocabulary.txt'

with open(vocabularyfile, 'r', encoding='utf-8') as g:
    vocabulary = []
    for word in g:
        vocabulary.append(word.strip())
    vocabulary_size = len(vocabulary)+2

input_dataset = tf.data.TextLineDataset(inputfile)
label_dataset = tf.data.TextLineDataset(labelfile)
labeled_dataset = tf.data.Dataset.zip((input_dataset, label_dataset))

tokenizer = tfds.features.text.Tokenizer()
encoder = tfds.features.text.TokenTextEncoder(vocabulary, tokenizer=tokenizer)

def encode(text_tensor, label_tensor):
    return encoder.encode(text_tensor.numpy()), encoder.encode(label_tensor.numpy())[0]

def encode_map_fn(text, label):
    return tf.py_function(encode, [text, label], (tf.int64, tf.int64))

encoded_dataset = labeled_dataset.map(encode_map_fn)
test_data = encoded_dataset.take(20000).padded_batch(1000, padded_shapes=([-1],[]))
validation_data = encoded_dataset.skip(20000).take(20000).padded_batch(1000, padded_shapes=([-1],[]))
train_data = encoded_dataset.skip(40000).padded_batch(1000, padded_shapes=([-1],[]))

input = tf.keras.Input(shape=(10,))
nl = tf.keras.layers.Embedding(vocabulary_size, embedding_dimension)(input)
nl = tf.keras.layers.GlobalAveragePooling1D()(nl)
output = tf.keras.layers.Dense(vocabulary_size, activation='softmax')(nl)
model = tf.keras.Model(inputs = input, outputs = output)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  
model.fit(train_data, epochs=10, validation_data=validation_data, callbacks=[tf.keras.callbacks.EarlyStopping(patience=1, verbose=1, restore_best_weights=True)])
model.evaluate(test_data)

model.save('emr_cbow_embeddings.h5')