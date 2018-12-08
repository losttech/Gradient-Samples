import tensorflow as tf

train, test = tf.keras.datasets.fashion_mnist.load_data()
trainImages, trainLabels = train
testImages, testLabels = test

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(trainImages, trainLabels, epochs=5)

eval_result = model.evaluate(testImages, testLabels)

loss, acc = eval_result

print([loss, acc])
