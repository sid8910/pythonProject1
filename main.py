# Load the TensorBoard notebook extension
import tensorboard
import tensorflow as tf
import datetime

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255

print(x_train.shape)
print(x_test.shape)

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.98):
            print('\n Reached 98% training accuracy so cancelling training!')
            self.model.stop_training = True


callback = myCallback()

model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(
    x=x_train,
    y=y_train,
    epochs=50,
    validation_data=(x_test, y_test),
    callbacks=[callback]
)