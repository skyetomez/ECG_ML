import tensorflow as tf


class residual_layer(tf.keras.layers.Layer):
    """custom residual layer based on paper, not trainable"""

    def __init__(
        self, units=32, kernel=5, pool_size=5, stride=2, name="resid", **kwargs
    ) -> None:

        super(residual_layer, self).__init__(name=name)
        self.conv1d = tf.keras.layers.Conv1D(
            filters=units, kernel_size=kernel, strides=1, padding="same"
        )
        self.conv1d.trainable = False  # Freeze the layer
        self.residual = tf.keras.layers.Add()
        self.residual.trainable = False  # Freeze the layer
        self.pooling = tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=stride)
        self.pooling.trainable = False  # Freeze the layer
        self.relu = tf.keras.layers.ReLU()
        self.relu.trainable = False  # Freeze the layer
        super(residual_layer, self).__init__(**kwargs)
        return None

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.relu(x)
        x = self.conv1d(x)
        x = self.residual([inputs, x])
        x = self.relu(x)
        return self.pooling(x)

    def get_config(self):
        config = super(residual_layer, self).get_config()
        config.update(
            {
                "conv1d": self.conv1d,
                "relu": self.relu,
                "conv1d": self.conv1d,
                "residual": self.residual,
                "relu": self.relu,
                "pool": self.pooling,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ECGModel(tf.keras.Model):
    """ECG model base with only final connected layers trainable"""

    def __init__(self, units=32, kernel=5, name="ECGModel") -> None:
        super(ECGModel, self).__init__(name=name)
        self.residual_block = residual_layer()
        self.fc_block = tf.keras.layers.Dense(units)
        self.relu = tf.keras.layers.ReLU()
        self.conv_block = tf.keras.layers.Conv1D(
            filters=units, kernel_size=kernel, padding="same"
        )
        self.conv_block.trainable = False  # Freeze the layer
        self.final_block = tf.keras.layers.Dense(5)

    def call(self, inpu):
        x = self.conv_block(inpu)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.fc_block(x)
        x = self.relu(x)
        x = self.fc_block(x)
        x = self.final_block(x)
        return tf.nn.softmax(x)
