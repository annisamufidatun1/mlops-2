"""Training module
"""

import os

import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from kerastuner import HyperParameters
from kerastuner.tuners import RandomSearch
from stroke_pred_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)
from stroke_pred_tuner import input_fn

NUM_EPOCHS = 5

def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature"""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn

def get_model(hyperparameters, show_summary = True):
    """
    This function defines a Keras model and returns the model as a
    Keras object.
    """

    # one-hot categorical features
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim,), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )
    concatenate = tf.keras.layers.concatenate(input_features)
    deep = tf.keras.layers.Dense(hyperparameters["units_1"], activation="relu")(concatenate)
    deep = tf.keras.layers.Dense(hyperparameters["units_2"], activation="relu")(deep)
    deep = tf.keras.layers.Dense(hyperparameters["units_3"], activation="relu")(deep)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    
    learning_rate = hyperparameters["learning_rate"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    if show_summary:
        model.summary()

    return model

# def gzip_reader_fn(filenames):
#     """Loads compressed data"""
#     return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
    """Train the model based on given args.
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    hyperparameters = fn_args.hyperparameters["values"]

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")

    # tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, NUM_EPOCHS)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, NUM_EPOCHS)


    # log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    
    model = get_model(hyperparameters)
    
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10
    )

    signatures = {
        'serving_default': get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        ),
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )

    plot_model(
        model,
        to_file='images/model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )