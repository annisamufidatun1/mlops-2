import os
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from kerastuner import HyperParameters, RandomSearch
from kerastuner.tuners import RandomSearch
from keras_tuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs
# from tfx.components.tuner.component import TunerFnResult
from stroke_pred_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

NUM_EPOCHS = 5

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )
 
    return dataset

def build_model(hp):
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(tf.keras.Input(shape=(dim,), name=transformed_name(key)))
    
    for feature in NUMERICAL_FEATURES:
        input_features.append(tf.keras.Input(shape=(1,), name=transformed_name(feature)))
    
    concatenate = tf.keras.layers.concatenate(input_features)
    x = tf.keras.layers.Dense(
        units=hp.Choice("units_1", values=[128, 256, 512]),
        activation="relu"
    )(concatenate)
    x = tf.keras.layers.Dense(
        units=hp.Choice("units_2", values=[32, 64, 128]),
        activation="relu"
    )(x)
    x = tf.keras.layers.Dense(
        units=hp.Choice("units_3", values=[8, 16, 32]),
        activation="relu"
    )(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    return model

def tuner_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    tuner = RandomSearch(
        hypermodel=build_model,
        objective='val_binary_accuracy',
        max_trials=2,
        hyperparameters=HyperParameters(),
        directory=fn_args.working_dir,
        project_name='stroke_pred_tuning'
    )

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, NUM_EPOCHS)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, NUM_EPOCHS)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )

# def get_serve_tf_examples_fn(model, tf_transform_output):
#     model.tft_layer = tf_transform_output.transform_features_layer()
 
#     @tf.function
#     def serve_tf_examples_fn(serialized_tf_examples):
#         feature_spec = tf_transform_output.raw_feature_spec()
#         feature_spec.pop(LABEL_KEY)
#         parsed_features = tf.io.parse_example(
#             serialized_tf_examples, feature_spec
#         )
#         transformed_features = model.tft_layer(parsed_features)
#         outputs = model(transformed_features)
#         return {"outputs": outputs}
 
#     return serve_tf_examples_fn


