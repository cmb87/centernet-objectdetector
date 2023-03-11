import logging
import os
import pathlib

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


# Freezes a modelWithoutPosthead to pb
class ModelFreezer:

    # =======================================
    @staticmethod
    def loadFrozen(
        pbfile,
        inputs=["x:0"],
        outputs=["Identity:0"],
        print_graph=False,
    ):

        logging.info(f"Loading frozen model: {pbfile}")

        # Load file
        with tf.io.gfile.GFile(pbfile, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        # No idea what this is
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        # Turn into concrete function
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])

        return wrapped_import.prune(
            tf.nest.map_structure(wrapped_import.graph.as_graph_element, inputs),
            tf.nest.map_structure(wrapped_import.graph.as_graph_element, outputs),
        )

    # =======================================
    @staticmethod
    def convert2PbModel(model, outputdir="./", showgraph=True):

        # Convert Keras modelWithoutPosthead to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )

        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        
        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(
            graph_or_graph_def=frozen_func.graph,
            logdir=outputdir,
            name="frozen.pb",
            as_text=False,
        )

        layers = [op.name for op in frozen_func.graph.get_operations()]

        logging.info("-" * 50)
        logging.info("Frozen modelWithoutPosthead layers: ")

        for layerName in layers:
            logging.info(f"layer: {layerName}")

        logging.info("-" * 50)
        logging.info("Frozen modelWithoutPosthead inputs: ")
        logging.info(frozen_func.inputs)
        logging.info("Frozen modelWithoutPosthead outputs: ")

    # =======================================
    @staticmethod
    def convert2tflite(model, outputdir):

        # Get input shapes of image
        ih = model.inputs[0].shape[1]
        iw = model.inputs[0].shape[2]
        ic = model.inputs[0].shape[3]

        # Convert Keras modelWithoutPosthead to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )

        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        layers = [op.name for op in frozen_func.graph.get_operations()]
        for n, l in enumerate(layers):
            print(f"Layer {n}: {l}")
        # convert concrete function into TF Lite modelWithoutPosthead using TFLiteConverter
        converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func])

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.allow_custom_ops = True

        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]  # tf.lite.OpsSet.TFLITE_BUILTINS_INT8,

        converter.experimental_new_converter = True

        tflite_model = converter.convert()

        # save modelWithoutPosthead
        tflite_model_files = pathlib.Path(os.path.join(outputdir, "model.tflite"))
        tflite_model_files.write_bytes(tflite_model)



if __name__ == "__main__":
    ModelFreezer.convert2tflite(model, "./frozen")
    ModelFreezer.convert2PbModel(model, "./frozen")