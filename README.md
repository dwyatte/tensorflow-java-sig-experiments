# tensorflow-java-sig-experiments

Experimenting with the new TensorFlow [Java SIG API](https://github.com/tensorflow/java).

## Usage

`python/` contains a Python script `create_models.py` that exports a few different toy TensorFlow
models.

`java/` contains a "Hello World" program (`src/main/java/HelloTensorFlow.java`), a barebones
inference program (`src/main/java/SavedModelPredictor.java`), and a more general inference
program that inspects the MetaGraph (`src/main/java/SavedModelPredictorInspectMetagraph.java`).
