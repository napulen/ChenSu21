# How to train (by napulen)

- Download the `BPS` dataset, which is located in the `functional-harmony` repo (older version of HT)
- Build `Dockerfile` with `docker build . -t chensu21`
- Run the image with `docker run -it chensu21`
- Inside the image, you may need to install `apt python3-tk`, but it seems I don't need to anymore
- Run `python BPS_FH_preprocessing.py` to generate the `.pickle` file
- Run `python functional_harmony_recognition.py` to train the model

# Harmony-Transformer-v2

An improved version of the [Harmony Transformer](https://github.com/Tsung-Ping/Harmony-Transformer). We evaluated the new model in terms of automatic chord recognition for symbolic music. For more details, please refer to ["Attend to Chords: Improving Harmonic Analysis of Symbolic Music Using Transformer-Based Models" (TISMIR 2021)](https://transactions.ismir.net/articles/10.5334/tismir.65/).

## File Descriptions
 * `BPS_FH_preprocessing.py`: preprocessing of the [BPS-FH dataset](https://github.com/Tsung-Ping/functional-harmony)
 * `chord_recognition_models.py`: implementations of the three chord recognition models: the Harmony Transformer (HT/HTv2), the Bi-directional Transformer for Chord Recognition (BTC), and the convolutional recurrent neural network (CRNN)
 * `chord_symbol_recognition.py`: train the chord recognition models using the 24 maj-min chord representations
 * `functional_harmony_recognition.py`: train the chord recognition models using the chord representations of Roman numeral analysis

## Requirements
 * python >= 3.6.4
 * tensorflow >= 1.8.0
 * numpy >= 1.16.2
 * xlrd >= 1.1.0
 * scipy >= 1.5.4

