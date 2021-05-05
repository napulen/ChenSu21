# Harmony-Transformer-v2

An improved version of the [Harmony Transformer](https://github.com/Tsung-Ping/Harmony-Transformer). We evaluated the new model in terms of automatic chord recognition for symbolic music. For more details, please refer to ["Attend to Chords: Improving Harmonic Analysis of Symbolic Music Using Transformer-Based Models" (TISMIR 2021)](https://transactions.ismir.net/articles/10.5334/tismir.65/).

## File descriptions
 * `BPS_FH_preprocessing.py`: preprocessing of the [BPS-FH dataset](https://github.com/Tsung-Ping/functional-harmony)
 * `chord_recognition_models.py`: implementations of the three models in comparison: the Harmony Transformer (HT/HTv2), the Bi-directional Transformer for Chord Recognition (BTC), and a convolutional recurrent neural network (CRNN)
 * `chord_symbol_recognition.py`: chord recognition using 24 maj-min chord vocabulary 
 * `functional_harmony_recognition.py`:  chord recognition using vocabulary of Roman numeral (RN) analysis

## Requirements
 * python >= 3.6.4
 * tensorflow >= 1.8.0
 * numpy >= 1.16.2
 * xlrd >= 1.1.0
 * scipy >= 1.5.4

