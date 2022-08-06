"""Code to facilitate the evaluation of this network against AugmentedNet.

Nestor Napoles Lopez, August 6, 2022.
"""
import argparse
from music21.pitch import Pitch
from music21.note import Note
from music21.chord import Chord
from music21.harmony import NoChord
from music21.converter import parse as m21parse
import numpy as np
import pandas as pd

from functional_harmony_recognition import (
    key_dict,
    degree_decoder_dict,
    key_decoder_dict,
    roman_decoder_dict,
    inference_HT,
)

from AugmentedNet.AugmentedNet.cache import forceTonicization
from AugmentedNet.AugmentedNet.cache import getTonicizationScaleDegree
from AugmentedNet.AugmentedNet.chord_vocabulary import frompcset
from AugmentedNet.AugmentedNet.inference import formatRomanNumeral
from AugmentedNet.AugmentedNet.inference import generateRomanText
from AugmentedNet.AugmentedNet.score_parser import parseScore

# Collapsing vocabulary to 38 keys
key_decoder_collapsed = dict(key_decoder_dict)
replacements = [
    ("E#", "F"),
    ("A#", "B-"),
    ("B#", "C"),
    ("c-", "b"),
    ("f-", "e"),
]
for old, new in replacements:
    key_decoder_collapsed[key_dict[old]] = key_decoder_collapsed[key_dict[new]]

quality_to_intervals = {
    "M": ["M3", "P5"],
    "m": ["m3", "P5"],
    "a": ["M3", "A5"],
    "d": ["m3", "d5"],
    "M7": ["M3", "P5", "M7"],
    "m7": ["m3", "P5", "m7"],
    "D7": ["M3", "P5", "m7"],
    "d7": ["m3", "d5", "d7"],
    "h7": ["m3", "d5", "m7"],
    "a6": ["d3", "d5"],
}

inversions = {
    "triad": {0: "", 1: "6", 2: "64",},
    "seventh": {0: "7", 1: "65", 2: "43", 3: "2",},
}

chord_onset_decoder = {1: 0, 0: np.nan}


def padToSequenceLength(arr, sequenceLength, value=0):
    frames, features = arr.shape
    featuresPerSequence = sequenceLength * features
    featuresInExample = frames * features
    padding = featuresPerSequence - (featuresInExample % featuresPerSequence)
    paddingTimesteps = int(padding / features)
    arr = np.pad(arr, ((0, paddingTimesteps), (0, 0)), constant_values=value)
    arr = arr.reshape(-1, sequenceLength, features)
    return arr, paddingTimesteps


def assemble_rn_label(keys, rns_decoded):
    rn_labels = []
    lastKey = ""
    for i in range(len(rns_decoded)):
        if rns_decoded[i] == "pad":
            # If the network predicted padding,
            # repeat the last label
            rn_labels.append(rn_labels[-1])
            continue
        d1, d2, q, inv = rns_decoded[i]
        if keys[i] == "pad":
            key = lastKey
        else:
            key = keys[i]
            lastKey = key
        m21Key = Pitch(f"{key}4")
        intervalTon = degree_decoder_dict[d1]
        m21Tonicization = m21Key.transpose(intervalTon)
        intervalNumerator = degree_decoder_dict[d2]
        m21Numerator = m21Tonicization.transpose(intervalNumerator)
        notes = [m21Numerator] + [
            m21Numerator.transpose(interv)
            for interv in quality_to_intervals[q]
        ]
        notes[int(inv)].octave -= 1
        rn_labels.append(notes)
    return rn_labels


def resolveRomanNumeral(pcset, key, tonicizedKey, inv):
    # if the chord is nondiatonic to the tonicizedKey
    # force a tonicization where the chord does exist
    if tonicizedKey not in frompcset[pcset]:
        # print("Forcing a tonicization")
        candidateKeys = list(frompcset[pcset].keys())
        # prioritize modal mixture
        tonicizedKey = forceTonicization(key, candidateKeys)
    rnfigure = frompcset[pcset][tonicizedKey]["rn"]
    chord = frompcset[pcset][tonicizedKey]["chord"]
    quality = frompcset[pcset][tonicizedKey]["quality"]
    chordtype = "seventh" if len(pcset) == 4 else "triad"
    invfigure = inversions[chordtype][inv]
    if invfigure in ["65", "43", "2"]:
        rnfigure = rnfigure.replace("7", invfigure)
    elif invfigure in ["6", "64"]:
        rnfigure += invfigure
    rn = rnfigure
    if tonicizedKey != key:
        denominator = getTonicizationScaleDegree(key, tonicizedKey)
        rn = f"{rn}/{denominator}"
    chordLabel = f"{chord[0]}{quality}"
    if inv != 0:
        chordLabel += f"/{chord[inv]}"
    return rn, chordLabel


def inference_and_rntxt_annotation(
    inputPath,
    model_checkpoint="model/HT_functional_harmony_recognition_BPS_FH_1.ckpt",
    annealing_slope=3.4522712143931042,
):
    sequence_length = 128
    # df = pd.read_csv(inputPath, sep="\t")
    df = parseScore(inputPath)
    dflen = len(df.index)
    # df["s_notes"] = df["s_notes"].apply(eval)
    pianoroll = np.zeros((len(df.index), 88), dtype=np.int32)
    for idx, row in enumerate(df.itertuples()):
        indexes = [Pitch(x).midi - 21 for x in row.s_notes]
        pianoroll[idx][indexes] = 1
    pianoroll = pianoroll[::2]
    pianoroll, padding = padToSequenceLength(pianoroll, sequence_length)
    prlen = np.array([sequence_length] * pianoroll.shape[0], dtype=np.int32)
    prlen[-1] = sequence_length - padding
    test_data = {"pianoroll": pianoroll, "len": prlen}

    # chord_changes, keys, rns = inference_HT(test_data, model_checkpoint, annealing_slope)
    chord_changes, keys, rns = inference_HT(
        model_checkpoint, test_data=test_data, annealing_slope=annealing_slope
    )
    chordonsets_decoded = [
        chord_onset_decoder[o] for o in chord_changes.reshape(-1)
    ]
    keys_decoded = [key_decoder_collapsed[k] for k in keys.reshape(-1)]
    rns_decoded = [roman_decoder_dict[rn] for rn in rns.reshape(-1)]
    rn_labels = assemble_rn_label(keys_decoded, rns_decoded)
    pcsets = [tuple(sorted([n.pitchClass for n in rn])) for rn in rn_labels]
    inversions = [Chord(rn).inversion() for rn in rn_labels]

    # Chen and Su samples at 16th notes, I sample at 32nd notes
    # making the conversion back to original sampling in dataframe
    chordchanges_32nds = [np.nan] * dflen
    chordchanges_32nds[::2] = chordonsets_decoded[:-padding]
    keys_32nds = [np.nan] * dflen
    keys_32nds[::2] = keys_decoded[:-padding]
    pcsets_32nds = [np.nan] * dflen
    pcsets_32nds[::2] = pcsets[:-padding]
    inversions_32nds = [np.nan] * dflen
    inversions_32nds[::2] = inversions[:-padding]

    # Plug back predictions from Chen and Su in AugmentedNet's dataframe
    df["HarmonicRhythm_ChenSu"] = chordchanges_32nds
    df["LocalKey_ChenSu21"] = keys_32nds
    df["PitchClassSet_ChenSu21"] = pcsets_32nds
    df["Inversion_ChenSu21"] = inversions_32nds
    df.LocalKey_ChenSu21.fillna(method="ffill", inplace=True)
    df.PitchClassSet_ChenSu21.fillna(method="ffill", inplace=True)
    df.Inversion_ChenSu21.fillna(method="ffill", inplace=True)

    chords = df[df.HarmonicRhythm_ChenSu == 0]
    s = m21parse(inputPath)
    # remove all lyrics from score
    for note in s.recurse().notes:
        note.lyrics = []
    prevkey = ""
    for analysis in chords.itertuples():
        notes = []
        for n in s.flat.notes.getElementsByOffset(analysis.Index):
            if isinstance(n, Note):
                notes.append((n, n.pitch.midi))
            elif isinstance(n, Chord) and not isinstance(n, NoChord):
                notes.append((n, n[0].pitch.midi))
        if not notes:
            continue
        bass = sorted(notes, key=lambda n: n[1])[0][0]
        thiskey = analysis.LocalKey_ChenSu21
        tonicizedKey = thiskey  # Always force
        inversion = int(analysis.Inversion_ChenSu21)
        pcset = analysis.PitchClassSet_ChenSu21
        rn2, _ = resolveRomanNumeral(pcset, thiskey, tonicizedKey, inversion)
        if thiskey != prevkey:
            rn2fig = f"{thiskey}:{rn2}"
            prevkey = thiskey
        else:
            rn2fig = rn2
        bass.addLyric(formatRomanNumeral(rn2fig, thiskey))
    rntxt = generateRomanText(s)
    rntxt = rntxt.replace(
        "AugmentedNet v1.9.0 - https://github.com/napulen/AugmentedNet",
        "Chen and Su 2021, translated by napulen - https://github.com/napulen/ChenSu21",
    )
    filename, _ = inputPath.rsplit(".", 1)
    # annotatedScore = f"{filename}_annotated.musicxml"
    # annotationCSV = f"{filename}_annotated.csv"
    annotatedRomanText = f"{filename}.rntxt"
    # s.write(fp=annotatedScore)
    # dfout.to_csv(annotationCSV)
    with open(annotatedRomanText, "w") as fd:
        fd.write(rntxt)
    return rntxt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputPath", help="input score file")
    args = parser.parse_args()
    rntxt = inference_and_rntxt_annotation(args.inputPath)
    print(rntxt)

