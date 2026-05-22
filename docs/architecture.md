# Architecture Notes

## System goal

One text review enters the system and the model predicts:

1. sentiment
2. authenticity

The backend returns both class labels and probabilities to the web interface.

## ML architecture

Initial model design:

- shared encoder: `XLM-RoBERTa`
- pooled text representation from the encoder
- task head 1: sentiment classification
- task head 2: authenticity classification

This is a classic hard-parameter-sharing multitask setup. It is strong enough for a dissertation MVP and gives us a clean comparison against single-task baselines.

## Why this design

- one encoder reduces inference cost compared to two separate models
- both tasks depend on linguistic signals in the same review text
- multitask learning may improve representation quality when labels are limited for authenticity detection

## Planned experiment ladder

1. `TF-IDF + LogisticRegression` baselines for each task
2. single-task Transformer for sentiment
3. single-task Transformer for authenticity
4. joint multitask Transformer
5. optional extension: metadata or behavioral features for authenticity

## Dissertation-ready comparison

Core comparison should answer:

- does multitask learning improve authenticity detection?
- does authenticity supervision improve sentiment robustness?
- how much quality is gained relative to training cost?

