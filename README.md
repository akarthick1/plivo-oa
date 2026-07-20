# End-of-Turn (EOT) Detection for Voice AI

## Overview
Every conversational Voice AI agent needs to know when a human is done speaking and not just pausing to think. If the agent jumps in too early, it interrupts the user. If it waits too long, the conversation feels laggy and awkward. 

This project solves that by building a machine learning model from scratch to predict the probability that a turn has ended using causal audio features.

## How the Model Works
Standard Voice Activity Detection usually just looks for silence, which forces the system to wait around 1.6 seconds to be sure the person isn't just taking a breath. 

Instead of just measuring silence, this model looks at how the person stopped speaking. It analyzes the 1.5 seconds of audio right before the pause to check the pitch and energy slopes. Because human speech naturally trails off at the end of a sentence, the model can detect this drop and confidently predict that the turn is over. We used a Random Forest Classifier trained on both English and Hindi datasets so it learns universal speech patterns rather than just memorizing one language.

## Results
The goal was to lower the response delay as much as possible while keeping the interruption rate under 5%. 

The basic silence-timer baseline had a response delay of 1600 ms. Our custom model managed to cut that down to about 521 ms. This means the Voice AI responds a full second faster without cutting people off.

## Project Files
- train.py: Extracts the audio features, trains the model, and saves it.
- predict.py: Loads the saved model and makes predictions on new data.
- score.py: Simulates a live voice agent to test the model's delay and interruption rate.
- features.py: Contains the audio processing math to get pitch and energy.

## How to Run

First, install the needed libraries by running:
`pip install -r requirements.txt`

To train the model on the provided data, run:
`python train.py --data_dirs data/english data/hindi`

To generate predictions for the English dataset, run:
`python predict.py --data_dir data/english --out outputs/english_predictions.csv`

To test the performance and see your final score, run:
`python score.py --data_dir data/english --pred outputs/english_predictions.csv`
