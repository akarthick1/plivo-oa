# Run Log

**Run 1: Silence Baseline**
*   **Score:** 1600 ms response delay @ 0.0% interruptions (AUC = 0.514).
*   **Changes:** Ran the provided `baseline.py`.
*   **Reasoning:** Established the baseline performance of a naive silence-timer.

**Run 2: Logistic Regression (Causal Prosody Features)**
*   **Score:** 1402 ms response delay @ 5.0% interruptions (AUC = 0.606).
*   **Changes:** Implemented `extract_features` focusing strictly on the 1.5s window before `pause_start`. Extracted total duration, energy mean, energy slope, pitch mean, and pitch slope. Trained a balanced Logistic Regression model.
*   **Reasoning:** A falling pitch and dropping energy usually signal the end of a thought. The slopes mathematically capture this trailing off. 

**Run 3: Random Forest Classifier**
*   **Score:** 521 ms response delay @ 5.0% interruptions (AUC = 0.970).
*   **Changes:** Upgraded the model in `train.py` from Logistic Regression to a Random Forest Classifier (`n_estimators=100`, `max_depth=5`). 
*   **Reasoning:** Linear models struggle with the complex interactions of prosody (e.g., flat pitch but steep energy drop). The Random Forest easily captured these non-linear thresholds, drastically improving confidence and lowering response delay.

**Run 4: Multilingual Training**
*   **Score:** (Generalized across both English and Hindi test sets).
*   **Changes:** Modified `train.py` to ingest and concatenate both `data/english` and `data/hindi` datasets simultaneously with unique turn IDs.
*   **Reasoning:** To prevent overfitting to English-specific prosody, forcing the model to learn universal, cross-lingual rules for turn-taking.
