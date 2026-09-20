# Model Performance Comparison

| Model                  |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   CV F1 (Mean) |   CV F1 (Std) |   False Negatives |   False Positives |
|:-----------------------|-----------:|------------:|---------:|-----------:|----------:|---------------:|--------------:|------------------:|------------------:|
| Logistic Regression    |   0.902174 |    0.896226 | 0.931373 |   0.913462 |  0.936633 |       0.875056 |     0.0295109 |                 7 |                11 |
| Decision Tree          |   0.815217 |    0.84     | 0.823529 |   0.831683 |  0.845289 |       0.846792 |     0.0207447 |                18 |                16 |
| Random Forest          |   0.896739 |    0.88785  | 0.931373 |   0.909091 |  0.933046 |       0.888479 |     0.0358024 |                 7 |                12 |
| Support Vector Machine |   0.880435 |    0.877358 | 0.911765 |   0.894231 |  0.943926 |       0.878342 |     0.0312157 |                 9 |                13 |

**Selected Model for Clinical Deployment**: Logistic Regression
