# Industrial Equipment Failure Prediction for Predictive Maintenance

## Dataset
NASA CMAPSS FD001 dataset  
Link: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Download the FD001 dataset and place it in:

data/raw/train_FD001.txt

---

## Preprocessing
- RUL computation
- Binary failure labeling (threshold = 30)
- Low-variance feature removal
- Engine-wise data split (no leakage)

---

## How to Run Preprocessing

```python
from src.preprocess import preprocess_pipeline

train_df, val_df, test_df = preprocess_pipeline("data/raw/train_FD001.txt")

## Model Training
### Models Used
  Logistic Regression
  Random Forest
  Gradient Boosting
### Run Training
  cd ./src
  python train.py
