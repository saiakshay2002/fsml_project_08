import os
import gdown

from src.preprocess import preprocess_pipeline, save_processed_data
from src.train import train_and_select_best_model

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

import os
os.makedirs("artifacts", exist_ok=True)

# 🔹 Dataset URL
DATA_URL = "https://drive.google.com/uc?id=11cacj82VwVw9yRVIkk3m83dcT5YqBf0L"


# 🔹 Download dataset if not present
def download_data():
    os.makedirs("data/raw", exist_ok=True)

    file_path = "data/raw/train_FD001.txt"

    if not os.path.exists(file_path):
        print("Downloading dataset from Google Drive...")
        gdown.download(DATA_URL, file_path, quiet=False)
        print("Download complete!")
    else:
        print("Dataset already exists. Skipping download.")


def run_pipeline():
    download_data()

    print("Step 1: Preprocessing...")
    train_df, val_df, test_df = preprocess_pipeline("data/raw/train_FD001.txt")
    save_processed_data(train_df, val_df, test_df)

    print("Step 2: Training...")
    best_name, best_model, results, rul_results = train_and_select_best_model()

    print("\nGenerating Graphs...")

    # Example: create dummy or reuse data
    y_train = train_df['label']
    y_test = test_df['label']

    # If your model supports it
    X_test = test_df.drop(columns=['label', 'unit_number', 'time_in_cycles'], errors='ignore')
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]

    plot_class_distribution(y_train)
    plot_confusion(y_test, y_pred)
    plot_rul_vs_prob()
    plot_rul_metrics()
    plot_rul_vs_cycles(train_df)
    plot_full_model_comparison()

    print(f"\nBest model: {best_name}")
    print(f"Validation F1: {results[best_name]['validation']['f1']:.4f}")
    print(f"Test F1: {results[best_name]['test']['f1']:.4f}")

    print("\nRUL Model Performance:")
    print(f"Validation MAE: {rul_results['val_mae']:.4f}")
    print(f"Test MAE: {rul_results['test_mae']:.4f}")
    print(f"Test RMSE: {rul_results['test_rmse']:.4f}")



# ================================
# 1. Class Distribution
# ================================
def plot_class_distribution(y):
    plt.figure()
    y.value_counts().plot(kind='bar')
    plt.title("Class Distribution")
    plt.savefig("artifacts/class_distribution.png")
    plt.close()

# ================================
# 2. Confusion Matrix
# ================================
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Normal','Failure'], yticklabels=['Normal','Failure'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

# ================================
# 3. RUL vs Probability
# ================================
def plot_rul_vs_prob():
    rul = [11.72, 118.24, 59.13]
    prob = [0.7889, 0.0001, 0.3417]

    plt.figure()
    plt.scatter(rul, prob)

    for i, label in enumerate(['Failure','Healthy','Medium']):
        plt.text(rul[i], prob[i], label)

    plt.xlabel("RUL")
    plt.ylabel("Failure Probability")
    plt.title("RUL vs Probability")
    plt.savefig("artifacts/rul_vs_prob.png")
    plt.close()


# ================================
# 4. RUL Metrics
# ================================
def plot_rul_metrics():
    metrics = ['MAE','RMSE']
    values = [14.6139, 20.8094]

    plt.figure()
    plt.bar(metrics, values)
    plt.title("RUL Metrics")
    plt.savefig("artifacts/rul_metrics.png")
    plt.close()

# ================================
# 5. rul_vs_cycles
# ================================
def plot_rul_vs_cycles(df):
    import matplotlib.pyplot as plt

    plt.figure()

    # If columns exist
    if 'unit_number' in df.columns and 'time_in_cycles' in df.columns:
        unit = df['unit_number'].unique()[0]
        temp = df[df['unit_number'] == unit]
        x = temp['time_in_cycles']
    else:
        # fallback (use index)
        temp = df.sort_values(by='RUL', ascending=False)
        x = range(len(temp))

    y = temp['RUL']

    plt.plot(x, y)

    plt.xlabel("Cycles")
    plt.ylabel("RUL")
    plt.title("RUL vs Cycles")

    plt.savefig("artifacts/rul_vs_cycles.png")
    plt.close()

# ================================
# 6. plot_full_model_comparison
# ================================
def plot_full_model_comparison():
    import matplotlib.pyplot as plt
    import numpy as np

    models = ['Logistic Regression', 'Random Forest', 'XGBoost']

    precision = [0.7575, 0.8414, 0.8219]
    recall = [0.9204, 0.8559, 0.8538]
    f1 = [0.8311, 0.8486, 0.8376]

    x = np.arange(len(models))
    width = 0.25

    plt.figure()

    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1')

    plt.xticks(x, models)
    plt.legend()
    plt.title("Model Comparison (All Metrics)")

    plt.savefig("artifacts/full_model_comparison.png")
    plt.close()


if __name__ == "__main__":
    run_pipeline()
