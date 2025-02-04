import matplotlib.pyplot as plt # type: ignore
import seaborn as sns           # type: ignore
import pandas as pd             # type: ignore

def plot_feature_importance(feature_importance):
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance.keys(), feature_importance.values())
    plt.xticks(rotation=45)
    plt.title('Feature Importance in Predicting Student Churn')
    plt.tight_layout()
    plt.savefig('../images/feature_importance.png')

def plot_metrics(metrics):
    metrics_df = pd.DataFrame(metrics).transpose()
    
    plt.figure(figsize=(10, 6))
    metrics_df.plot(kind='bar')
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('../images/model_metrics.png')