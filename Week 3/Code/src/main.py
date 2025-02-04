from data_preprocessing import load_data, preprocess_data, prepare_features
from model_training import ChurnPredictor
from visualization import plot_feature_importance, plot_metrics

def main():
    # Load and preprocess data
    data = load_data('Raw Data.csv')
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = prepare_features(processed_data)
    
    # Train and evaluate models
    predictor = ChurnPredictor()
    predictor.train_models(X_train, y_train)
    metrics = predictor.evaluate_models(X_test, y_test)
    
    # Get feature importance
    feature_names = ['Age', 'Gender', 'Country', 'Current/Intended Major', 
                    'Days_Until_Start', 'Application_Processing_Time']
    importance = predictor.get_feature_importance(feature_names)
    
    # Create visualizations
    plot_feature_importance(importance)
    plot_metrics(metrics)
    
    # Print results
    print("\nModel Performance Metrics:")
    for model, scores in metrics.items():
        print(f"\n{model}:")
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")
    
    print("\nFeature Importance:")
    for feature, importance in importance.items():
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main()