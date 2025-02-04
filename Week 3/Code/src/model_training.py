from sklearn.ensemble import RandomForestClassifier         # type: ignore
from sklearn.linear_model import LogisticRegression         # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score     # type: ignore
from sklearn.model_selection import cross_val_score         # type: ignore
import numpy as np                                          # type: ignore

class ChurnPredictor:
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_estimators=200,
            max_depth=10
        )
        self.lr_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        
    def train_models(self, X_train, y_train):
        self.rf_model.fit(X_train, y_train)
        
        self.lr_model.fit(X_train, y_train)
        
    def evaluate_models(self, X_test, y_test):
        results = {}
        
        for name, model in [('Random Forest', self.rf_model), 
                          ('Logistic Regression', self.lr_model)]:
            y_pred = model.predict(X_test)
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
        return results
    
    def get_feature_importance(self, feature_names):
        importance = self.rf_model.feature_importances_
        feature_imp = dict(zip(feature_names, importance))
        return dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))