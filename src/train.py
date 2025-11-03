from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn


class ModelTrainer:
    def __init__(self, input_path, model_dir, target_col='membresia_premium_Sí', test_size=0.25, random_state=42):
        self.input_path = input_path
        self.model_dir = Path(model_dir)
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        
        mlflow.set_experiment("restaurant_premium_prediction")
    
    def load_data(self):
        df = pd.read_parquet(self.input_path)
        
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Loaded data: {X.shape}")
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self
    
    def initialize_models(self):
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        }
        print(f"Initialized {len(self.models)} models")
        return self
    
    def train_models(self):
        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                print(f"\nTraining {name}...")
                model.fit(self.X_train, self.y_train)
                
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)[:, 1]
                
                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1_score': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_proba)
                }
                
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, f"model_{name}")
                
                self.results[name] = {**metrics, 'model': model}
                
                print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                      f"Recall: {metrics['recall']:.4f}, "
                      f"F1: {metrics['f1_score']:.4f}, "
                      f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return self
    
    def select_best_model(self, metric='recall'):
        best_score = 0
        for name, result in self.results.items():
            if result[metric] > best_score:
                best_score = result[metric]
                self.best_model_name = name
                self.best_model = result['model']
        
        print(f"\nBest model: {self.best_model_name} ({metric}: {best_score:.4f})")
        return self
    
    def save_best_model(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / f'best_model_{self.best_model_name}.pkl'
        
        joblib.dump(self.best_model, model_path)
        print(f"Saved best model to {model_path}")
        
        return self
    
    def run_pipeline(self):
        self.load_data()
        self.initialize_models()
        self.train_models()
        self.select_best_model()
        self.save_best_model()
        return self.best_model


def main():
    trainer = ModelTrainer(
        input_path='data/processed/restaurante_clean.parquet',
        model_dir='models/'
    )
    
    best_model = trainer.run_pipeline()
    return best_model


if __name__ == "__main__":
    main()