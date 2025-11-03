from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


class ModelEvaluator:
    def __init__(self, model_path, data_path, results_path, output_dir='models/evaluation'):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_col = 'membresia_premium_Sí'
        
        self.performance_metrics = {}
        self.consistency_scores = {}
        self.scalability_results = {}
    
    def load_model_and_data(self):
        self.model = joblib.load(self.model_path)
        
        df = pd.read_parquet(self.data_path)
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Loaded model: {self.model_path.name}")
        print(f"Data shape: {X.shape}")
        return self
    
    def evaluate_performance(self):
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        self.performance_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_proba)
        }
        
        print("\nPerformance Metrics:")
        for metric, value in self.performance_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        return self
    
    def evaluate_consistency(self, cv_folds=5):
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        print("\nConsistency Evaluation (Cross-Validation):")
        for metric in scoring_metrics:
            scores = cross_val_score(
                self.model, self.X_train, self.y_train, 
                cv=cv_folds, scoring=metric, n_jobs=-1
            )
            
            self.consistency_scores[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            print(f"  {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return self
    
    def evaluate_scalability(self):
        common_params = {
            "X": self.X_train,
            "y": self.y_train,
            "train_sizes": np.linspace(0.1, 1.0, 5),
            "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=42),
            "n_jobs": -1,
            "return_times": True,
        }
        
        scoring_metric = "recall"
        
        print("\nScalability Evaluation (Learning Curve):")
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
            self.model, **common_params, scoring=scoring_metric
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        score_times_mean = np.mean(score_times, axis=1)
        score_times_std = np.std(score_times, axis=1)
        
        self.scalability_results = {
            'train_sizes': train_sizes,
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'fit_times_mean': fit_times_mean,
            'fit_times_std': fit_times_std,
            'score_times_mean': score_times_mean,
            'score_times_std': score_times_std
        }
        
        print(f"  Training Sizes: {train_sizes}")
        print(f"  Test Scores: {test_mean}")
        print(f"  Fit Times (s): {fit_times_mean}")
        
        self.plot_learning_curve(scoring_metric)
        self.plot_scalability_time()
        
        return self
    
    def plot_learning_curve(self, scoring_metric):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        r = self.scalability_results
        ax.plot(r['train_sizes'], r['train_mean'], 'o-', label='Training score')
        ax.plot(r['train_sizes'], r['test_mean'], 'o-', color='orange', label='Cross-validation score')
        ax.fill_between(r['train_sizes'], 
                        r['train_mean'] - r['train_std'], 
                        r['train_mean'] + r['train_std'], alpha=0.3)
        ax.fill_between(r['train_sizes'], 
                        r['test_mean'] - r['test_std'], 
                        r['test_mean'] + r['test_std'], alpha=0.3, color='orange')
        
        ax.set_title(f"Learning Curve - {self.model.__class__.__name__}")
        ax.set_xlabel("Training examples")
        ax.set_ylabel(scoring_metric)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curve.png')
        print(f"\nSaved learning curve to {self.output_dir / 'learning_curve.png'}")
        plt.close()
        
        return self
    
    def plot_scalability_time(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        r = self.scalability_results
        
        axes[0].plot(r['train_sizes'], r['fit_times_mean'], 'o-')
        axes[0].fill_between(r['train_sizes'], 
                            r['fit_times_mean'] - r['fit_times_std'],
                            r['fit_times_mean'] + r['fit_times_std'], alpha=0.3)
        axes[0].set_title("Fit Time vs Training Size")
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Fit time (seconds)")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(r['train_sizes'], r['score_times_mean'], 'o-', color='green')
        axes[1].fill_between(r['train_sizes'],
                            r['score_times_mean'] - r['score_times_std'],
                            r['score_times_mean'] + r['score_times_std'], alpha=0.3, color='green')
        axes[1].set_title("Score Time vs Training Size")
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("Score time (seconds)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scalability_time.png')
        print(f"Saved scalability time plot to {self.output_dir / 'scalability_time.png'}")
        plt.close()
        
        return self
    
    def save_results(self):
        results = {
            'performance': self.performance_metrics,
            'consistency': self.consistency_scores,
            'scalability': self.scalability_results
        }
        
        results_file = self.output_dir / 'evaluation_results.pkl'
        joblib.dump(results, results_file)
        print(f"\nSaved all results to {results_file}")
        
        return self
    
    def run_full_evaluation(self):
        self.load_model_and_data()
        self.evaluate_performance()
        self.evaluate_consistency()
        self.evaluate_scalability()
        self.save_results()
        
        print("\nEvaluation completed!")
        return self


def main():
    model_files = list(Path('models/').glob('best_model_*.pkl'))
    
    if not model_files:
        print("No model found in models/ directory")
        return
    
    model_path = model_files[0]
    
    evaluator = ModelEvaluator(
        model_path=model_path,
        data_path='data/processed/restaurante_clean.parquet',
        results_path='models/training_results.pkl'
    )
    
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()