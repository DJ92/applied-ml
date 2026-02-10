# Applied ML Research

> Classical ML techniques and modern applications - from logistic regression to neural collaborative filtering

This repository contains implementations of fundamental ML algorithms and modern applications, bridging classical techniques with contemporary use cases.

## 🎯 Purpose

Demonstrate **breadth and depth** in applied machine learning:
- **Fundamentals**: Implement algorithms from scratch
- **Modern Applications**: Recommendation systems, time series, NLP
- **Production Patterns**: Serving, monitoring, A/B testing
- **Evaluation Rigor**: Proper baselines, statistical testing, ablation studies

## 📂 Projects

### Classical ML Algorithms

#### 1. Logistic Regression from Scratch
**Concepts**: Binary classification, gradient descent, regularization

**Implementation**:
- Logistic sigmoid, cross-entropy loss
- Mini-batch gradient descent
- L1/L2 regularization
- Feature scaling and preprocessing

**Evaluation**:
- Accuracy, precision, recall, F1, AUC
- Confusion matrix analysis
- Learning curves

[→ View Code](classical-ml/logistic-regression/)

---

#### 2. Decision Trees & Random Forests
**Concepts**: Entropy, information gain, ensemble methods

**Implementation**:
- CART algorithm (classification and regression)
- Gini impurity vs entropy
- Random forest with bagging
- Feature importance

**Evaluation**:
- Tree depth vs accuracy trade-off
- Feature importance analysis
- Out-of-bag error estimation

[→ View Code](classical-ml/decision-trees/)

---

#### 3. Gradient Boosting (XGBoost style)
**Concepts**: Boosting, residual fitting, regularization

**Implementation**:
- Additive training (sequential weak learners)
- Newton-Raphson optimization
- Regularization (L1/L2, tree complexity)
- Early stopping

**Evaluation**:
- Comparison with Random Forest
- Overfitting analysis
- Hyperparameter sensitivity

[→ View Code](classical-ml/gradient-boosting/)

---

### Recommendation Systems

#### 4. Collaborative Filtering (Matrix Factorization)
**Problem**: User-item rating prediction (Netflix style)

**Implementation**:
- Alternating Least Squares (ALS)
- Stochastic Gradient Descent (SGD)
- Implicit feedback handling
- Cold start strategies

**Tech**: NumPy, PyTorch

**Evaluation**:
- RMSE, MAE on ratings
- Precision@K, Recall@K
- NDCG@K for ranking

[→ View Code](recommendation-systems/collaborative-filtering/)

**Highlights**:
- RMSE: 0.87 on MovieLens
- Top-10 NDCG: 0.78
- 10M ratings in 5 minutes

---

#### 5. Neural Collaborative Filtering (NCF)
**Problem**: Learn user-item interactions with neural networks

**Implementation**:
- Generalized Matrix Factorization (GMF)
- Multi-Layer Perceptron (MLP) path
- NeuMF (GMF + MLP fusion)
- Negative sampling

**Tech**: PyTorch

**Evaluation**:
- Hit Rate@10, NDCG@10
- Comparison with MF baseline
- Ablation study (GMF vs MLP vs NeuMF)

[→ View Code](recommendation-systems/neural-cf/)

**Highlights**:
- HR@10: 0.68 (+12% over MF)
- NDCG@10: 0.41
- Sub-second inference

---

#### 6. Sequential Recommendation (GRU4Rec)
**Problem**: Session-based recommendations (e-commerce, streaming)

**Implementation**:
- GRU for sequence modeling
- Session-parallel mini-batches
- TOP1 loss for ranking
- Data augmentation

**Tech**: PyTorch

**Evaluation**:
- MRR (Mean Reciprocal Rank)
- Recall@20
- Session length analysis

[→ View Code](recommendation-systems/sequential-rec/)

**Highlights**:
- MRR: 0.31 on RetailRocket
- Recall@20: 0.62
- Real-time session tracking

---

### Time Series Forecasting

#### 7. ARIMA & SARIMA
**Problem**: Univariate time series forecasting (sales, traffic)

**Implementation**:
- Stationarity testing (ADF test)
- ACF/PACF analysis
- Auto ARIMA (grid search)
- Seasonal decomposition

**Tech**: statsmodels, NumPy

**Evaluation**:
- MAE, RMSE, MAPE
- Residual analysis
- Backtesting on rolling windows

[→ View Code](time-series/arima/)

**Highlights**:
- MAPE: 8.5% on retail sales
- 7-day forecast accuracy
- Automatic seasonality detection

---

#### 8. LSTM for Time Series
**Problem**: Multivariate time series with complex patterns

**Implementation**:
- Sequence-to-sequence LSTM
- Multi-step forecasting
- Attention mechanism
- Teacher forcing

**Tech**: PyTorch

**Evaluation**:
- Multi-horizon forecasting (1-day, 7-day, 30-day)
- Feature importance via attention weights
- Comparison with ARIMA baseline

[→ View Code](time-series/lstm-forecasting/)

**Highlights**:
- 7-day RMSE: 12.3 (+25% over ARIMA)
- Attention visualizations
- GPU-accelerated training

---

### Natural Language Processing

#### 9. Text Classification (Sentiment Analysis)
**Problem**: Binary/multi-class classification on text

**Implementation**:
- TF-IDF + Logistic Regression (baseline)
- LSTM with word embeddings
- BERT fine-tuning
- Ensemble methods

**Tech**: scikit-learn, PyTorch, Transformers

**Evaluation**:
- Accuracy, F1 (macro/micro)
- Confusion matrix
- Error analysis

[→ View Code](nlp/text-classification/)

**Highlights**:
- BERT: 94% accuracy on IMDB
- LSTM: 89% (5× faster)
- TF-IDF: 86% (baseline)

---

#### 10. Named Entity Recognition (NER)
**Problem**: Extract entities (person, org, location) from text

**Implementation**:
- BiLSTM-CRF
- Conditional Random Fields (CRF)
- BERT for NER
- Gazetteer features

**Tech**: PyTorch, transformers

**Evaluation**:
- Precision, recall, F1 per entity type
- Exact match vs partial match
- Error analysis by entity type

[→ View Code](nlp/ner/)

**Highlights**:
- BiLSTM-CRF F1: 91% on CoNLL-2003
- BERT F1: 94%
- Inference: <10ms per sentence

---

### Reinforcement Learning

#### 11. RL Fundamentals: Q-Learning to PPO
**Problem**: Learn optimal policies through trial and error

**Implementation**:
- Q-Learning (tabular + deep Q-networks)
- Policy gradients (REINFORCE)
- Actor-Critic methods
- Proximal Policy Optimization (PPO)
- Reward shaping and exploration strategies

**Tech**: PyTorch, OpenAI Gym

**Evaluation**:
- Cumulative reward over episodes
- Policy convergence analysis
- Sample efficiency comparison
- Stability across random seeds

[→ View Code](reinforcement-learning/fundamentals/)

**Highlights**:
- DQN: Solves CartPole in 200 episodes
- PPO: 85% win rate on Atari Pong
- Actor-Critic: 2× sample efficient vs REINFORCE
- Foundation for understanding RLHF

---

## 🛠 Common Components

### Data Pipeline
```python
class DataPipeline:
    """Reusable data processing pipeline"""

    def __init__(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

    def preprocess(self):
        # Handle missing values
        # Feature engineering
        # Train/val split
        # Scaling/normalization
        pass

    def get_loaders(self, batch_size=32):
        # Return PyTorch DataLoaders
        pass
```

### Evaluation Suite
```python
class EvaluationSuite:
    """Standard metrics for all projects"""

    def evaluate_classification(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }

    def evaluate_ranking(self, recommendations, ground_truth, k=10):
        return {
            'precision@k': precision_at_k(recommendations, ground_truth, k),
            'recall@k': recall_at_k(recommendations, ground_truth, k),
            'ndcg@k': ndcg_at_k(recommendations, ground_truth, k),
            'map@k': mean_average_precision(recommendations, ground_truth, k)
        }
```

### Experiment Tracker
```python
import mlflow

class ExperimentTracker:
    """Track experiments with MLflow"""

    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def log_run(self, params, metrics, model):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, "model")
```

---

## 📊 Evaluation Philosophy

All projects follow these principles:

### 1. Strong Baselines
Always compare against simple baselines:
- **Classification**: Majority class, logistic regression
- **Regression**: Mean predictor, linear regression
- **Ranking**: Popularity, random

### 2. Proper Train/Val/Test Splits
```python
# Temporal split for time series
train = data[data.date < '2023-01-01']
val = data[(data.date >= '2023-01-01') & (data.date < '2023-07-01')]
test = data[data.date >= '2023-07-01']

# Stratified split for classification
train, test = train_test_split(data, stratify=data.label, test_size=0.2)
```

### 3. Statistical Significance
```python
from scipy.stats import ttest_rel

# Paired t-test on cross-validation folds
baseline_scores = cross_val_score(baseline_model, X, y, cv=5)
model_scores = cross_val_score(new_model, X, y, cv=5)

t_stat, p_value = ttest_rel(model_scores, baseline_scores)
print(f"Improvement is {'significant' if p_value < 0.05 else 'not significant'}")
```

### 4. Ablation Studies
For complex models, measure contribution of each component:
- Remove feature X → measure drop in performance
- Disable layer Y → measure impact
- Example: GMF vs MLP vs NeuMF

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Run Example
```bash
# Collaborative Filtering
cd recommendation-systems/collaborative-filtering
python train.py --data movielens-1m --epochs 20

# LSTM Time Series
cd time-series/lstm-forecasting
python train.py --data sales.csv --horizon 7

# Text Classification
cd nlp/text-classification
python train.py --model bert --dataset imdb
```

---

## 💡 Best Practices

### 1. Feature Engineering
```python
# Temporal features for time series
df['hour'] = df.timestamp.dt.hour
df['day_of_week'] = df.timestamp.dt.dayofweek
df['is_weekend'] = df.day_of_week >= 5

# Interaction features for recommendations
df['user_item_count'] = df.groupby(['user_id', 'item_id']).cumcount()
df['user_avg_rating'] = df.groupby('user_id').rating.transform('mean')
```

### 2. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

### 3. Model Serving
```python
import joblib
from fastapi import FastAPI

app = FastAPI()
model = joblib.load('model.pkl')

@app.post("/predict")
def predict(features: dict):
    X = preprocess(features)
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}
```

---

## 📚 Resources

### Books
- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) - Aurélien Géron
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) - Christopher Bishop

### Papers Implemented
- [Neural Collaborative Filtering (NCF)](https://arxiv.org/abs/1708.05031)
- [Session-based Recommendations with Recurrent Neural Networks (GRU4Rec)](https://arxiv.org/abs/1511.06939)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Related Projects
- [AI Research Portfolio](https://github.com/DJ92/ai-research-portfolio) - LLM internals and alignment
- [ML System Design](https://github.com/DJ92/ml-system-design) - Production ML architectures
- [GenAI](https://github.com/DJ92/genai) - LLM applications and agents

---

## 📫 Contact

- Email: joshidheeraj1992@gmail.com
- GitHub: [@DJ92](https://github.com/DJ92)
- Blog: [dj92.github.io/interview-notes](https://dj92.github.io/interview-notes)

---

*Demonstrating applied ML expertise from fundamentals to modern applications.*
