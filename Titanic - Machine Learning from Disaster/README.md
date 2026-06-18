# 🚢 Titanic Survival Prediction

A complete end-to-end machine learning project built on the Titanic dataset from the Kaggle competition. This project explores data analysis, feature engineering, model experimentation, hyperparameter tuning, and ensemble learning techniques to predict passenger survival.

## Project Overview

The objective is to predict whether a passenger survived the Titanic disaster using passenger demographics, ticket information, travel class, and other available features.

The project focuses on:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training
- Cross Validation
- Hyperparameter Optimization
- Ensemble Learning
- Submission Generation

---

# Dataset

The dataset contains information such as:

| Feature | Description |
|----------|------------|
| PassengerId | Unique passenger identifier |
| Survived | Target variable |
| Pclass | Passenger class |
| Name | Passenger name |
| Sex | Gender |
| Age | Passenger age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Ticket fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation |

---

# Project Workflow

```text
Load Data
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Data Preprocessing
    ↓
Feature Encoding
    ↓
Feature Scaling
    ↓
Model Training
    ↓
Cross Validation
    ↓
Hyperparameter Tuning
    ↓
Ensemble Learning
    ↓
Submission Generation
```

---

# Exploratory Data Analysis (EDA)

## Dataset Inspection

Initial exploration included:

```python
t_data.info()
t_data.describe()
```

Performed:

- Dataset structure inspection
- Data type analysis
- Missing value detection
- Statistical summaries

---

## Numerical Feature Analysis

Numerical variables examined:

```python
Age
SibSp
Parch
Fare
```

### Methods Used

- Histograms
- Correlation Matrix
- Heatmaps
- Survival-based Pivot Tables

### Correlation Analysis

```python
sns.heatmap(df_num.corr())
```

Used to identify relationships between numerical features.

---

## Survival Analysis

Compared survival rates across:

- Age
- Fare
- SibSp
- Parch

Using:

```python
pd.pivot_table(
    t_data,
    index='Survived',
    values=['Age','SibSp','Parch','Fare']
)
```

---

## Categorical Feature Analysis

Analyzed:

```python
Pclass
Sex
Ticket
Cabin
Embarked
```

### Methods Used

- Count plots
- Bar plots
- Pivot tables
- Frequency distributions

### Key Insights

- Female passengers had significantly higher survival rates.
- First-class passengers survived more frequently.
- Fare correlated positively with survival probability.
- Cabin information contained valuable hidden patterns.
- Family-related features influenced survival outcomes.

---

# Feature Engineering Experiments

A major focus of this project was creating new features from existing information.

---

## 1. Cabin Features

### Cabin Count Feature

Created:

```python
cabin_multiple
```

Implementation:

```python
t_data['cabin_multiple'] = t_data.Cabin.apply(
    lambda x: 0 if pd.isna(x) else len(x.split(' '))
)
```

Purpose:

- Identify passengers associated with multiple cabins.
- Capture possible socioeconomic indicators.

---

### Cabin Deck Extraction

Created:

```python
cabin_adv
```

Implementation:

```python
t_data['cabin_adv'] = t_data.Cabin.apply(
    lambda x: str(x)[0]
)
```

Extracted deck information:

```text
A
B
C
D
E
F
G
T
```

Purpose:

- Approximate passenger location on the ship.
- Capture survival patterns by deck.

---

## 2. Ticket-Based Features

### Numeric Ticket Indicator

Created:

```python
numeric_ticket
```

Implementation:

```python
lambda x: 1 if x.isnumeric() else 0
```

Purpose:

- Differentiate numeric and non-numeric ticket formats.

---

### Ticket Prefix Extraction

Created:

```python
ticket_letters
```

Implementation:

```python
''.join(x.split(' ')[:-1])
```

Purpose:

- Capture ticket group information.
- Identify booking patterns.

---

## 3. Passenger Title Extraction

Created:

```python
name_title
```

Implementation:

```python
t_data.Name.apply(
    lambda x: x.split(',')[1].split('.')[0].strip()
)
```

Examples:

```text
Mr
Mrs
Miss
Master
Dr
Rev
Col
```

Purpose:

- Encode social status.
- Capture age and gender signals.
- Improve predictive performance.

---

## 4. Fare Transformation

Applied logarithmic scaling:

```python
d_set['norm_fare'] = np.log(d_set.Fare + 1)
```

Purpose:

- Reduce skewness.
- Improve model stability.
- Handle extreme fare values.

---

## 5. Sibling/Spouse Transformation

Experimented with:

```python
norm_sibsp = np.log(SibSp + 1)
```

Although explored, this feature was not ultimately used in the final model.

---

# Data Preprocessing

## Missing Value Handling

### Age

Filled using median age:

```python
Age.fillna(median)
```

### Fare

Filled using median fare:

```python
Fare.fillna(median)
```

### Embarked

Rows with missing values were removed:

```python
dropna(subset=['Embarked'])
```

---

## Encoding Categorical Variables

Used one-hot encoding:

```python
pd.get_dummies()
```

Encoded:

- Pclass
- Sex
- Embarked
- Cabin Deck
- Cabin Count
- Ticket Features
- Passenger Titles

---

## Feature Scaling

Used:

```python
StandardScaler()
```

Scaled features:

```python
Age
SibSp
Parch
norm_fare
```

Required primarily for:

- KNN
- Logistic Regression
- SVM

---

# Model Training Pipelines

Several machine learning algorithms were trained and compared.

---

## 1. Gaussian Naive Bayes

Used as the baseline model.

```python
GaussianNB()
```

Purpose:

- Fast benchmark.
- Simple probabilistic classifier.

---

## 2. Logistic Regression

```python
LogisticRegression(max_iter=2000)
```

Advantages:

- Strong baseline
- Interpretable coefficients
- Fast training

---

## 3. Decision Tree Classifier

```python
DecisionTreeClassifier()
```

Advantages:

- Non-linear decision boundaries
- Easy interpretation

---

## 4. K-Nearest Neighbors (KNN)

```python
KNeighborsClassifier()
```

Advantages:

- Simple implementation
- Effective after feature scaling

---

## 5. Random Forest Classifier

```python
RandomForestClassifier()
```

Advantages:

- Reduced overfitting
- Handles feature interactions well
- Provides feature importance scores

---

## 6. Support Vector Machine (SVM)

```python
SVC(probability=True)
```

Advantages:

- Strong performance on structured data
- Effective in high-dimensional spaces

---

## 7. XGBoost

```python
XGBClassifier()
```

Advantages:

- Gradient boosting
- Handles complex feature interactions
- Strong Kaggle performance

---

# Validation Strategies

## 5-Fold Cross Validation

All models were evaluated using:

```python
cross_val_score(model, X, y, cv=5)
```

Benefits:

- Reliable performance estimates
- Reduced variance
- Fair model comparison

---

# Hyperparameter Tuning

## GridSearchCV

Used to optimize:

- Logistic Regression
- KNN
- SVM
- Random Forest
- XGBoost

---

## Logistic Regression Search Space

```python
C
Penalty
Solver
```

---

## KNN Search Space

```python
n_neighbors
weights
algorithm
p
```

---

## SVM Search Space

```python
kernel
gamma
C
degree
```

---

## Random Forest Search Space

```python
n_estimators
criterion
max_depth
max_features
min_samples_split
min_samples_leaf
```

---

## XGBoost Search Space

```python
n_estimators
learning_rate
subsample
gamma
reg_alpha
reg_lambda
colsample_bytree
```

---

# Feature Importance Analysis

Random Forest feature importance was used to understand model behavior.

```python
best_rf.feature_importances_
```

Top contributing features generally included:

- Sex
- Passenger Class
- Fare
- Age
- Cabin Features
- Passenger Titles

---

# Ensemble Learning

The project explored several ensemble strategies.

---

## Hard Voting Classifier

Combined:

```python
KNN
Random Forest
SVM
```

Voting strategy:

```python
voting='hard'
```

Uses majority voting.

---

## Soft Voting Classifier

Combined:

```python
KNN
Random Forest
SVM
```

Voting strategy:

```python
voting='soft'
```

Uses prediction probabilities.

---

## Full Ensemble

Combined:

```python
Logistic Regression
KNN
Random Forest
Gaussian Naive Bayes
SVM
```

Purpose:

- Leverage strengths of multiple models.
- Improve generalization.

---

## XGBoost Ensemble

Combined:

```python
KNN
Random Forest
SVM
XGBoost
```

Purpose:

- Blend boosting with traditional classifiers.

---

These files enabled comparison between:

- Single tuned models
- Voting classifiers
- Ensemble approaches

---

# Notes and Lessons Learned

## What Worked Well

- Feature engineering significantly improved performance.
- Passenger titles provided strong predictive signals.
- Cabin-derived features added useful information.
- Ensemble models improved consistency.
- Log-transforming fare improved distribution quality.
- Scaling benefited KNN and SVM substantially.

---

## Challenges

- Missing values required careful handling.
- Ticket information was noisy and difficult to interpret.
- Hyperparameter tuning was computationally expensive.
- Ensemble models increased complexity and training time.

---

## Key Takeaways

### 1. Feature Engineering Matters

Meaningful engineered features often contributed more than changing algorithms.

### 2. Domain Knowledge Helps

Cabin location, ticket structure, and passenger titles contained valuable information.

### 3. Cross Validation Is Essential

A single train/test split can produce misleading conclusions.

### 4. Scaling Depends On The Model

Beneficial for:

- KNN
- Logistic Regression
- SVM

Less important for:

- Random Forest
- Decision Trees

### 5. Ensembles Improve Robustness

Combining multiple models reduced variance and improved overall performance.

---

# Future Improvements

Potential next steps:

- Family size feature engineering
- Ticket group clustering
- Target encoding
- Feature selection
- CatBoost implementation
- LightGBM implementation
- Stacking ensembles
- Bayesian hyperparameter optimization

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost

---
