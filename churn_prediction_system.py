"""
Churn Prediction System
A comprehensive machine learning solution for predicting customer churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChurnPredictionSystem:
    """
    A comprehensive churn prediction system that includes:
    - Data preprocessing and feature engineering
    - Multiple ML model training and evaluation
    - Business insights and visualizations
    - Model deployment capabilities
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self, file_path=None, use_sample=True):
        """
        Load customer churn dataset
        If no file provided, will use sample telecom data
        """
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                print(f"Data loaded successfully from {file_path}")
            except Exception as e:
                print(f"Error loading data: {e}")
                if use_sample:
                    print("Using sample data instead...")
                    self._create_sample_data()
        else:
            print("Creating sample telecom customer churn dataset...")
            self._create_sample_data()
            
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        return self.data
    
    def _create_sample_data(self):
        """Create a realistic sample telecom customer churn dataset"""
        np.random.seed(42)
        n_customers = 5000
        
        # Generate realistic customer data
        data = {
            'customerID': [f'CUST_{i:06d}' for i in range(1, n_customers + 1)],
            'gender': np.random.choice(['Male', 'Female'], n_customers),
            'SeniorCitizen': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_customers, p=[0.5, 0.5]),
            'Dependents': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
            'tenure': np.random.randint(1, 73, n_customers),  # 1-72 months
            'PhoneService': np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_customers, p=[0.4, 0.5, 0.1]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.3, 0.4, 0.3]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.2, 0.6, 0.2]),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.3, 0.5, 0.2]),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.3, 0.5, 0.2]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.2, 0.6, 0.2]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.3, 0.5, 0.2]),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.3, 0.5, 0.2]),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.25, 0.2]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_customers, p=[0.6, 0.4]),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                            n_customers, p=[0.3, 0.2, 0.25, 0.25]),
            'MonthlyCharges': np.round(np.random.normal(65, 25, n_customers), 2),
            'TotalCharges': np.round(np.random.normal(2300, 1500, n_customers), 2)
        }
        
        # Create realistic churn based on business logic
        churn_prob = []
        for i in range(n_customers):
            prob = 0.1  # Base churn probability
            
            # Higher churn for month-to-month contracts
            if data['Contract'][i] == 'Month-to-month':
                prob += 0.3
            
            # Higher churn for electronic check payment
            if data['PaymentMethod'][i] == 'Electronic check':
                prob += 0.2
            
            # Higher churn for high monthly charges
            if data['MonthlyCharges'][i] > 80:
                prob += 0.15
            
            # Higher churn for short tenure
            if data['tenure'][i] < 12:
                prob += 0.2
            
            # Higher churn for no online security
            if data['OnlineSecurity'][i] == 'No':
                prob += 0.1
            
            churn_prob.append(min(prob, 0.9))  # Cap at 90%
        
        # Generate churn labels
        data['Churn'] = np.random.binomial(1, churn_prob)
        
        self.data = pd.DataFrame(data)
        
        # Clean up TotalCharges for new customers
        self.data.loc[self.data['tenure'] == 0, 'TotalCharges'] = 0
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
        self.data['TotalCharges'].fillna(0, inplace=True)
        
        print("Sample telecom dataset created successfully!")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print("\nDataset Overview:")
        print(f"Shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\nData Types:")
        print(self.data.dtypes)
        
        # Missing values
        print("\nMissing Values:")
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")
        
        # Target variable distribution
        print("\nTarget Variable (Churn) Distribution:")
        churn_counts = self.data['Churn'].value_counts()
        churn_pct = self.data['Churn'].value_counts(normalize=True) * 100
        print(f"Non-Churn: {churn_counts[0]} ({churn_pct[0]:.1f}%)")
        print(f"Churn: {churn_counts[1]} ({churn_pct[1]:.1f}%)")
        
        # Numerical features summary
        print("\nNumerical Features Summary:")
        print(self.data.describe())
        
        return self.data
    
    def visualize_data(self):
        """Create comprehensive data visualizations"""
        print("Creating data visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Churn distribution
        plt.subplot(4, 3, 1)
        churn_counts = self.data['Churn'].value_counts()
        colors = ['#2E8B57', '#DC143C']
        plt.pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold')
        
        # 2. Tenure distribution by churn
        plt.subplot(4, 3, 2)
        sns.histplot(data=self.data, x='tenure', hue='Churn', kde=True, alpha=0.7)
        plt.title('Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
        plt.xlabel('Tenure (months)')
        
        # 3. Monthly charges by churn
        plt.subplot(4, 3, 3)
        sns.boxplot(data=self.data, x='Churn', y='MonthlyCharges')
        plt.title('Monthly Charges by Churn Status', fontsize=14, fontweight='bold')
        plt.xlabel('Churn')
        plt.ylabel('Monthly Charges ($)')
        
        # 4. Contract type vs churn
        plt.subplot(4, 3, 4)
        contract_churn = pd.crosstab(self.data['Contract'], self.data['Churn'], normalize='index') * 100
        contract_churn.plot(kind='bar', stacked=True, color=['#2E8B57', '#DC143C'])
        plt.title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
        plt.xlabel('Contract Type')
        plt.ylabel('Percentage')
        plt.legend(['No Churn', 'Churn'])
        plt.xticks(rotation=45)
        
        # 5. Payment method vs churn
        plt.subplot(4, 3, 5)
        payment_churn = pd.crosstab(self.data['PaymentMethod'], self.data['Churn'], normalize='index') * 100
        payment_churn.plot(kind='bar', stacked=True, color=['#2E8B57', '#DC143C'])
        plt.title('Churn Rate by Payment Method', fontsize=14, fontweight='bold')
        plt.xlabel('Payment Method')
        plt.ylabel('Percentage')
        plt.legend(['No Churn', 'Churn'])
        plt.xticks(rotation=45)
        
        # 6. Internet service vs churn
        plt.subplot(4, 3, 6)
        internet_churn = pd.crosstab(self.data['InternetService'], self.data['Churn'], normalize='index') * 100
        internet_churn.plot(kind='bar', stacked=True, color=['#2E8B57', '#DC143C'])
        plt.title('Churn Rate by Internet Service', fontsize=14, fontweight='bold')
        plt.xlabel('Internet Service')
        plt.ylabel('Percentage')
        plt.legend(['No Churn', 'Churn'])
        
        # 7. Senior citizen vs churn
        plt.subplot(4, 3, 7)
        senior_churn = pd.crosstab(self.data['SeniorCitizen'], self.data['Churn'], normalize='index') * 100
        senior_churn.plot(kind='bar', stacked=True, color=['#2E8B57', '#DC143C'])
        plt.title('Churn Rate by Senior Citizen Status', fontsize=14, fontweight='bold')
        plt.xlabel('Senior Citizen')
        plt.ylabel('Percentage')
        plt.legend(['No Churn', 'Churn'])
        plt.xticks([0, 1], ['No', 'Yes'])
        
        # 8. Total charges distribution
        plt.subplot(4, 3, 8)
        sns.histplot(data=self.data, x='TotalCharges', hue='Churn', kde=True, alpha=0.7)
        plt.title('Total Charges Distribution by Churn', fontsize=14, fontweight='bold')
        plt.xlabel('Total Charges ($)')
        
        # 9. Correlation heatmap
        plt.subplot(4, 3, 9)
        # Select numerical columns for correlation
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn']
        corr_matrix = self.data[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 10. Churn rate by tenure groups
        plt.subplot(4, 3, 10)
        self.data['tenure_group'] = pd.cut(self.data['tenure'], 
                                         bins=[0, 12, 24, 48, 72], 
                                         labels=['0-12', '13-24', '25-48', '49-72'])
        tenure_churn = pd.crosstab(self.data['tenure_group'], self.data['Churn'], normalize='index') * 100
        tenure_churn.plot(kind='bar', stacked=True, color=['#2E8B57', '#DC143C'])
        plt.title('Churn Rate by Tenure Groups', fontsize=14, fontweight='bold')
        plt.xlabel('Tenure Group (months)')
        plt.ylabel('Percentage')
        plt.legend(['No Churn', 'Churn'])
        
        # 11. Monthly charges vs tenure scatter
        plt.subplot(4, 3, 11)
        sns.scatterplot(data=self.data, x='tenure', y='MonthlyCharges', hue='Churn', alpha=0.6)
        plt.title('Monthly Charges vs Tenure', fontsize=14, fontweight='bold')
        plt.xlabel('Tenure (months)')
        plt.ylabel('Monthly Charges ($)')
        
        # 12. Feature importance preview (placeholder)
        plt.subplot(4, 3, 12)
        # This will be updated after model training
        plt.text(0.5, 0.5, 'Feature Importance\n(Will be updated after model training)', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Data visualizations saved as 'data_exploration.png'")
    
    def preprocess_data(self):
        """Clean and preprocess the data for machine learning"""
        print("PREPROCESSING DATA")
        print("=" * 50)
        
        # Create a copy for preprocessing
        df = self.data.copy()
        
        # Handle missing values
        print("1. Handling missing values...")
        if df['TotalCharges'].isnull().any():
            df['TotalCharges'].fillna(0, inplace=True)
        
        # Feature engineering
        print("2. Feature engineering...")
        
        # Create tenure groups
        df['tenure_group'] = pd.cut(df['tenure'], 
                                   bins=[0, 12, 24, 48, 72], 
                                   labels=['New', 'Regular', 'Loyal', 'VIP'])
        
        # Create monthly charges groups
        df['monthly_charges_group'] = pd.cut(df['MonthlyCharges'], 
                                           bins=[0, 35, 70, 100, 200], 
                                           labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Create total charges groups
        df['total_charges_group'] = pd.cut(df['TotalCharges'], 
                                         bins=[0, 1000, 3000, 5000, 10000], 
                                         labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Create service count features
        service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Count active services
        df['service_count'] = 0
        for col in service_columns:
            if col in df.columns:
                df['service_count'] += (df[col] == 'Yes').astype(int)
        
        # Create high-value customer flag
        df['high_value_customer'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)) & 
                                   (df['tenure'] > df['tenure'].quantile(0.5))).astype(int)
        
        # Create risk score based on multiple factors
        df['risk_score'] = 0
        df['risk_score'] += (df['Contract'] == 'Month-to-month').astype(int) * 3
        df['risk_score'] += (df['PaymentMethod'] == 'Electronic check').astype(int) * 2
        df['risk_score'] += (df['tenure'] < 12).astype(int) * 2
        df['risk_score'] += (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.8)).astype(int) * 1
        df['risk_score'] += (df['OnlineSecurity'] == 'No').astype(int) * 1
        
        # Encode categorical variables
        print("3. Encoding categorical variables...")
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        categorical_columns.remove('customerID')  # Remove ID column
        
        # Use label encoding for ordinal variables and one-hot for nominal
        ordinal_columns = ['tenure_group', 'monthly_charges_group', 'total_charges_group']
        nominal_columns = [col for col in categorical_columns if col not in ordinal_columns]
        
        # Label encode ordinal variables
        le_dict = {}
        for col in ordinal_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                le_dict[col] = le
        
        # One-hot encode nominal variables
        df_encoded = pd.get_dummies(df, columns=nominal_columns, drop_first=True)
        
        # Prepare features and target
        print("4. Preparing features and target...")
        feature_columns = [col for col in df_encoded.columns 
                          if col not in ['customerID', 'Churn']]
        
        X = df_encoded[feature_columns]
        y = df_encoded['Churn']
        
        # Store customer IDs for later reference
        self.customer_ids = df_encoded['customerID']
        
        print(f"Preprocessing complete!")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Samples: {X.shape[0]}")
        print(f"   - Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple machine learning models"""
        print("TRAINING MACHINE LEARNING MODELS")
        print("=" * 50)
        
        # Split the data
        print("1. Splitting data into train/test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        print("2. Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        # Train and evaluate models
        print("3. Training and evaluating models...")
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for Logistic Regression, original for tree-based models
            if name == 'Logistic Regression':
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = dict(zip(X.columns, abs(model.coef_[0])))
            
            print(f"   {name} - Accuracy: {accuracy:.3f}, ROC-AUC: {roc_auc:.3f}")
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        print(f"\nBest Model: {best_model} (ROC-AUC: {self.results[best_model]['roc_auc']:.3f})")
        
        return self.results
    
    def evaluate_models(self):
        """Create comprehensive model evaluation visualizations"""
        print("CREATING MODEL EVALUATION VISUALIZATIONS")
        print("=" * 50)
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model comparison
        plt.subplot(3, 3, 1)
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width*2, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. ROC Curves
        plt.subplot(3, 3, 2)
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {results['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        plt.subplot(3, 3, 3)
        for name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, results['y_pred_proba'])
            plt.plot(recall, precision, label=name)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4-6. Confusion Matrices
        for i, (name, results) in enumerate(self.results.items()):
            plt.subplot(3, 3, 4 + i)
            cm = confusion_matrix(self.y_test, results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'])
            plt.title(f'{name} Confusion Matrix', fontsize=14, fontweight='bold')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
        
        # 7-9. Feature Importance
        for i, (name, importance) in enumerate(self.feature_importance.items()):
            plt.subplot(3, 3, 7 + i)
            # Get top 10 features
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, scores = zip(*top_features)
            
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
            plt.xlabel('Importance Score')
            plt.title(f'{name} Feature Importance', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Model evaluation visualizations saved as 'model_evaluation.png'")
    
    def generate_business_insights(self):
        """Generate business insights and recommendations"""
        print("GENERATING BUSINESS INSIGHTS")
        print("=" * 50)
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
        best_model = self.models[best_model_name]
        
        # Get feature importance
        if best_model_name in self.feature_importance:
            importance = self.feature_importance[best_model_name]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            top_features = []
        
        # Calculate churn probabilities for all customers
        if best_model_name == 'Logistic Regression':
            X_scaled = self.scaler.transform(self.X_train)
            churn_proba = best_model.predict_proba(X_scaled)[:, 1]
        else:
            churn_proba = best_model.predict_proba(self.X_train)[:, 1]
        
        # Create customer segments
        high_risk_threshold = np.percentile(churn_proba, 80)
        medium_risk_threshold = np.percentile(churn_proba, 60)
        
        segments = []
        for prob in churn_proba:
            if prob >= high_risk_threshold:
                segments.append('High Risk')
            elif prob >= medium_risk_threshold:
                segments.append('Medium Risk')
            else:
                segments.append('Low Risk')
        
        # Business insights
        insights = {
            'total_customers': len(self.data),
            'churn_rate': self.data['Churn'].mean() * 100,
            'high_risk_customers': segments.count('High Risk'),
            'medium_risk_customers': segments.count('Medium Risk'),
            'low_risk_customers': segments.count('Low Risk'),
            'top_churn_factors': top_features[:5],
            'best_model': best_model_name,
            'model_accuracy': self.results[best_model_name]['accuracy'],
            'model_roc_auc': self.results[best_model_name]['roc_auc']
        }
        
        # Print insights
        print(f"\nKEY BUSINESS INSIGHTS:")
        print(f"   - Total Customers: {insights['total_customers']:,}")
        print(f"   - Current Churn Rate: {insights['churn_rate']:.1f}%")
        print(f"   - High Risk Customers: {insights['high_risk_customers']:,} ({insights['high_risk_customers']/len(segments)*100:.1f}%)")
        print(f"   - Medium Risk Customers: {insights['medium_risk_customers']:,} ({insights['medium_risk_customers']/len(segments)*100:.1f}%)")
        print(f"   - Low Risk Customers: {insights['low_risk_customers']:,} ({insights['low_risk_customers']/len(segments)*100:.1f}%)")
        
        print(f"\nTOP CHURN RISK FACTORS:")
        for i, (feature, score) in enumerate(insights['top_churn_factors'], 1):
            print(f"   {i}. {feature.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"   - Best Model: {insights['best_model']}")
        print(f"   - Accuracy: {insights['model_accuracy']:.3f}")
        print(f"   - ROC-AUC: {insights['model_roc_auc']:.3f}")
        
        # Recommendations
        print(f"\nBUSINESS RECOMMENDATIONS:")
        print("   1. Focus retention efforts on High Risk customers (top 20%)")
        print("   2. Implement proactive outreach for customers with month-to-month contracts")
        print("   3. Encourage automatic payment methods to reduce churn risk")
        print("   4. Promote online security services to increase customer stickiness")
        print("   5. Monitor monthly charges and offer competitive pricing for high-value customers")
        print("   6. Create loyalty programs for long-tenure customers")
        print("   7. Develop mobile app features to increase engagement")
        print("   8. Implement regular customer satisfaction surveys")
        
        return insights
    
    def create_dashboard(self):
        """Create an interactive dashboard with Plotly"""
        print("CREATING INTERACTIVE DASHBOARD")
        print("=" * 50)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Churn Distribution', 'Tenure vs Churn', 'Monthly Charges vs Churn',
                          'Contract Type Impact', 'Payment Method Impact', 'Internet Service Impact',
                          'Risk Segmentation', 'Model Performance', 'Feature Importance'),
            specs=[[{"type": "pie"}, {"type": "scatter"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Churn distribution pie chart
        churn_counts = self.data['Churn'].value_counts()
        fig.add_trace(
            go.Pie(labels=['No Churn', 'Churn'], values=churn_counts.values,
                   marker_colors=['#2E8B57', '#DC143C']),
            row=1, col=1
        )
        
        # 2. Tenure vs Churn scatter
        fig.add_trace(
            go.Scatter(x=self.data['tenure'], y=self.data['Churn'],
                      mode='markers', name='Tenure vs Churn',
                      marker=dict(color=self.data['Churn'], colorscale='RdYlGn')),
            row=1, col=2
        )
        
        # 3. Monthly charges box plot
        for churn_val in [0, 1]:
            data_subset = self.data[self.data['Churn'] == churn_val]['MonthlyCharges']
            fig.add_trace(
                go.Box(y=data_subset, name=f'Churn: {churn_val}'),
                row=1, col=3
            )
        
        # 4. Contract type impact
        contract_churn = pd.crosstab(self.data['Contract'], self.data['Churn'], normalize='index') * 100
        fig.add_trace(
            go.Bar(x=contract_churn.index, y=contract_churn[1], name='Churn Rate %'),
            row=2, col=1
        )
        
        # 5. Payment method impact
        payment_churn = pd.crosstab(self.data['PaymentMethod'], self.data['Churn'], normalize='index') * 100
        fig.add_trace(
            go.Bar(x=payment_churn.index, y=payment_churn[1], name='Churn Rate %'),
            row=2, col=2
        )
        
        # 6. Internet service impact
        internet_churn = pd.crosstab(self.data['InternetService'], self.data['Churn'], normalize='index') * 100
        fig.add_trace(
            go.Bar(x=internet_churn.index, y=internet_churn[1], name='Churn Rate %'),
            row=2, col=3
        )
        
        # 7. Risk segmentation
        # Calculate risk segments (simplified)
        high_risk = len(self.data[self.data['Contract'] == 'Month-to-month'])
        medium_risk = len(self.data[self.data['PaymentMethod'] == 'Electronic check'])
        low_risk = len(self.data) - high_risk - medium_risk
        
        fig.add_trace(
            go.Bar(x=['Low Risk', 'Medium Risk', 'High Risk'], 
                   y=[low_risk, medium_risk, high_risk],
                   marker_color=['#2E8B57', '#FFA500', '#DC143C']),
            row=3, col=1
        )
        
        # 8. Model performance
        if self.results:
            models = list(self.results.keys())
            roc_aucs = [self.results[model]['roc_auc'] for model in models]
            fig.add_trace(
                go.Bar(x=models, y=roc_aucs, name='ROC-AUC Score'),
                row=3, col=2
            )
        
        # 9. Feature importance
        if self.feature_importance:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
            importance = self.feature_importance[best_model]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            features, scores = zip(*top_features)
            
            fig.add_trace(
                go.Bar(x=list(scores), y=[f.replace('_', ' ').title() for f in features],
                       orientation='h', name='Importance'),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Customer Churn Prediction Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        # Save dashboard
        fig.write_html("churn_dashboard.html")
        print("Interactive dashboard saved as 'churn_dashboard.html'")
        
        return fig
    
    def save_results(self):
        """Save all results and model artifacts"""
        print("SAVING RESULTS AND MODEL ARTIFACTS")
        print("=" * 50)
        
        # Save predictions
        if hasattr(self, 'X_test') and self.results:
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
            predictions_df = pd.DataFrame({
                'customerID': self.customer_ids.iloc[self.X_test.index],
                'actual_churn': self.y_test,
                'predicted_churn': self.results[best_model_name]['y_pred'],
                'churn_probability': self.results[best_model_name]['y_pred_proba']
            })
            
            predictions_df.to_csv('churn_predictions.csv', index=False)
            print("Predictions saved as 'churn_predictions.csv'")
        
        # Save model performance summary
        if self.results:
            performance_df = pd.DataFrame(self.results).T
            performance_df.to_csv('model_performance.csv')
            print("Model performance saved as 'model_performance.csv'")
        
        # Save feature importance
        if self.feature_importance:
            importance_df = pd.DataFrame(self.feature_importance).fillna(0)
            importance_df.to_csv('feature_importance.csv')
            print("Feature importance saved as 'feature_importance.csv'")
        
        print("All results saved successfully!")
    
    def generate_pdf_report(self):
        """Generate comprehensive PDF business report"""
        print("GENERATING PDF BUSINESS REPORT")
        print("=" * 50)
        
        try:
            from pdf_report_generator import ChurnReportGenerator
            
            # Create report generator
            report_gen = ChurnReportGenerator(
                data=self.data,
                results=self.results,
                feature_importance=self.feature_importance,
                models=self.models
            )
            
            # Generate PDF report
            report_file = report_gen.generate_pdf_report("churn_prediction_report.pdf")
            print(f"PDF report generated: {report_file}")
            
        except ImportError:
            print("PDF report generator not available. Install reportlab: pip install reportlab")
        except Exception as e:
            print(f"Error generating PDF report: {e}")

def main():
    """Main function to run the complete churn prediction system"""
    print("CUSTOMER CHURN PREDICTION SYSTEM")
    print("=" * 60)
    print("This system will help you predict customer churn using machine learning")
    print("and provide actionable business insights.\n")
    
    # Initialize the system
    churn_system = ChurnPredictionSystem()
    
    # Load data
    print("Loading customer data...")
    churn_system.load_data()
    
    # Explore data
    churn_system.explore_data()
    
    # Visualize data
    churn_system.visualize_data()
    
    # Preprocess data
    X, y = churn_system.preprocess_data()
    
    # Train models
    churn_system.train_models(X, y)
    
    # Evaluate models
    churn_system.evaluate_models()
    
    # Generate business insights
    insights = churn_system.generate_business_insights()
    
    # Create dashboard
    churn_system.create_dashboard()
    
    # Save results
    churn_system.save_results()
    
    # Generate PDF report
    churn_system.generate_pdf_report()
    
    print("\nCHURN PREDICTION SYSTEM COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Generated Files:")
    print("   - data_exploration.png - Data analysis visualizations")
    print("   - model_evaluation.png - Model performance charts")
    print("   - churn_dashboard.html - Interactive dashboard")
    print("   - churn_predictions.csv - Customer predictions")
    print("   - model_performance.csv - Model metrics")
    print("   - feature_importance.csv - Feature importance scores")
    print("   - churn_prediction_report.pdf - Business report")
    print("\nNext Steps:")
    print("   1. Review the generated visualizations and insights")
    print("   2. Open churn_dashboard.html in your browser for interactive analysis")
    print("   3. Read the PDF report for business recommendations")
    print("   4. Use the predictions to identify high-risk customers")
    print("   5. Implement retention strategies based on the insights")
    print("   6. Run the Streamlit app for interactive model demonstration")
    print("\nTo launch the interactive web app:")
    print("   streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
