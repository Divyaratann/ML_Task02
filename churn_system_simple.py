"""
Customer Churn Prediction System - Simplified Version
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
    """A comprehensive churn prediction system"""
    
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
        """Load customer churn dataset"""
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
        ordinal_columns = ['tenure_group', 'monthly_charges_group']
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
    
    # Preprocess data
    X, y = churn_system.preprocess_data()
    
    # Train models
    churn_system.train_models(X, y)
    
    # Generate business insights
    insights = churn_system.generate_business_insights()
    
    # Save results
    churn_system.save_results()
    
    print("\nCHURN PREDICTION SYSTEM COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Generated Files:")
    print("   - churn_predictions.csv - Customer predictions")
    print("   - model_performance.csv - Model metrics")
    print("   - feature_importance.csv - Feature importance scores")
    print("\nNext Steps:")
    print("   1. Review the generated insights and predictions")
    print("   2. Use the predictions to identify high-risk customers")
    print("   3. Implement retention strategies based on the insights")
    print("   4. Run the Streamlit app for interactive model demonstration")
    print("\nTo launch the interactive web app:")
    print("   streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()


