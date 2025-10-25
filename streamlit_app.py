"""
Streamlit Web Application for Customer Churn Prediction System
Interactive dashboard for model demonstration and customer analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io
import base64

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitChurnApp:
    """Streamlit application for churn prediction system"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    def load_sample_data(self):
        """Load or create sample data"""
        if self.data is None:
            # Create sample data (same as in main system)
            np.random.seed(42)
            n_customers = 5000
            
            data = {
                'customerID': [f'CUST_{i:06d}' for i in range(1, n_customers + 1)],
                'gender': np.random.choice(['Male', 'Female'], n_customers),
                'SeniorCitizen': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
                'Partner': np.random.choice(['Yes', 'No'], n_customers, p=[0.5, 0.5]),
                'Dependents': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
                'tenure': np.random.randint(1, 73, n_customers),
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
            
            # Create realistic churn
            churn_prob = []
            for i in range(n_customers):
                prob = 0.1
                if data['Contract'][i] == 'Month-to-month':
                    prob += 0.3
                if data['PaymentMethod'][i] == 'Electronic check':
                    prob += 0.2
                if data['MonthlyCharges'][i] > 80:
                    prob += 0.15
                if data['tenure'][i] < 12:
                    prob += 0.2
                if data['OnlineSecurity'][i] == 'No':
                    prob += 0.1
                churn_prob.append(min(prob, 0.9))
            
            data['Churn'] = np.random.binomial(1, churn_prob)
            self.data = pd.DataFrame(data)
            
            # Clean TotalCharges
            self.data.loc[self.data['tenure'] == 0, 'TotalCharges'] = 0
            self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
            self.data['TotalCharges'].fillna(0, inplace=True)
        
        return self.data
    
    def preprocess_data(self, df):
        """Preprocess data for machine learning"""
        # Feature engineering
        df['tenure_group'] = pd.cut(df['tenure'], 
                                   bins=[0, 12, 24, 48, 72], 
                                   labels=['New', 'Regular', 'Loyal', 'VIP'])
        
        df['monthly_charges_group'] = pd.cut(df['MonthlyCharges'], 
                                           bins=[0, 35, 70, 100, 200], 
                                           labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Service count
        service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        df['service_count'] = 0
        for col in service_columns:
            if col in df.columns:
                df['service_count'] += (df[col] == 'Yes').astype(int)
        
        # Risk score
        df['risk_score'] = 0
        df['risk_score'] += (df['Contract'] == 'Month-to-month').astype(int) * 3
        df['risk_score'] += (df['PaymentMethod'] == 'Electronic check').astype(int) * 2
        df['risk_score'] += (df['tenure'] < 12).astype(int) * 2
        df['risk_score'] += (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.8)).astype(int) * 1
        df['risk_score'] += (df['OnlineSecurity'] == 'No').astype(int) * 1
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        categorical_columns.remove('customerID')
        
        ordinal_columns = ['tenure_group', 'monthly_charges_group']
        nominal_columns = [col for col in categorical_columns if col not in ordinal_columns]
        
        # Label encode ordinal variables
        for col in ordinal_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        
        # One-hot encode nominal variables
        df_encoded = pd.get_dummies(df, columns=nominal_columns, drop_first=True)
        
        # Prepare features and target
        feature_columns = [col for col in df_encoded.columns 
                          if col not in ['customerID', 'Churn']]
        
        X = df_encoded[feature_columns]
        y = df_encoded['Churn']
        
        return X, y, df_encoded
    
    def train_models(self, X, y):
        """Train machine learning models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        # Train models
        for name, model in models.items():
            if name == 'Logistic Regression':
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            model.fit(X_train_use, y_train)
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_test': y_test
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = dict(zip(X.columns, abs(model.coef_[0])))
        
        return X_test, y_test

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚨 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    
    # Initialize app
    app = StreamlitChurnApp()
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Load data option
    st.sidebar.subheader("📊 Data Source")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload CSV File"]
    )
    
    if data_option == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload customer data CSV",
            type=['csv'],
            help="Upload a CSV file with customer data"
        )
        if uploaded_file is not None:
            try:
                app.data = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ Data uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {e}")
                st.sidebar.info("Using sample data instead...")
                app.load_sample_data()
        else:
            app.load_sample_data()
    else:
        app.load_sample_data()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "📈 Predictions", 
        "💼 Business Insights", 
        "🎯 Customer Analysis"
    ])
    
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(app.data):,}")
        
        with col2:
            churn_rate = app.data['Churn'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        with col3:
            avg_tenure = app.data['tenure'].mean()
            st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
        
        with col4:
            avg_charges = app.data['MonthlyCharges'].mean()
            st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(app.data.head(10))
        
        # Data distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Churn Distribution")
            churn_counts = app.data['Churn'].value_counts()
            fig = px.pie(values=churn_counts.values, 
                        names=['No Churn', 'Churn'],
                        color_discrete_sequence=['#2E8B57', '#DC143C'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📅 Tenure Distribution")
            fig = px.histogram(app.data, x='tenure', nbins=20, 
                             title="Customer Tenure Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("🔍 Key Data Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.write("**Top Churn Factors:**")
            contract_churn = pd.crosstab(app.data['Contract'], app.data['Churn'], normalize='index') * 100
            st.write(f"• Month-to-month contracts: {contract_churn.loc['Month-to-month', 1]:.1f}% churn rate")
            
            payment_churn = pd.crosstab(app.data['PaymentMethod'], app.data['Churn'], normalize='index') * 100
            st.write(f"• Electronic check payment: {payment_churn.loc['Electronic check', 1]:.1f}% churn rate")
        
        with insights_col2:
            st.write("**Customer Segments:**")
            st.write(f"• New customers (< 12 months): {len(app.data[app.data['tenure'] < 12]):,}")
            st.write(f"• High-value customers (> $80/month): {len(app.data[app.data['MonthlyCharges'] > 80]):,}")
            st.write(f"• Long-term customers (> 2 years): {len(app.data[app.data['tenure'] > 24]):,}")
    
    with tab2:
        st.header("🤖 Model Training & Evaluation")
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a few moments."):
                # Preprocess data
                X, y, df_encoded = app.preprocess_data(app.data)
                
                # Train models
                X_test, y_test = app.train_models(X, y)
                
                st.success("✅ Models trained successfully!")
        
        if app.results:
            st.subheader("📊 Model Performance Comparison")
            
            # Performance metrics
            performance_data = []
            for model_name, metrics in app.results.items():
                performance_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'ROC-AUC': metrics['roc_auc']
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            # Best model
            best_model = max(app.results.keys(), key=lambda x: app.results[x]['roc_auc'])
            st.success(f"🏆 Best Model: **{best_model}** (ROC-AUC: {app.results[best_model]['roc_auc']:.3f})")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 ROC Curves")
                fig = go.Figure()
                
                for name, results in app.results.items():
                    fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, 
                                           name=f"{name} (AUC = {results['roc_auc']:.3f})",
                                           line=dict(width=2)))
                
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                       name='Random', 
                                       line=dict(dash='dash', color='gray')))
                
                fig.update_layout(
                    title="ROC Curves Comparison",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    width=500,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Feature Importance")
                if app.feature_importance:
                    best_model_importance = app.feature_importance[best_model]
                    top_features = sorted(best_model_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
                    
                    features, scores = zip(*top_features)
                    fig = px.bar(x=list(scores), y=[f.replace('_', ' ').title() for f in features],
                               orientation='h', title=f"Top Features - {best_model}")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📈 Customer Churn Predictions")
        
        if not app.results:
            st.warning("⚠️ Please train models first in the 'Model Training' tab.")
        else:
            # Get best model
            best_model_name = max(app.results.keys(), key=lambda x: app.results[x]['roc_auc'])
            best_model = app.models[best_model_name]
            
            # Predictions for all customers
            X, y, df_encoded = app.preprocess_data(app.data)
            
            if best_model_name == 'Logistic Regression':
                X_scaled = app.scaler.transform(X)
                churn_proba = best_model.predict_proba(X_scaled)[:, 1]
            else:
                churn_proba = best_model.predict_proba(X)[:, 1]
            
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'Customer ID': app.data['customerID'],
                'Actual Churn': app.data['Churn'],
                'Churn Probability': churn_proba,
                'Risk Level': pd.cut(churn_proba, bins=[0, 0.3, 0.7, 1], 
                                   labels=['Low', 'Medium', 'High'])
            })
            
            # Risk segmentation
            st.subheader("🎯 Risk Segmentation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                low_risk = len(predictions_df[predictions_df['Risk Level'] == 'Low'])
                st.metric("Low Risk", f"{low_risk:,}", 
                         f"{low_risk/len(predictions_df)*100:.1f}%")
            
            with col2:
                medium_risk = len(predictions_df[predictions_df['Risk Level'] == 'Medium'])
                st.metric("Medium Risk", f"{medium_risk:,}", 
                         f"{medium_risk/len(predictions_df)*100:.1f}%")
            
            with col3:
                high_risk = len(predictions_df[predictions_df['Risk Level'] == 'High'])
                st.metric("High Risk", f"{high_risk:,}", 
                         f"{high_risk/len(predictions_df)*100:.1f}%")
            
            # High-risk customers
            st.subheader("🚨 High-Risk Customers")
            high_risk_customers = predictions_df[predictions_df['Risk Level'] == 'High'].sort_values(
                'Churn Probability', ascending=False)
            
            st.dataframe(high_risk_customers.head(20), use_container_width=True)
            
            # Download predictions
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
            
            # Risk distribution
            st.subheader("📊 Risk Distribution")
            fig = px.pie(predictions_df, names='Risk Level', 
                        title="Customer Risk Distribution",
                        color_discrete_sequence=['#2E8B57', '#FFA500', '#DC143C'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("💼 Business Insights & Recommendations")
        
        if not app.results:
            st.warning("⚠️ Please train models first in the 'Model Training' tab.")
        else:
            # Key metrics
            st.subheader("📊 Key Business Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_customers = len(app.data)
                st.metric("Total Customers", f"{total_customers:,}")
            
            with col2:
                churn_rate = app.data['Churn'].mean() * 100
                st.metric("Current Churn Rate", f"{churn_rate:.1f}%")
            
            with col3:
                # Calculate potential revenue loss
                avg_monthly_revenue = app.data['MonthlyCharges'].mean()
                churned_customers = app.data['Churn'].sum()
                monthly_revenue_loss = churned_customers * avg_monthly_revenue
                st.metric("Monthly Revenue Loss", f"${monthly_revenue_loss:,.0f}")
            
            with col4:
                # High-risk customers
                X, y, df_encoded = app.preprocess_data(app.data)
                best_model_name = max(app.results.keys(), key=lambda x: app.results[x]['roc_auc'])
                best_model = app.models[best_model_name]
                
                if best_model_name == 'Logistic Regression':
                    X_scaled = app.scaler.transform(X)
                    churn_proba = best_model.predict_proba(X_scaled)[:, 1]
                else:
                    churn_proba = best_model.predict_proba(X)[:, 1]
                
                high_risk_count = sum(churn_proba > 0.7)
                st.metric("High-Risk Customers", f"{high_risk_count:,}")
            
            # Top churn factors
            st.subheader("🎯 Top Churn Risk Factors")
            
            if app.feature_importance:
                best_model_name = max(app.results.keys(), key=lambda x: app.results[x]['roc_auc'])
                importance = app.feature_importance[best_model_name]
                top_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for i, (factor, score) in enumerate(top_factors, 1):
                    st.write(f"{i}. **{factor.replace('_', ' ').title()}**: {score:.3f}")
            
            # Business recommendations
            st.subheader("💡 Business Recommendations")
            
            recommendations = [
                "🎯 **Focus on High-Risk Customers**: Implement targeted retention campaigns for customers with >70% churn probability",
                "📞 **Proactive Outreach**: Contact customers with month-to-month contracts before renewal dates",
                "💳 **Payment Method Optimization**: Encourage automatic payment methods to reduce churn risk",
                "🛡️ **Security Services**: Promote online security and backup services to increase customer stickiness",
                "💰 **Pricing Strategy**: Review pricing for high-value customers to prevent price-related churn",
                "🎁 **Loyalty Programs**: Create retention incentives for long-tenure customers",
                "📱 **Digital Engagement**: Develop mobile app features to increase customer engagement",
                "📊 **Regular Monitoring**: Implement monthly churn risk assessments and early warning systems"
            ]
            
            for rec in recommendations:
                st.write(rec)
            
            # Action plan
            st.subheader("📋 Immediate Action Plan")
            
            action_plan = {
                "Week 1": [
                    "Identify top 100 high-risk customers",
                    "Set up automated churn risk monitoring",
                    "Prepare retention campaign materials"
                ],
                "Week 2": [
                    "Launch targeted outreach to high-risk customers",
                    "Implement payment method migration incentives",
                    "Begin security service promotion campaign"
                ],
                "Week 3": [
                    "Analyze campaign effectiveness",
                    "Adjust retention strategies based on response",
                    "Plan long-term customer engagement initiatives"
                ],
                "Week 4": [
                    "Review monthly churn metrics",
                    "Update risk models with new data",
                    "Plan next month's retention activities"
                ]
            }
            
            for week, actions in action_plan.items():
                with st.expander(f"📅 {week}"):
                    for action in actions:
                        st.write(f"• {action}")
    
    with tab5:
        st.header("🎯 Individual Customer Analysis")
        
        if not app.results:
            st.warning("⚠️ Please train models first in the 'Model Training' tab.")
        else:
            # Customer search
            st.subheader("🔍 Customer Lookup")
            
            customer_id = st.selectbox(
                "Select Customer ID:",
                options=app.data['customerID'].tolist(),
                help="Choose a customer to analyze their churn risk"
            )
            
            if customer_id:
                # Get customer data
                customer_data = app.data[app.data['customerID'] == customer_id].iloc[0]
                
                # Calculate churn probability
                X, y, df_encoded = app.preprocess_data(app.data)
                customer_idx = app.data[app.data['customerID'] == customer_id].index[0]
                customer_features = X.iloc[customer_idx:customer_idx+1]
                
                best_model_name = max(app.results.keys(), key=lambda x: app.results[x]['roc_auc'])
                best_model = app.models[best_model_name]
                
                if best_model_name == 'Logistic Regression':
                    customer_features_scaled = app.scaler.transform(customer_features)
                    churn_prob = best_model.predict_proba(customer_features_scaled)[0, 1]
                else:
                    churn_prob = best_model.predict_proba(customer_features)[0, 1]
                
                # Display customer information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("👤 Customer Profile")
                    st.write(f"**Customer ID**: {customer_data['customerID']}")
                    st.write(f"**Gender**: {customer_data['gender']}")
                    st.write(f"**Senior Citizen**: {'Yes' if customer_data['SeniorCitizen'] else 'No'}")
                    st.write(f"**Partner**: {customer_data['Partner']}")
                    st.write(f"**Dependents**: {customer_data['Dependents']}")
                    st.write(f"**Tenure**: {customer_data['tenure']} months")
                    st.write(f"**Contract**: {customer_data['Contract']}")
                    st.write(f"**Payment Method**: {customer_data['PaymentMethod']}")
                
                with col2:
                    st.subheader("💰 Financial Information")
                    st.write(f"**Monthly Charges**: ${customer_data['MonthlyCharges']:.2f}")
                    st.write(f"**Total Charges**: ${customer_data['TotalCharges']:.2f}")
                    st.write(f"**Paperless Billing**: {customer_data['PaperlessBilling']}")
                    
                    # Services
                    st.subheader("📱 Services")
                    services = ['PhoneService', 'MultipleLines', 'InternetService', 
                              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                              'TechSupport', 'StreamingTV', 'StreamingMovies']
                    
                    for service in services:
                        if service in customer_data:
                            st.write(f"**{service.replace('_', ' ').title()}**: {customer_data[service]}")
                
                # Risk assessment
                st.subheader("🎯 Churn Risk Assessment")
                
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                
                with risk_col1:
                    if churn_prob < 0.3:
                        risk_level = "Low"
                        risk_color = "green"
                    elif churn_prob < 0.7:
                        risk_level = "Medium"
                        risk_color = "orange"
                    else:
                        risk_level = "High"
                        risk_color = "red"
                    
                    st.metric("Risk Level", risk_level)
                
                with risk_col2:
                    st.metric("Churn Probability", f"{churn_prob:.1%}")
                
                with risk_col3:
                    actual_churn = "Yes" if customer_data['Churn'] else "No"
                    st.metric("Actual Churn", actual_churn)
                
                # Risk factors
                st.subheader("⚠️ Risk Factors")
                
                risk_factors = []
                if customer_data['Contract'] == 'Month-to-month':
                    risk_factors.append("Month-to-month contract")
                if customer_data['PaymentMethod'] == 'Electronic check':
                    risk_factors.append("Electronic check payment")
                if customer_data['tenure'] < 12:
                    risk_factors.append("Short tenure (< 12 months)")
                if customer_data['MonthlyCharges'] > 80:
                    risk_factors.append("High monthly charges")
                if customer_data['OnlineSecurity'] == 'No':
                    risk_factors.append("No online security")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.write("No major risk factors identified")
                
                # Recommendations
                st.subheader("💡 Personalized Recommendations")
                
                if churn_prob > 0.7:
                    st.error("🚨 **High Risk Customer** - Immediate action required!")
                    st.write("• Schedule immediate retention call")
                    st.write("• Offer special discount or promotion")
                    st.write("• Consider contract upgrade incentives")
                elif churn_prob > 0.3:
                    st.warning("⚠️ **Medium Risk Customer** - Monitor closely")
                    st.write("• Regular check-in calls")
                    st.write("• Proactive service recommendations")
                    st.write("• Loyalty program enrollment")
                else:
                    st.success("✅ **Low Risk Customer** - Maintain relationship")
                    st.write("• Continue current service level")
                    st.write("• Upsell additional services")
                    st.write("• Referral program participation")

if __name__ == "__main__":
    main()

