#!/usr/bin/env python3
"""
Simple test script to check Streamlit app functionality
"""

def test_imports():
    """Test if all required imports work"""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ Seaborn imported successfully")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import xgboost as xgb
        print("✅ XGBoost imported successfully")
    except ImportError as e:
        print(f"❌ XGBoost import failed: {e}")
        return False
    
    return True

def test_streamlit_app():
    """Test if the Streamlit app can be imported"""
    try:
        # Import the main app class
        import sys
        sys.path.append('.')
        from streamlit_app import StreamlitChurnApp
        
        # Create an instance
        app = StreamlitChurnApp()
        print("✅ StreamlitChurnApp class created successfully")
        
        # Test data loading
        data = app.load_sample_data()
        print(f"✅ Sample data loaded successfully: {len(data)} customers")
        
        return True
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Streamlit App Dependencies...")
    print("=" * 50)
    
    imports_ok = test_imports()
    print("\n" + "=" * 50)
    
    if imports_ok:
        print("🔍 Testing Streamlit App Functionality...")
        print("=" * 50)
        app_ok = test_streamlit_app()
        
        if app_ok:
            print("\n🎉 All tests passed! Streamlit app should work.")
        else:
            print("\n❌ App functionality test failed.")
    else:
        print("\n❌ Import tests failed. Please install missing dependencies.")

