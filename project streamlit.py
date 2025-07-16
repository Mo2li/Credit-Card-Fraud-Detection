import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(
    page_title="Banking Fraud Detection System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        border-radius: 10px;
        padding: 15px;
        background-color: #f8f9fa;
        margin-bottom: 15px;
    }
    .fraud-alert {
        background-color: #fff0f0;
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
    }
    .legit-card {
        background-color: #f0fff0;
        border-left: 5px solid #00d154;
        padding: 15px;
        border-radius: 5px;
    }
    .feature-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #f9f9f9;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🏦 Banking Transaction Fraud Detection")
st.markdown("---")

# ====================== DATA LOADING & PROCESSING ======================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"D:\AI & data\data sets\bs140513_032310.csv")
        
        # Clean data
        df = df.replace("'", "", regex=True)
        for col in ['customer', 'zipcodeOri', 'merchant', 'zipMerchant', 'category']:
            df[col] = df[col].str.replace("'", "").str.strip()
        
        # Create required features
        df['day_of_week'] = df['step'] % 7
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['customer_txn_count'] = df.groupby('customer').cumcount() + 1
        
        # Amount binning
        max_amount = df['amount'].max()
        if max_amount <= 10000:
            max_amount = 10001  
        bins = [0, 100, 1000, 10000, max_amount]
        labels = ['low', 'medium', 'high', 'very_high']
        df['amount_bin'] = pd.cut(df['amount'], bins=bins, labels=labels, include_lowest=True)
        df['amount_bin'] = df['amount_bin'].map({'low': 0, 'medium': 1, 'high': 2, 'very_high': 3})
        
        # Convert categorical features
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        
        # Handle gender
        df['gender'] = df['gender'].map({'M': 0, 'F': 1, 'E': np.nan, 'U': np.nan})
        df['gender'] = df['gender'].fillna(df['gender'].mode()[0]).astype(int)
        
        # Handle age
        age_map = {1: 20, 2: 30, 3: 40, 4: 50, 5: 60}
        df['age'] = df['age'].astype(str).str.replace("'", "").str.strip()
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].map(age_map)
        df['age'] = df['age'].fillna(df['age'].mean()).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ====================== DATA ANALYSIS ======================
def show_data_analysis(df):
    st.header("🔍 Transaction Data Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        st.metric("Fraudulent Transactions", df['fraud'].sum())
    with col3:
        st.metric("Fraud Rate", f"{df['fraud'].mean()*100:.2f}%")
    
    # New visualizations
    st.subheader("Fraud Distribution by Category (Bar Plot)")
    fraud_by_cat = df.groupby('category')['fraud'].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fraud_by_cat.plot(kind='bar', color='orange', ax=ax1)
    st.pyplot(fig1)
    
    st.subheader("Transaction Amount Distribution (KDE Plot)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='amount', hue='fraud', fill=True, palette=['green', 'red'])
    st.pyplot(fig2)
    
    st.subheader("Fraud vs Legitimate Transactions (Pie Chart)")
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    df['fraud'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Legit', 'Fraud'], colors=['green', 'red'])
    st.pyplot(fig3)
    
    # Original visualizations remain unchanged
    st.subheader("Fraud Distribution by Category")
    fraud_by_cat = df.groupby('category')['fraud'].mean().sort_values(ascending=False)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=fraud_by_cat.values, y=fraud_by_cat.index, palette='viridis')
    st.pyplot(fig4)
    
    st.subheader("Transaction Amount Distribution")
    fig5, ax5 = plt.subplots()
    sns.boxplot(data=df, x='fraud', y='amount', palette=['#00D154', '#FF4B4B'])
    st.pyplot(fig5)

# ====================== MODEL TRAINING ======================
@st.cache_resource
def train_model(df):
    try:
        # Prepare features
        features = [
            'amount', 
            'is_weekend', 
            'gender', 
            'age',
            'customer_txn_count',
            'day_of_week',
            'category_encoded',
            'amount_bin'
        ]
        
        X = df[features]
        y = df['fraud']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Model training
        model = RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            random_state=42,
            max_depth=12,
            min_samples_split=10,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Save model for later use
        joblib.dump(model, "fraud_model.pkl")
        
        return model, X_test, y_test, features
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None, None

# ====================== MODEL EVALUATION ======================
def evaluate_model(model, X_test, y_test, features):
    st.header("📊 Model Performance Evaluation")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, color='#d4f1d4'))
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Predicted Legit', 'Predicted Fraud'],
               yticklabels=['Actual Legit', 'Actual Fraud'],
               annot_kws={"size": 14})
    ax.set_title("Confusion Matrix (Counts)", pad=20)
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='#4E79A7', label=f'ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='Blues_r')
    st.pyplot(fig)

# ====================== REAL-TIME DETECTION ======================
def real_time_detection(model, features, categories):
    st.header("🔎 Transaction Fraud Check")
    
    with st.form("transaction_form"):
        st.subheader("Enter Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount", min_value=0.0, value=100.0)
            is_weekend = st.selectbox("Is Weekend?", [False, True])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 20, 60, 30, step=10)
        
        with col2:
            customer_txn_count = st.number_input("Customer Transaction Count", min_value=1, value=1)
            day_of_week = st.selectbox("Day of Week", range(7), 
                                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
            category = st.selectbox("Category", categories)
            amount_bin = st.selectbox("Amount Bin", [0, 1, 2, 3], 
                                    format_func=lambda x: ["Low","Medium","High","Very High"][x])
        
        submitted = st.form_submit_button("Check Fraud Risk")
    
    if submitted:
        try:
            # Prepare input
            input_data = {
                'amount': amount,
                'is_weekend': int(is_weekend),
                'gender': 0 if gender == "Male" else 1,
                'age': age,
                'customer_txn_count': customer_txn_count,
                'day_of_week': day_of_week,
                'category_encoded': categories.index(category),
                'amount_bin': amount_bin
            }
            
            input_df = pd.DataFrame([input_data])[features]
            
            # Make prediction
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)[0]
            
            # Display results
            if prediction[0] == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h3>⚠️ High Fraud Risk Detected</h3>
                    <p>Fraud Probability: {proba[1]*100:.1f}%</p>
                    <p>This transaction matches known fraud patterns.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-card">
                    <h3>✅ Low Fraud Risk</h3>
                    <p>Fraud Probability: {proba[1]*100:.1f}%</p>
                    <p>No significant fraud indicators detected.</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ====================== MAIN APP ======================
def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Get unique categories for dropdown
    categories = sorted(df['category'].unique())
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Analysis", "Model Training", "Fraud Detection"])
    
    if page == "Data Analysis":
        show_data_analysis(df)
    elif page == "Model Training":
        st.header("🤖 Fraud Detection Model Training")
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model, X_test, y_test, features = train_model(df)
                if model is not None:
                    st.success("Model trained successfully!")
                    evaluate_model(model, X_test, y_test, features)
    elif page == "Fraud Detection":
        st.header("🔍 Real-time Fraud Detection")
        try:
            model = joblib.load("fraud_model.pkl")
            features = ['amount', 'is_weekend', 'gender', 'age',
                       'customer_txn_count', 'day_of_week', 'category_encoded', 'amount_bin']
            real_time_detection(model, features, categories)
        except Exception as e:
            st.error(f"Please train the model first: {str(e)}")

if __name__ == "__main__":
    main()