# User Conversion Prediction - Digital Marketing Campaign

A machine learning project that predicts user conversion in digital marketing campaigns using Gradient Boosting classification. This project includes data preprocessing, model training, and a Streamlit web application for easy predictions.

## 📋 Project Overview

This machine learning pipeline analyzes digital marketing campaign data to predict whether users will convert or not. It provides both a Jupyter notebook for data exploration and model training, and a Streamlit web app for easy deployment and inference.

**Key Features:**
- Data preprocessing and feature engineering
- Gradient Boosting classifier model
- CSV upload and manual input prediction modes
- Real-time predictions with downloadable results
- Production-ready Streamlit deployment

## 🗂️ Project Structure

```
├── project22.ipynb                          # Main ML notebook with data exploration and model training
├── streamlit_app.py                         # Streamlit web application for predictions
├── digital_marketing_campaign_dataset.csv   # Original dataset
├── processed_dataset.csv                    # Preprocessed dataset
├── GB.pkl                                   # Trained Gradient Boosting model (serialized)
├── requirements.txt                         # Python dependencies (to be created)
└── README.md                                # This file
```

## 🛠️ Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git (optional, for version control)

## 📦 Installation

### 1. Clone or Navigate to the Project
```bash
cd "d:\university_projects\sem3(2)\ml_sadat"
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Or using conda
conda create -n conversion-prediction python=3.9
conda activate conversion-prediction
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning models and preprocessing
- `streamlit` - Web app framework
- `joblib` - Model serialization
- `matplotlib` - Visualization (for notebooks)
- `seaborn` - Statistical visualization

## 🚀 Deployment Guide

### Option 1: Local Streamlit Deployment

Run the Streamlit app locally:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: Cloud Deployment

#### **Streamlit Cloud (Recommended)**
1. Push your project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from your GitHub repository
4. Configure secrets if needed

#### **Heroku Deployment**
1. Create `Procfile`:
   ```
   web: streamlit run --server.port $PORT streamlit_app.py
   ```

2. Create `.streamlit/config.toml`:
   ```toml
   [server]
   port = $PORT
   enableCORS = false
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### **Docker Deployment**
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "streamlit_app.py"]
   ```

2. Build and run:
   ```bash
   docker build -t conversion-prediction .
   docker run -p 8501:8501 conversion-prediction
   ```

### Option 3: Azure/AWS Deployment
Contact your cloud provider's documentation for containerized app deployment using the Dockerfile above.

## 💻 Usage

### 1. **Using the Streamlit App**

**CSV Upload Mode:**
- Select "Upload CSV" option
- Upload a CSV file with the required features
- Click "Predict Conversion"
- Download predictions as CSV

**Manual Input Mode:**
- Select "Manual Input" option
- Enter feature values using the form
- Get instant conversion prediction

### 2. **Using the Jupyter Notebook**

```bash
jupyter notebook project22.ipynb
```

Explore the full ML pipeline including:
- Data loading and exploration
- Feature preprocessing
- Feature selection
- Model training and evaluation
- Hyperparameter tuning

## 📊 Model Details

**Algorithm:** Histogram-based Gradient Boosting Classifier
- Fast and efficient on large datasets
- Handles missing values automatically
- High predictive accuracy

**Model File:** `GB.pkl`
- Pre-trained and ready for deployment
- Includes full preprocessing pipeline
- Loaded via joblib in the Streamlit app

## 📥 Input Features

The model expects the following features (exact names may vary based on preprocessing):
- Digital marketing campaign metrics
- User engagement indicators
- Campaign characteristics

Refer to `digital_marketing_campaign_dataset.csv` for the exact feature structure.

## 🔍 Data Preprocessing Pipeline

The model includes:
- Label encoding for categorical variables
- Feature scaling/standardization
- Feature selection (top percentiles)
- Missing value handling

All preprocessing is automatically applied to new predictions.

## 📈 Model Performance

Evaluate model performance in the Jupyter notebook:
- Classification metrics: Accuracy, Precision, Recall, F1-Score
- Cross-validation scores
- Feature importance analysis

## 🐛 Troubleshooting

**Issue:** Model file `GB.pkl` not found
- **Solution:** Ensure `GB.pkl` is in the same directory as `streamlit_app.py`

**Issue:** Missing dependencies
- **Solution:** Run `pip install -r requirements.txt` again

**Issue:** Port 8501 already in use
- **Solution:** Run `streamlit run streamlit_app.py --server.port 8502`

**Issue:** CSV upload fails
- **Solution:** Ensure CSV has the same features as the training data

## 📝 Configuration

Edit `.streamlit/config.toml` for customization:
```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"

[server]
maxUploadSize = 200
```

## 🔐 Security Considerations

- Keep API keys in environment variables (if using cloud services)
- Use `.streamlit/secrets.toml` for sensitive data
- Never commit sensitive information to Git
- Validate input data on the server side

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

## 👤 Author

Created as part of ML coursework (Sem 3)

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the Jupyter notebook for implementation details
3. Ensure all dependencies are installed correctly

---

**Last Updated:** November 2025
**Status:** Ready for Deployment ✅
