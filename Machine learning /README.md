
# 🏡 Machine Learning Scripts - SCT Internship

> 🚀 My **first internship ML project** — predicting California house prices using **20,640 real housing records** and customer segmentation analysis!

---

## ✨ What's Inside?

This directory contains two machine learning scripts that demonstrate different ML techniques:

### 1. 🏠 SCT_ML_1.py - California Housing Price Prediction
**Advanced House Price Prediction System using California Housing Dataset**

- **Dataset**: California Housing Dataset (20,640 real housing districts)
- **Algorithm**: Linear Regression with feature engineering
- **Features**: 13 engineered features including income, location, property characteristics
- **Output**: House price predictions with interpretable coefficients

**Key Features:**
- Comprehensive data exploration and visualization
- Smart feature engineering (distance to SF, coastal proximity, etc.)
- Realistic test scenarios for different California regions
- Performance metrics and coefficient analysis

### 2. 👥 SCT_ML_2.py - Customer Segmentation using K-Means
**Customer Segmentation Analysis with 3D Visualization**

- **Dataset**: Mall Customers Dataset (200 customers)
- **Algorithm**: K-Means Clustering
- **Features**: Age, Annual Income, Spending Score
- **Output**: Customer segments with 2D and 3D visualizations

**Key Features:**
- Elbow method for optimal cluster selection
- 2D and 3D scatter plots
- Cluster profile analysis
- Interactive visualizations

---

## 📊 Cool Insights

### California Housing Analysis:
💰 **Income** strongly influences house prices  
🌊 Houses near the **coast** are way more expensive  
⏳ **House age** shows an interesting non-linear trend  

### Customer Segmentation Results:
- **Optimal Clusters**: 5 customer segments
- **Segments Identified**: Different customer profiles based on age, income, and spending patterns

---

## 🛠️ Tech Stack
🐍 Python  
📊 Pandas · NumPy · Matplotlib · Seaborn  
🤖 Scikit-learn  

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation Steps

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Install required packages:**
   ```bash
   pip3 install --break-system-packages -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip3 install --break-system-packages pandas numpy scikit-learn matplotlib seaborn
   ```

---

## 🎯 Running the Scripts

### California Housing Price Prediction
```bash
cd "Machine learning /SCT_ML_1"
source .venv/bin/activate
python3 SCT_ML_1.py
```

**Expected Output:**
- Dataset loading and exploration
- Market analysis and visualizations
- Linear regression model training
- Feature importance analysis
- Realistic price predictions for different California regions

### Customer Segmentation
```bash
cd "Machine learning /SCT_ML_2"
source .venv/bin/activate
python3 SCT_ML_2.py
```

**Expected Output:**
- Elbow method plot for optimal clusters
- 2D scatter plot of customer segments
- 3D visualization of segments
- Cluster profile statistics

---

## 📁 Project Structure

```
Machine learning/
├── SCT_ML_1/
│   ├── SCT_ML_1.py          # California housing price prediction script
│   ├── requirements.txt     # Python dependencies
│   └── README.md           # Task 1 documentation
├── SCT_ML_2/
│   ├── SCT_ML_2.py         # Customer segmentation script
│   ├── Mall_Customers.csv  # Customer dataset (200 records)
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Task 2 documentation
├── requirements.txt        # Main dependencies file
└── README.md              # This main documentation file
```

---

## 🎯 Results Summary

### California Housing Model Performance:
- **R² Score**: 0.6519 (65.2% variance explained)
- **Average Prediction Error**: $67,536
- **Key Drivers**: Income levels, geographic location, property characteristics

---

## 🔧 Issues Solved

1. **Missing Dependencies**: Installed all required Python packages (pandas, numpy, scikit-learn, matplotlib, seaborn)
2. **Missing Dataset**: Created `Mall_Customers.csv` with 200 customer records
3. **Data Type Error**: Fixed the groupby operation to only calculate means for numeric columns
4. **Virtual Environment**: Properly configured and activated the virtual environment

---

## 🚧 Next Steps
🔹 Try advanced models (Random Forest, XGBoost)  
🔹 Tune hyperparameters  
🔹 Deploy with Streamlit/Flask  
🔹 Add model persistence and API endpoints  
🔹 Create interactive dashboards  

---

## 🙌 My Takeaway
This project taught me the **core workflow of ML**:  
📥 Get data → 🔍 Explore → ⚙️ Engineer → 🤖 Train → 📉 Evaluate → 🎨 Visualize  

---

## 🆘 Troubleshooting

If you encounter issues:

1. **ModuleNotFoundError**: Make sure the virtual environment is activated and packages are installed
2. **FileNotFoundError**: Ensure `Mall_Customers.csv` is in the same directory as the script
3. **Display Issues**: The scripts generate matplotlib plots - ensure you have a display environment

---

🔥 First step into my Machine Learning journey — more creative projects coming soon!  

---

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/) 
[![Machine Learning](https://img.shields.io/badge/Topic-Machine%20Learning-green)]()
