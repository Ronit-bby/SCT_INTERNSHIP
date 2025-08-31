# 🏠 California Housing Price Prediction

> 🎯 **Task 1**: Advanced house price prediction system using California Housing Dataset with 20,640 real housing records

---

## 📊 Project Overview

This project implements a comprehensive house price prediction system using the California Housing Dataset. It demonstrates advanced machine learning techniques including feature engineering, linear regression modeling, and detailed market analysis to predict housing prices across California.

---

## 🎯 What This Project Does

### **Advanced House Price Prediction System**
- **Dataset**: California Housing Dataset (20,640 real housing districts)
- **Algorithm**: Linear Regression with feature engineering
- **Features**: 13 engineered features including income, location, property characteristics
- **Output**: House price predictions with interpretable coefficients

### **Key Features:**
- ✅ **Comprehensive Data Exploration**: Market analysis and visualizations
- ✅ **Smart Feature Engineering**: Distance to SF, coastal proximity, income brackets
- ✅ **Realistic Test Scenarios**: Predictions for different California regions
- ✅ **Performance Metrics**: R², MAE, RMSE analysis
- ✅ **Coefficient Analysis**: Understanding what drives house prices

---

## 📁 Files Description

- `SCT_ML_1.py` - Main California housing price prediction script
- `requirements.txt` - Python dependencies
- `README.md` - This documentation file

---

## 🛠️ Tech Stack

🐍 **Python**  
📊 **Pandas** · **NumPy** · **Matplotlib** · **Seaborn**  
🤖 **Scikit-learn** (Linear Regression, StandardScaler)  
🏠 **California Housing Dataset** (sklearn.datasets)

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

## 🎯 Running the Script

```bash
source .venv/bin/activate
python3 SCT_ML_1.py
```

### **Expected Output:**
1. **Dataset Loading**: California Housing Dataset (20,640 districts)
2. **Market Analysis**: Income vs housing costs, geographic patterns
3. **Advanced Visualizations**: 4-panel housing market insights
4. **Feature Engineering**: 6 new intelligent features created
5. **Model Training**: Linear regression with 13 features
6. **Performance Metrics**: R², MAE, RMSE results
7. **Coefficient Analysis**: What drives California house prices
8. **Realistic Predictions**: Test scenarios for different regions

---

## 📊 Results Summary

### **California Housing Model Performance:**
- **R² Score**: 0.6519 (65.2% variance explained)
- **Mean Absolute Error**: $48,613
- **Root Mean Square Error**: $67,536
- **Average Prediction Error**: $67,536

### **Key Price Drivers (Coefficient Analysis):**
1. **Latitude** (-0.881): North-South location (Bay Area premium)
2. **Longitude** (-0.881): East-West location (coastal premium)
3. **Median Income** (0.792): The #1 factor driving prices
4. **RoomsPerPerson** (0.520): Spaciousness indicator
5. **BedroomRatio** (0.241): Bedroom efficiency

---

## 🏡 Test Scenarios

The model predicts prices for realistic California properties:

### **San Francisco Bay Area - High End**
- **Location**: Lat 37.7, Lon -122.3
- **Income**: $85,000
- **Predicted Value**: ~$412,559

### **Central Valley - Middle Class**
- **Location**: Lat 36.5, Lon -119.8
- **Income**: $42,000
- **Predicted Value**: ~$171,110

### **Los Angeles Suburbs - Upper Middle**
- **Location**: Lat 34.1, Lon -118.2
- **Income**: $61,000
- **Predicted Value**: ~$309,971

### **Rural Northern California - Budget**
- **Location**: Lat 40.2, Lon -122.1
- **Income**: $28,000
- **Predicted Value**: ~$97,456

---

## 🔧 Issues Solved

1. **Missing Dependencies**: Installed all required Python packages
2. **Virtual Environment**: Properly configured and activated
3. **Data Processing**: Handled California Housing Dataset efficiently
4. **Feature Engineering**: Created meaningful derived features

---

## 📈 Key Insights

### **California Housing Market Patterns:**
- 💰 **Income Premium**: High income areas are 2.8x more expensive
- 🌊 **Coastal Premium**: Coastal areas command premium prices
- 🏖️ **Geographic Patterns**: Bay Area and coastal regions are most expensive
- 📊 **Property Characteristics**: Age, size, and occupancy affect prices
- 🎯 **Location Matters**: Proximity to economic hubs drives prices

---

## 🚧 Future Enhancements

🔹 **Advanced Models**: Random Forest, XGBoost, Neural Networks  
🔹 **Cross-Validation**: Implement k-fold cross-validation  
🔹 **Feature Selection**: Automated feature importance analysis  
🔹 **Model Persistence**: Save and load trained models  
🔹 **API Development**: Create REST API for predictions  
🔹 **Interactive Dashboard**: Streamlit web application  

---

## 🆘 Troubleshooting

If you encounter issues:

1. **ModuleNotFoundError**: Make sure the virtual environment is activated and packages are installed
2. **Display Issues**: The scripts generate matplotlib plots - ensure you have a display environment
3. **Memory Issues**: Large dataset may require sufficient RAM
4. **Plot Display**: Some environments may need backend configuration for matplotlib

---

## 🎓 Learning Outcomes

This project demonstrates:
- **Supervised Learning**: Linear regression implementation
- **Feature Engineering**: Creating meaningful derived features
- **Data Visualization**: Advanced plotting techniques
- **Model Evaluation**: Comprehensive performance metrics
- **Real Estate Analytics**: Business insights from housing data

---

## 🏆 Model Performance

The linear regression model achieves:
- **65.2% variance explained** (R² = 0.6519)
- **$67,536 average prediction error**
- **Clear interpretable coefficients**
- **Realistic predictions** for different California regions

---

🔥 **This project showcases the power of machine learning in real estate analytics and market understanding!**

---

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/) 
[![Machine Learning](https://img.shields.io/badge/Topic-Regression-green)]()
[![Real Estate](https://img.shields.io/badge/Topic-Real%20Estate-orange)]()
