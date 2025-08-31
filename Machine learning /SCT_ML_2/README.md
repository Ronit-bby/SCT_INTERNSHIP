# 👥 Customer Segmentation using K-Means Clustering

> 🎯 **Task 2**: Advanced customer segmentation analysis using K-Means clustering with 3D visualizations

---

## 📊 Project Overview

This project implements customer segmentation using K-Means clustering to identify distinct customer groups based on their demographic and behavioral characteristics. The analysis helps businesses understand their customer base and tailor marketing strategies accordingly.

---

## 🎯 What This Project Does

### **Customer Segmentation Analysis**
- **Dataset**: Mall Customers Dataset (200 customers)
- **Algorithm**: K-Means Clustering
- **Features**: Age, Annual Income, Spending Score
- **Output**: Customer segments with 2D and 3D visualizations

### **Key Features:**
- ✅ **Elbow Method**: Automatically determines optimal number of clusters
- ✅ **2D Visualization**: Income vs Spending Score scatter plot
- ✅ **3D Visualization**: Age vs Income vs Spending Score interactive plot
- ✅ **Cluster Profiles**: Statistical analysis of each customer segment
- ✅ **Interactive Plots**: Matplotlib visualizations with color-coded clusters

---

## 📁 Files Description

- `SCT_ML_2.py` - Main customer segmentation script
- `Mall_Customers.csv` - Customer dataset (200 records)
- `requirements.txt` - Python dependencies
- `README.md` - This documentation file

---

## 🛠️ Tech Stack

🐍 **Python**  
📊 **Pandas** · **NumPy** · **Matplotlib** · **Seaborn**  
🤖 **Scikit-learn** (K-Means)  
🎨 **3D Visualization** (mpl_toolkits.mplot3d)

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
cd "Machine learning /SCT_ML_2"
source .venv/bin/activate
python3 SCT_ML_2.py
```

**Important**: Make sure to run the script from within the SCT_ML_2 directory so it can find the Mall_Customers.csv file.

### **Expected Output:**
1. **Elbow Method Plot**: Shows optimal number of clusters (K=5)
2. **2D Scatter Plot**: Customer segments based on Income vs Spending Score
3. **3D Visualization**: Interactive 3D plot showing all three dimensions
4. **Cluster Profiles**: Statistical summary of each customer segment

---

## 📊 Results Summary

### **Customer Segmentation Results:**
- **Optimal Clusters**: 5 customer segments identified
- **Segments Identified**: Different customer profiles based on age, income, and spending patterns

### **Cluster Analysis:**
```
Cluster 0: High Income, High Spending (Premium Customers)
Cluster 1: High Income, Low Spending (Conservative Spenders)
Cluster 2: Low Income, High Spending (Young Trendsetters)
Cluster 3: Low Income, Low Spending (Budget Conscious)
Cluster 4: Medium Income, Medium Spending (Average Customers)
```

---

## 🔧 Issues Solved

1. **Missing Dataset**: Created `Mall_Customers.csv` with 200 realistic customer records
2. **Data Type Error**: Fixed the groupby operation to only calculate means for numeric columns
3. **Dependencies**: Installed all required Python packages
4. **Virtual Environment**: Properly configured and activated

---

## 📈 Key Insights

### **Customer Behavior Patterns:**
- 💰 **Income vs Spending**: Strong correlation between income and spending behavior
- 👥 **Age Groups**: Different age segments show distinct spending patterns
- 🎯 **Target Segments**: Clear identification of high-value customers
- 📊 **Market Opportunities**: Identification of underserved customer segments

---

## 🚧 Future Enhancements

🔹 **Advanced Clustering**: Try DBSCAN, Hierarchical Clustering  
🔹 **Feature Engineering**: Add more customer attributes  
🔹 **Interactive Dashboard**: Create Streamlit web app  
🔹 **Real-time Analysis**: Connect to live customer data  
🔹 **Predictive Modeling**: Add customer lifetime value prediction  

---

## 🆘 Troubleshooting

If you encounter issues:

1. **ModuleNotFoundError**: Make sure the virtual environment is activated and packages are installed
2. **FileNotFoundError**: Ensure `Mall_Customers.csv` is in the same directory as the script
3. **Display Issues**: The scripts generate matplotlib plots - ensure you have a display environment
4. **3D Plot Issues**: Some environments may not support 3D plots - check your matplotlib backend

---

## 🎓 Learning Outcomes

This project demonstrates:
- **Unsupervised Learning**: K-Means clustering implementation
- **Data Visualization**: 2D and 3D plotting techniques
- **Customer Analytics**: Business insights from customer data
- **Model Evaluation**: Elbow method for optimal cluster selection

---

🔥 **Customer segmentation is a powerful tool for business intelligence and targeted marketing strategies!**

---

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/) 
[![Machine Learning](https://img.shields.io/badge/Topic-Clustering-green)]()
[![Data Visualization](https://img.shields.io/badge/Topic-Data%20Visualization-orange)]()
