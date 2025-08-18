# SCT_INTERNSHIP
🏡 Advanced House Price Prediction System
A Machine Learning project that predicts house prices using the California Housing Dataset (20,640 real housing records). This project was built as my first internship task in Machine Learning, going beyond a basic regression model with multiple features, visualizations, and performance evaluation.
📌 Features
📊 Exploratory Data Analysis (EDA) with plots for insights
🧩 Feature Engineering (rooms per person, bedroom ratio, coastal proximity, etc.)
⚡ Linear Regression Model trained on real-world housing data
📉 Performance Metrics: R², MAE, RMSE
🌍 Visualization of Housing Prices Across California
📂 Dataset
We use the California Housing Dataset, which includes:
20,640 housing records
Features: Median income, house age, total rooms, bedrooms, population, location, proximity to ocean, etc.
Target: Median House Value
The dataset is available through Scikit-learn’s fetch_california_housing function.
🚀 Installation & Setup
Clone the repo:
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
Install dependencies:
pip install -r requirements.txt
Run the notebook / script:
python house_price_prediction.py
📊 Results & Insights
Median income and geography are the strongest predictors of house prices
Feature engineering (like ratios) improved interpretability
Visualizations showed clear patterns:
Higher incomes → higher house values
Coastal areas tend to be more expensive
House age has an interesting non-linear impact
Sample Visualization: (add a screenshot if you want)
[Insert plot here: e.g. income vs house price scatter]
🧠 What I Learned
Basics of Regression Modeling
Importance of EDA & feature engineering
Evaluating models with multiple metrics
Using visualizations to tell data stories
🛠️ Tech Stack
Python 🐍
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
📌 Future Work
Try advanced models (Random Forest, Gradient Boosting, XGBoost)
Add hyperparameter tuning
Deploy as a simple web app (Flask/Streamlit)
🙌 Acknowledgements
Dataset from Scikit-learn
Internship task guidance from my mentors
🔥 First step into my Machine Learning journey. More exciting projects coming soon!
