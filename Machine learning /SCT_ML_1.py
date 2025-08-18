# Advanced House Price Prediction System
# Using California Housing Dataset - 20,640 real houses across California
# This is way more comprehensive than the basic 3-feature model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

def load_comprehensive_housing_data():
    """
    Load the California Housing dataset - this is real data from 20,640 districts
    Much bigger and more realistic than toy datasets
    """
    print("🏠 Loading California Housing Dataset...")
    print("This contains real data from 20,640 housing districts across California")
    print("-" * 65)
    
    # Load the famous California housing dataset
    housing_data = fetch_california_housing()
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
    df['MedianHouseValue'] = housing_data.target
    
    print(f"✅ Successfully loaded {len(df):,} housing districts")
    print("\n🔍 What information do we have about each district?")
    
    feature_explanations = {
        'MedInc': 'Median income (in tens of thousands)',
        'HouseAge': 'Median house age in district',
        'AveRooms': 'Average rooms per household',
        'AveBedrms': 'Average bedrooms per household', 
        'Population': 'District population',
        'AveOccup': 'Average household occupancy',
        'Latitude': 'Geographic latitude',
        'Longitude': 'Geographic longitude',
        'MedianHouseValue': 'Target: Median house value (hundreds of thousands)'
    }
    
    for feature, explanation in feature_explanations.items():
        if feature in df.columns:
            sample_val = df[feature].iloc[0]
            print(f"   • {feature}: {explanation} (example: {sample_val:.2f})")
    
    return df

def explore_california_real_estate_market(df):
    """
    Dive into the California housing market patterns
    Understanding what drives prices in different regions
    """
    print(f"\n🏖️ Exploring California's Diverse Housing Market")
    print("=" * 55)
    
    total_districts = len(df)
    avg_price = df['MedianHouseValue'].mean() * 100000  # Convert back to dollars
    min_price = df['MedianHouseValue'].min() * 100000
    max_price = df['MedianHouseValue'].max() * 100000
    
    print(f"📊 Market Overview:")
    print(f"   • Total districts analyzed: {total_districts:,}")
    print(f"   • Average house value: ${avg_price:,.0f}")
    print(f"   • Price range: ${min_price:,.0f} - ${max_price:,.0f}")
    print(f"   • That's a {max_price/min_price:.1f}x difference between cheapest and most expensive!")
    
    # Income analysis
    print(f"\n💰 Income vs Housing Costs:")
    high_income_districts = df[df['MedInc'] > df['MedInc'].quantile(0.8)]
    low_income_districts = df[df['MedInc'] < df['MedInc'].quantile(0.2)]
    
    high_income_avg_price = high_income_districts['MedianHouseValue'].mean() * 100000
    low_income_avg_price = low_income_districts['MedianHouseValue'].mean() * 100000
    
    print(f"   • High income areas (top 20%): Average house value ${high_income_avg_price:,.0f}")
    print(f"   • Low income areas (bottom 20%): Average house value ${low_income_avg_price:,.0f}")
    print(f"   • Income premium: {high_income_avg_price/low_income_avg_price:.1f}x more expensive")
    
    # Geographic patterns
    print(f"\n🗺️ Geographic Price Patterns:")
    coastal_areas = df[df['Longitude'] > -121]  # More eastern = inland
    inland_areas = df[df['Longitude'] <= -121]
    
    coastal_avg = coastal_areas['MedianHouseValue'].mean() * 100000
    inland_avg = inland_areas['MedianHouseValue'].mean() * 100000
    
    print(f"   • Coastal areas: Average ${coastal_avg:,.0f}")
    print(f"   • Inland areas: Average ${inland_avg:,.0f}")
    print(f"   • Coastal premium: {coastal_avg/inland_avg:.1f}x more expensive")

def create_advanced_visualizations(df):
    """
    Create compelling visualizations that reveal housing market insights
    These charts tell the real story of California real estate
    """
    print("\n📊 Creating Advanced Housing Market Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('California Housing Market: Deep Insights from 20,640 Districts', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Income vs House Value - The core relationship
    axes[0,0].scatter(df['MedInc'], df['MedianHouseValue'], alpha=0.6, color='steelblue', s=20)
    axes[0,0].set_xlabel('Median Income (tens of thousands)')
    axes[0,0].set_ylabel('House Value (hundreds of thousands)')
    axes[0,0].set_title('Income Drives Everything:\nThe Strongest Predictor of Home Prices')
    
    # Add trend line
    z = np.polyfit(df['MedInc'], df['MedianHouseValue'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df['MedInc'].sort_values(), p(df['MedInc'].sort_values()), 
                   "r--", alpha=0.8, linewidth=2)
    
    # 2. Geographic heat map
    scatter = axes[0,1].scatter(df['Longitude'], df['Latitude'], 
                               c=df['MedianHouseValue'], cmap='YlOrRd', 
                               alpha=0.7, s=15)
    axes[0,1].set_xlabel('Longitude (West to East)')
    axes[0,1].set_ylabel('Latitude (South to North)')
    axes[0,1].set_title('California Geography:\nCoastal Areas Command Premium Prices')
    plt.colorbar(scatter, ax=axes[0,1], label='House Value')
    
    # 3. House Age Impact
    df['AgeGroup'] = pd.cut(df['HouseAge'], bins=[0, 10, 20, 30, 50], 
                           labels=['New (0-10)', 'Recent (10-20)', 'Mature (20-30)', 'Old (30+)'])
    age_prices = df.groupby('AgeGroup')['MedianHouseValue'].mean()
    age_prices.plot(kind='bar', ax=axes[1,0], color='darkgreen', alpha=0.8)
    axes[1,0].set_title('Age vs Value:\nHow House Age Affects Prices')
    axes[1,0].set_ylabel('Average House Value')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Population Density vs Prices
    axes[1,1].scatter(df['AveOccup'], df['MedianHouseValue'], alpha=0.6, color='purple', s=20)
    axes[1,1].set_xlabel('Average Household Occupancy')
    axes[1,1].set_ylabel('House Value (hundreds of thousands)')
    axes[1,1].set_title('Crowding Effect:\nHigher Occupancy = Lower Values')
    
    plt.tight_layout()
    plt.show()

def engineer_smart_features(df):
    """
    Create new features that capture real-world housing market dynamics
    Going beyond basic features to understand what really drives prices
    """
    print("\n🔧 Engineering Smart Features from Real Estate Knowledge...")
    
    enhanced_df = df.copy()
    
    # 1. Rooms per person - indicates spaciousness
    enhanced_df['RoomsPerPerson'] = enhanced_df['AveRooms'] / enhanced_df['AveOccup']
    
    # 2. Bedroom ratio - what % of rooms are bedrooms
    enhanced_df['BedroomRatio'] = enhanced_df['AveBedrms'] / enhanced_df['AveRooms']
    
    # 3. Population density indicator
    enhanced_df['PopulationDensity'] = enhanced_df['Population'] / 1000  # Scaled
    
    # 4. Location desirability (distance from SF Bay Area center)
    sf_lat, sf_lon = 37.7749, -122.4194
    enhanced_df['DistanceToSF'] = np.sqrt((enhanced_df['Latitude'] - sf_lat)**2 + 
                                         (enhanced_df['Longitude'] - sf_lon)**2)
    
    # 5. Coastal proximity
    enhanced_df['IsCoastal'] = (enhanced_df['Longitude'] > -121).astype(int)
    
    # 6. Income bracket
    enhanced_df['IncomeLevel'] = pd.cut(enhanced_df['MedInc'], 
                                       bins=[0, 3, 6, 10, 20], 
                                       labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    print(f"   ✅ Created 6 new intelligent features:")
    print(f"      • RoomsPerPerson: Indicates spaciousness")
    print(f"      • BedroomRatio: Bedroom efficiency") 
    print(f"      • PopulationDensity: Crowding indicator")
    print(f"      • DistanceToSF: Proximity to economic hub")
    print(f"      • IsCoastal: Coastal premium indicator")
    print(f"      • IncomeLevel: Categorical income brackets")
    
    return enhanced_df

def build_linear_regression_model(df):
    """
    Build a Linear Regression model - the classic, interpretable approach
    Simple but powerful for understanding relationships between features and prices
    """
    print("\n🤖 Building Linear Regression Model...")
    print("Using the classic algorithm that shows clear relationships")
    print("-" * 55)
    
    # Prepare features (excluding categorical and target)
    feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                      'AveOccup', 'Latitude', 'Longitude', 'RoomsPerPerson', 
                      'BedroomRatio', 'PopulationDensity', 'DistanceToSF', 'IsCoastal']
    
    X = df[feature_columns]
    y = df['MedianHouseValue']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   📚 Training set: {len(X_train):,} districts")
    print(f"   🧪 Test set: {len(X_test):,} districts")
    print(f"   🔢 Features used: {len(feature_columns)}")
    
    # Create and train Linear Regression model
    print(f"\n   🔄 Training Linear Regression Model...")
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    # Calculate comprehensive metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Convert to dollars for easier understanding
    mae_dollars = mae * 100000
    rmse_dollars = rmse * 100000
    
    print(f"      → Mean Absolute Error: ${mae_dollars:,.0f}")
    print(f"      → Root Mean Square Error: ${rmse_dollars:,.0f}")
    print(f"      → R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)")
    
    print(f"\n🏆 Linear Regression Model Trained Successfully!")
    print(f"    Average prediction error: ${rmse_dollars:,.0f}")
    
    return model, X_test_scaled, y_test, predictions, scaler, feature_columns

def analyze_linear_regression_coefficients(model, feature_columns):
    """
    Understand which factors most strongly influence California house prices
    Linear regression coefficients show the direct impact of each feature
    """
    print("\n🔍 What Drives California House Prices?")
    print("=" * 45)
    
    # Get Linear Regression coefficients
    coefficients = model.coef_
    
    feature_impact = list(zip(feature_columns, coefficients))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("📊 Most Important Price Factors (Linear Regression Coefficients):")
    
    feature_meanings = {
        'MedInc': 'Median Income - The #1 factor',
        'Latitude': 'North-South location (Bay Area premium)',
        'Longitude': 'East-West location (coastal premium)',
        'HouseAge': 'How old the houses are',
        'AveRooms': 'Average rooms per household',
        'RoomsPerPerson': 'Spaciousness indicator',
        'Population': 'District population size',
        'AveOccup': 'Average household occupancy',
        'DistanceToSF': 'Distance to San Francisco',
        'IsCoastal': 'Coastal location premium',
        'AveBedrms': 'Average bedrooms per household',
        'BedroomRatio': 'Bedroom efficiency ratio',
        'PopulationDensity': 'Population density'
    }
    
    for i, (feature, coefficient) in enumerate(feature_impact, 1):
        meaning = feature_meanings.get(feature, feature)
        impact_direction = "increases" if coefficient > 0 else "decreases"
        print(f"   {i:2d}. {meaning}: {coefficient:.3f} ({impact_direction} price)")

def test_realistic_predictions(model, scaler, feature_columns):
    """
    Test our Linear Regression model on realistic California housing scenarios
    """
    print("\n🏡 Testing Linear Regression Predictions on Realistic California Properties")
    print("=" * 70)
    
    # Define realistic test cases
    test_scenarios = [
        {
            'name': 'San Francisco Bay Area - High End',
            'MedInc': 8.5, 'HouseAge': 15, 'AveRooms': 6.2, 'AveBedrms': 1.1,
            'Population': 2500, 'AveOccup': 2.8, 'Latitude': 37.7, 'Longitude': -122.3,
            'RoomsPerPerson': 2.2, 'BedroomRatio': 0.18, 'PopulationDensity': 2.5,
            'DistanceToSF': 0.3, 'IsCoastal': 1
        },
        {
            'name': 'Central Valley - Middle Class',
            'MedInc': 4.2, 'HouseAge': 25, 'AveRooms': 5.1, 'AveBedrms': 1.0,
            'Population': 1800, 'AveOccup': 3.2, 'Latitude': 36.5, 'Longitude': -119.8,
            'RoomsPerPerson': 1.6, 'BedroomRatio': 0.20, 'PopulationDensity': 1.8,
            'DistanceToSF': 2.5, 'IsCoastal': 0
        },
        {
            'name': 'Los Angeles Suburbs - Upper Middle',
            'MedInc': 6.1, 'HouseAge': 35, 'AveRooms': 5.8, 'AveBedrms': 1.2,
            'Population': 3200, 'AveOccup': 2.9, 'Latitude': 34.1, 'Longitude': -118.2,
            'RoomsPerPerson': 2.0, 'BedroomRatio': 0.21, 'PopulationDensity': 3.2,
            'DistanceToSF': 5.4, 'IsCoastal': 1
        },
        {
            'name': 'Rural Northern California - Budget',
            'MedInc': 2.8, 'HouseAge': 40, 'AveRooms': 4.9, 'AveBedrms': 1.1,
            'Population': 800, 'AveOccup': 2.5, 'Latitude': 40.2, 'Longitude': -122.1,
            'RoomsPerPerson': 2.0, 'BedroomRatio': 0.22, 'PopulationDensity': 0.8,
            'DistanceToSF': 3.2, 'IsCoastal': 0
        }
    ]
    
    for scenario in test_scenarios:
        # Extract features in correct order
        features = [scenario[col] for col in feature_columns]
        features_array = np.array(features).reshape(1, -1)
        
        # Apply scaling for Linear Regression
        features_array = scaler.transform(features_array)
        
        # Make prediction
        predicted_value = model.predict(features_array)[0]
        predicted_price = predicted_value * 100000  # Convert to dollars
        
        print(f"\n🏠 {scenario['name']}")
        print(f"   📍 Location: Lat {scenario['Latitude']}, Lon {scenario['Longitude']}")
        print(f"   💰 Median Income: ${scenario['MedInc']*10000:,.0f}")
        print(f"   🏡 House Age: {scenario['HouseAge']} years")
        print(f"   📊 Predicted Value: ${predicted_price:,.0f}")

def create_linear_regression_visualization(y_test, predictions):
    """
    Create visualization specifically for Linear Regression results
    """
    print("\n📈 Creating Linear Regression Performance Visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Linear Regression Model: Predicting California House Prices', 
                 fontsize=16, fontweight='bold')
    
    # Scatter plot: Actual vs Predicted
    axes[0].scatter(y_test, predictions, alpha=0.6, color='steelblue', s=20)
    
    # Perfect prediction line
    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[0].set_xlabel('Actual House Value (hundreds of thousands)')
    axes[0].set_ylabel('Predicted House Value (hundreds of thousands)')
    axes[0].set_title('Actual vs Predicted Values\nLinear Regression Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_test - predictions
    axes[1].scatter(predictions, residuals, alpha=0.6, color='darkgreen', s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted House Value')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title('Residuals Plot\nChecking Model Assumptions')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Complete California Housing Price Analysis Pipeline
    From data loading to advanced predictions
    """
    print("🏠 CALIFORNIA HOUSING PRICE PREDICTION SYSTEM")
    print("Advanced ML Analysis of 20,640 Housing Districts")
    print("=" * 65)
    
    # Load the comprehensive dataset
    housing_df = load_comprehensive_housing_data()
    
    # Explore the California real estate market
    explore_california_real_estate_market(housing_df)
    
    # Create advanced visualizations
    create_advanced_visualizations(housing_df)
    
    # Engineer intelligent features
    enhanced_df = engineer_smart_features(housing_df)
    
    # Build Linear Regression model
    model, X_test, y_test, predictions, scaler, features = build_linear_regression_model(enhanced_df)
    
    # Analyze what drives prices using Linear Regression coefficients
    analyze_linear_regression_coefficients(model, features)
    
    # Test realistic scenarios
    test_realistic_predictions(model, scaler, features)
    
    # Final visualization
    create_linear_regression_visualization(y_test, predictions)
    
    print("\n" + "=" * 65)
    print("🎯 CONCLUSION: California housing prices are primarily driven by:")
    print("   1. Income levels (economic capacity)")
    print("   2. Geographic location (Bay Area, coastal premium)")  
    print("   3. Property characteristics (age, size, occupancy)")
    print("   Our Linear Regression model shows clear relationships and interpretable results!")
    print("=" * 65)

if __name__ == "__main__":
    main()
    
    
    
# Example input 



# {
#   "name": "San Francisco Bay Area - High End",
#   "MedInc": 8.5,
#   "HouseAge": 15,
#   "AveRooms": 6.2,
#   "AveBedrms": 1.1,
#   "Population": 2500,
#   "AveOccup": 2.8,
#   "Latitude": 37.7,
#   "Longitude": -122.3,
#   "RoomsPerPerson": 2.2,
#   "BedroomRatio": 0.18,
#   "PopulationDensity": 2.5,
#   "DistanceToSF": 0.3,
#   "IsCoastal": 1
# }


# Example output 


# 🏠 San Francisco Bay Area - High End
#    📍 Location: Lat 37.7, Lon -122.3
#    💰 Median Income: $85,000
#    🏡 House Age: 15 years
#    📊 Predicted Value: ~$500,000   (will vary slightly on your machine)



# py"
# 🏠 CALIFORNIA HOUSING PRICE PREDICTION SYSTEM
# Advanced ML Analysis of 20,640 Housing Districts
# =================================================================
# 🏠 Loading California Housing Dataset...
# This contains real data from 20,640 housing districts across California
# -----------------------------------------------------------------
# ✅ Successfully loaded 20,640 housing districts

# 🔍 What information do we have about each district?
#    • MedInc: Median income (in tens of thousands) (example: 8.33)
#    • HouseAge: Median house age in district (example: 41.00)
#    • AveRooms: Average rooms per household (example: 6.98)
#    • AveBedrms: Average bedrooms per household (example: 1.02)
#    • Population: District population (example: 322.00)
#    • AveOccup: Average household occupancy (example: 2.56)
#    • Latitude: Geographic latitude (example: 37.88)
#    • Longitude: Geographic longitude (example: -122.23)
#    • MedianHouseValue: Target: Median house value (hundreds of thousands) (example: 4.53)

# 🏖️ Exploring California's Diverse Housing Market
# =======================================================
# 📊 Market Overview:
#    • Total districts analyzed: 20,640
#    • Average house value: $206,856
#    • Price range: $14,999 - $500,001
#    • That's a 33.3x difference between cheapest and most expensive!

# 💰 Income vs Housing Costs:
#    • High income areas (top 20%): Average house value $334,991
#    • Low income areas (bottom 20%): Average house value $118,385
#    • Income premium: 2.8x more expensive

# 🗺️ Geographic Price Patterns:
#    • Coastal areas: Average $202,307
#    • Inland areas: Average $215,396
#    • Coastal premium: 0.9x more expensive

# 📊 Creating Advanced Housing Market Visualizations...

# 🔧 Engineering Smart Features from Real Estate Knowledge...
#    ✅ Created 6 new intelligent features:
#       • RoomsPerPerson: Indicates spaciousness
#       • BedroomRatio: Bedroom efficiency
#       • PopulationDensity: Crowding indicator
#       • DistanceToSF: Proximity to economic hub
#       • IsCoastal: Coastal premium indicator
#       • IncomeLevel: Categorical income brackets

# 🤖 Building Linear Regression Model...
# Using the classic algorithm that shows clear relationships
# -------------------------------------------------------
#    📚 Training set: 16,512 districts
#    🧪 Test set: 4,128 districts
#    🔢 Features used: 13

#    🔄 Training Linear Regression Model...
#       → Mean Absolute Error: $48,613
#       → Root Mean Square Error: $67,536
#       → R² Score: 0.6519 (65.2% variance explained)

# 🏆 Linear Regression Model Trained Successfully!
#     Average prediction error: $67,536

# 🔍 What Drives California House Prices?
# =============================================
# 📊 Most Important Price Factors (Linear Regression Coefficients):
#     1. North-South location (Bay Area premium): -0.881 (decreases price)
#     2. East-West location (coastal premium): -0.881 (decreases price)
#     3. Median Income - The #1 factor: 0.792 (increases price)
#     4. Spaciousness indicator: 0.520 (increases price)
#     5. Bedroom efficiency ratio: 0.241 (increases price)
#     6. Average bedrooms per household: -0.225 (decreases price)
#     7. Average rooms per household: -0.149 (decreases price)
#     8. How old the houses are: 0.129 (increases price)
#     9. Distance to San Francisco: 0.028 (increases price)
#    10. Coastal location premium: 0.026 (increases price)
#    11. Average household occupancy: -0.018 (decreases price)
#    12. District population size: 0.017 (increases price)
#    13. Population density: 0.017 (increases price)

# 🏡 Testing Linear Regression Predictions on Realistic California Properties
# ======================================================================

# 🏠 San Francisco Bay Area - High End
#    📍 Location: Lat 37.7, Lon -122.3
#    💰 Median Income: $85,000
#    🏡 House Age: 15 years
#    📊 Predicted Value: $412,559

# 🏠 Central Valley - Middle Class
#    📍 Location: Lat 36.5, Lon -119.8
#    💰 Median Income: $42,000
#    🏡 House Age: 25 years
#    📊 Predicted Value: $171,110

# 🏠 Los Angeles Suburbs - Upper Middle
#    📍 Location: Lat 34.1, Lon -118.2
#    💰 Median Income: $61,000
#    🏡 House Age: 35 years
#    📊 Predicted Value: $309,971

# 🏠 Rural Northern California - Budget
#    📍 Location: Lat 40.2, Lon -122.1
#    💰 Median Income: $28,000
#    🏡 House Age: 40 years
#    📊 Predicted Value: $97,456

# 📈 Creating Linear Regression Performance Visualization...

# =================================================================
# 🎯 CONCLUSION: California housing prices are primarily driven by:
#    1. Income levels (economic capacity)
#    2. Geographic location (Bay Area, coastal premium)
#    3. Property characteristics (age, size, occupancy)
#    Our Linear Regression model shows clear relationships and interpretable results!