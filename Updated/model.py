import pandas as pd
import joblib
import numpy as np
from sklearn.inspection import permutation_importance
import os

class WindTurbinePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load('wind_turbine_model_complex.pkl')
            # Get feature names from training data
            data = pd.read_csv('es.csv')
            X = data.drop("Optimal for Turbine", axis=1)
            X_encoded = pd.get_dummies(X, drop_first=True)
            self.feature_names = X_encoded.columns.tolist()
            print(f"Model loaded successfully with {len(self.feature_names)} features")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def preprocess_input(self, input_data):
        """Preprocess input data to match model expectations"""
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        
        # Ensure all expected columns are present
        for feature in self.feature_names:
            if feature not in input_encoded.columns:
                input_encoded[feature] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[self.feature_names]
        
        return input_encoded
    
    def predict(self, input_data):
        """Make prediction"""
        input_processed = self.preprocess_input(input_data)
        prediction = self.model.predict(input_processed)[0]
        probability = self.model.predict_proba(input_processed)[0][1]
        return prediction, probability
    
    def get_feature_importance(self, input_data):
        """Get feature importance for the prediction"""
        try:
            input_processed = self.preprocess_input(input_data)
            
            # Calculate permutation importance
            result = permutation_importance(
                self.model, input_processed, self.model.predict(input_processed), 
                n_repeats=5, random_state=42, n_jobs=1
            )
            
            # Get importance scores
            importance_scores = result.importances_mean
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                feature_importance[feature] = float(importance_scores[i])
            
            # Sort by absolute importance and get top 10
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10])
            
            return feature_importance
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            # Return default importance if calculation fails
            return {"Wind Speed (m/s)": 0.1, "installation_cost (million $)": 0.08, "energy_output_potential (MW)": 0.07}

# Global predictor instance
predictor = WindTurbinePredictor()

def predict_site(input_data):
    """Main function to predict site optimality"""
    prediction, probability = predictor.predict(input_data)
    feature_importance = predictor.get_feature_importance(input_data)
    return prediction, probability, feature_importance

def get_feature_info():
    """Get information about features"""
    categorical_columns = ['Terrain Type', 'Land Use', 'seasonal_variation', 
                         'wildlife_habitats_nearby', 'protected_areas_nearby', 
                         'zoning_laws', 'permit_status', 'accessibility', 
                         'community_acceptance']
    
    numerical_columns = ['Wind Speed (m/s)', 'Grid Connectivity (km)', 
                       'Environmental Impact', 'Economic Viability', 
                       'wind_speed (m/s)', 'wind_direction (degrees)', 
                       'temperature (C)', 'humidity (%)', 'elevation (m)', 
                       'proximity_to_power_lines (km)', 'installation_cost (million $)', 
                       'energy_output_potential (MW)']
    
    return categorical_columns, numerical_columns