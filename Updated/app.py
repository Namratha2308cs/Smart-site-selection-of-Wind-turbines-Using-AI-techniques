from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import joblib
import numpy as np
from database import init_db, get_db, User, PredictionHistory
from auth import login_required, get_user_id
from model import predict_site
import json
from datetime import datetime
from auth import register_user, login_user, logout_user
import json
from jinja2 import Environment




app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
app.jinja_env.filters['fromjson'] = json.loads

# Initialize database
init_db(app)

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    user_id = get_user_id()
    
    # Get user's prediction history
    history = db.execute(
        'SELECT * FROM prediction_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 5',
        (user_id,)
    ).fetchall()
    
    return render_template('dashboard.html', history=history)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'Wind Speed (m/s)': float(request.form['wind_speed']),
                'Terrain Type': request.form['terrain_type'],
                'Land Use': request.form['land_use'],
                'Grid Connectivity (km)': float(request.form['grid_connectivity']),
                'Environmental Impact': int(request.form['environmental_impact']),
                'Economic Viability': int(request.form['economic_viability']),
                'wind_speed (m/s)': float(request.form['detailed_wind_speed']),
                'wind_direction (degrees)': float(request.form['wind_direction']),
                'temperature (C)': float(request.form['temperature']),
                'humidity (%)': float(request.form['humidity']),
                'seasonal_variation': request.form['seasonal_variation'],
                'elevation (m)': float(request.form['elevation']),
                'proximity_to_power_lines (km)': float(request.form['proximity_power_lines']),
                'wildlife_habitats_nearby': request.form['wildlife_habitats'],
                'protected_areas_nearby': request.form['protected_areas'],
                'zoning_laws': request.form['zoning_laws'],
                'permit_status': request.form['permit_status'],
                'accessibility': request.form['accessibility'],
                'community_acceptance': request.form['community_acceptance'],
                'installation_cost (million $)': float(request.form['installation_cost']),
                'energy_output_potential (MW)': float(request.form['energy_output'])
            }
            
            # Make prediction - this now returns all three values
            from model import predict_site
            prediction, probability, feature_importance = predict_site(form_data)
            
            # Save prediction to history
            db = get_db()
            db.execute(
                '''INSERT INTO prediction_history 
                (user_id, prediction, probability, input_data, feature_importance, created_at)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (get_user_id(), int(prediction), float(probability), 
                 json.dumps(form_data), json.dumps(feature_importance), datetime.utcnow())
            )
            db.commit()
            
            return render_template('result.html', 
                                prediction=prediction,
                                probability=probability,
                                feature_importance=feature_importance,
                                form_data=form_data)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')

    


@app.route('/prediction_history')
@login_required
def prediction_history():
    db = get_db()
    user_id = get_user_id()
    
    history = db.execute(
        'SELECT * FROM prediction_history WHERE user_id = ? ORDER BY created_at DESC',
        (user_id,)
    ).fetchall()
    
    return render_template('history.html', history=history)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        success, message = register_user(username, email, password)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        success, message = login_user(username, password)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('dashboard'))
        else:
            flash(message, 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))
@app.route('/api/prediction_stats')
@login_required
def prediction_stats():
    db = get_db()
    user_id = get_user_id()
    row = db.execute("""
        SELECT 
            COUNT(*)                                   AS total_predictions,
            COALESCE(SUM(CASE WHEN prediction=1 THEN 1 END), 0) AS optimal_predictions,
            COALESCE(AVG(probability), 0)              AS avg_probability
        FROM prediction_history
        WHERE user_id = ?
    """, (user_id,)).fetchone()

    payload = {
        "total_predictions": int(row["total_predictions"] or 0),
        "optimal_predictions": int(row["optimal_predictions"] or 0),
        "avg_probability": round(float(row["avg_probability"] or 0) * 100.0, 2),
    }

    from flask import make_response, jsonify
    resp = make_response(jsonify(payload), 200)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


if __name__ == '__main__':
    app.run(debug=True)