from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import requests
import os
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'error'

API_URL = os.getenv("API_URL", "http://sentiment-api:5000/predict")

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    chats = db.relationship('Chat', backref='user', lazy=True)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(1000), nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database
def initialize_database():
    with app.app_context():
        db.create_all()
        # Create a test user if none exists
        if not User.query.filter_by(username='test').first():
            hashed_password = generate_password_hash('test')
            test_user = User(username='test', password_hash=hashed_password)
            db.session.add(test_user)
            db.session.commit()

# Main Route
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# Auth Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please fill in both username and password fields')
            return redirect(url_for('login'))

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please fill in all fields')
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Chat Routes
@app.route('/chats')
@login_required
def get_chats():
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.asc()).all()
    return jsonify([{
        'id': chat.id,
        'message': chat.message,
        'sentiment': chat.sentiment,
        'confidence': chat.confidence,
        'created_at': chat.created_at.isoformat()
    } for chat in chats])

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response = requests.post(API_URL, json={'text': text})
        data = response.json()

        if 'sentiment' in data:
            # Save to database
            chat = Chat(
                user_id=current_user.id,
                message=data['text'],
                sentiment=data['sentiment'],
                confidence=data['confidence']
            )
            db.session.add(chat)
            db.session.commit()

            return jsonify(data)
        else:
            return jsonify({"error": "Invalid response from sentiment API"}), 500
            return jsonify({"error": "Invalid response from sentiment API"}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    try:
        # Delete all chats for the current user
        deleted_count = Chat.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({
            "success": True,
            "deleted": deleted_count
        })
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error clearing history: {str(e)}")
        return jsonify({
            "error": "Failed to clear chat history",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    initialize_database()
    app.run(host='0.0.0.0', port=5000, debug=True)

