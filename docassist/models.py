from datetime import datetime
from docassist import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    records = db.relationship('Patient', backref='author', lazy=True)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.image_file}')"

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(60), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    contact = db.Column(db.String(10), nullable=False)
    date_visited = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    weight = db.Column(db.Integer, nullable=True)    
    height = db.Column(db.Integer, nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    xray = db.Column(db.String(50), nullable=False, default='default.jpg')
    result = db.Column(db.String(5), nullable=True, default='NO')
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Patient('{self.name}')"