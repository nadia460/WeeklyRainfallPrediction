from flask_sqlalchemy import SQLAlchemy
from app import app
from sqlalchemy import create_engine

db = SQLAlchemy(app)

class DataLatih(db.Model):
    id = db.Column(db.Integer, unique=True, primary_key=True, nullable=False)
    nama_stasiun = db.Column(db.String, nullable=False)
    file = db.Column(db.LargeBinary)
    status = db.Column(db.String, nullable=False)
    ket = db.Column(db.String, nullable=False)
    
    def __repr__(self):
        return "<Name: {}>".format(self.nama_stasiun)