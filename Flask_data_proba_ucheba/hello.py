import os
from flask import Flask, render_template, session, redirect, url_for, flash
# from flask_bootstrap import Bootstrap
# from flask_moment import Moment
# from flask_wtf import FlaskForm
# from wtforms import StringField, SubmitField
# from wtforms.validators import DataRequired
from flask_sqlalchemy import SQLAlchemy
import datetime
from sqlalchemy.sql import func
from datetime import timezone

basedir = os.path.abspath(os.path.dirname(__file__))

# print('--------------------------')
# # выдает: C:\PROJECTS\where_is_my_place_for_a_car\Flask_data_proba_ucheba
# print(basedir)
# print('--------------------------')
app = Flask(__name__)
# app.config['SECRET_KEY'] = 'hard to guess string'

app.config['SQLALCHEMY_DATABASE_URI'] =\
    'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# bootstrap = Bootstrap(app)
# moment = Moment(app)
db = SQLAlchemy(app)


class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(
        db.Date, default=datetime.datetime.now().date(), nullable=False, onupdate=func.now())
    time = db.Column(
        db.Time, default=datetime.datetime.now().time(), nullable=False, onupdate=func.now())
    # date = db.Column(db.String(12), unique=False)
    # time = db.Column(db.String(64), unique=False)
    score = db.Column(db.String(2), unique=False)
    # score = db.Column(db.SmallInteger(), unique=False)

    def __repr__(self):
        return '<Role %r>' % self.name


# class User(db.Model):
#     __tablename__ = 'users'
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(64), unique=True, index=True)
#     role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))

#     def __repr__(self):
#         return '<User %r>' % self.username


# class NameForm(FlaskForm):
#     name = StringField('What is your name?', validators=[DataRequired()])
#     submit = SubmitField('Submit')


# @app.errorhandler(404)
# def page_not_found(e):
#     return render_template('404.html'), 404


# @app.errorhandler(500)
# def internal_server_error(e):
#     return render_template('500.html'), 500


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     form = NameForm()
#     if form.validate_on_submit():
#         old_name = session.get('name')
#         if old_name is not None and old_name != form.name.data:
#             flash('Looks like you have changed your name!')
#         session['name'] = form.name.data
#         return redirect(url_for('index'))
#     return render_template('index.html', form=form, name=session.get('name'))


# if __name__ == '__main__':
#     app.run(debug=True)
