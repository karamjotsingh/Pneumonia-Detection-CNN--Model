import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort
from docassist import app, db, bcrypt
from docassist.forms import RegistrationForm, LoginForm, UpdateAccountForm, RecordForm, SearchForm
from docassist.models import User, Patient
from flask_login import login_user, current_user, logout_user, login_required

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
import numpy as np


@app.route("/")
@app.route("/home")
def home(): 
    return render_template('home.html')    
            

@app.route("/my_records")
@login_required
def my_records():
    records = Patient.query.filter_by(author=current_user).order_by(Patient.date_visited.desc()).all()
    return render_template('my_records.html',records=records)  


@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)


def load_model():
    img_width, img_height = 150, 150

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    global model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    global graph
    graph = tf.get_default_graph()

    model.load_weights("/home/karamjot/My Flask App/Pneumonia Detection System/docassist/original_first_try.h5")

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


def save_xray(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/chest_xray', picture_fn)
    form_picture.save(picture_path)

    return picture_fn


def load_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_tensor = img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)        
    img_tensor /= 255.
    return img_tensor


def prediction(img_path):
    # load a image
    new_image = load_image(img_path)

    # check prediction
    with graph.as_default():
        pred = model.predict(new_image)

    return float(pred)*100;


@app.route("/record/new", methods=['GET', 'POST'])
@login_required
def new_record():
    form = RecordForm()
    if form.validate_on_submit():
        if form.xray.data:
            picture_file = save_xray(form.xray.data)
            picture_path = os.path.join(app.root_path, 'static/chest_xray', picture_file)
            xray_prediction = prediction(picture_path)
            xray_result = ""
            if(xray_prediction > 70):
                xray_result = "YES"
            else:
                xray_result = "NO"    
            record = Patient(name=form.name.data,age=form.age.data,gender=form.gender.data,contact=form.contact.data,
                            weight=form.weight.data, height=form.height.data,
                            medical_history = form.medical_history.data, xray = picture_file, result = xray_result,
                            author=current_user)
        else:
            record = Patient(name=form.name.data,age=form.age.data,gender=form.gender.data,contact=form.contact.data,
                         weight=form.weight.data, height=form.height.data,
                         medical_history = form.medical_history.data, author=current_user)

        db.session.add(record)
        db.session.commit()
        flash('New Patient Record Created!', 'success')
        return redirect(url_for('my_records'))
    return render_template('create_record.html', title='New Patient Record',
                           form=form, legend='New Patient Record')


@app.route("/record/<int:record_id>")
@login_required
def record(record_id):
    patient = Patient.query.get_or_404(record_id)
    image_file = url_for('static', filename='chest_xray/' + patient.xray)
    return render_template('record.html', title=patient.name, patient=patient, image_file=image_file)


@app.route("/record/<int:record_id>/update", methods=['GET', 'POST'])
@login_required
def update_record(record_id):
    patient = Patient.query.get_or_404(record_id)
    if patient.author != current_user:
        abort(403)
    form = RecordForm()
    if form.validate_on_submit():
        patient.name = form.name.data
        patient.age = form.age.data
        patient.gender = form.gender.data
        patient.contact = form.contact.data
        patient.weight = form.weight.data
        patient.height = form.height.data
        patient.medical_history = form.medical_history.data
        if form.xray.data:
            picture_file = save_xray(form.xray.data)
            patient.xray = picture_file
            
            picture_path = os.path.join(app.root_path, 'static/chest_xray', picture_file)
            xray_prediction = prediction(picture_path)
            xray_result = ""
            if(xray_prediction > 70):
                xray_result = "YES"
            else:
                xray_result = "NO"
            patient.result = xray_result    
        db.session.commit()
        image_file = url_for('static', filename='chest_xray/' + patient.xray)
        flash('Patient Record Has Been Updated!', 'success')
        return redirect(url_for('record', record_id=patient.id, image_file=image_file))
    elif request.method == 'GET':
        form.name.data = patient.name
        form.age.data = patient.age
        form.gender.data = patient.gender
        form.contact.data = patient.contact
        form.weight.data = patient.weight
        form.height.data = patient.height
        form.medical_history.data = patient.medical_history
    return render_template('create_record.html', title='Update Patient Record',
                           form=form, legend='Update Patient Record')


@app.route("/record/<int:record_id>/delete", methods=['POST'])
@login_required
def delete_record(record_id):
    patient = Patient.query.get_or_404(record_id)
    if patient.author != current_user:
        abort(403)
    db.session.delete(patient)
    db.session.commit()
    flash('Patient Record Has Been Deleted!', 'success')
    return redirect(url_for('home'))

@app.route("/search", methods=['GET', 'POST'])
@login_required
def search_record():
    form = SearchForm()
    if form.validate_on_submit():
        search_string = form.name.data
        records = Patient.query.filter(Patient.name.like(search_string+"%")).all();
        return render_template('searched_records.html',records=records)
    return render_template('search.html', title='Search Records',form=form)    