from flask import Flask, render_template,request,session,flash, jsonify
import sqlite3 as sql
import os
import pandas as pd
import subprocess
app = Flask(__name__)

gestures = {
    "Thumbs Up": "Approval",
    "Peace Sign": "Peace",
    "Wave": "Hello",
    "Clenched Fist": "Power",
}

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/gohome')
def homepage():
    return render_template('index.html')

@app.route('/service')
def servicepage():
    
    subprocess.call("python testing.py 1")
    return render_template('home.html')


@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("user.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO agriuser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",(nm,phonno,email,unm,passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)



@app.route('/userlogin')
def user_login():
   return render_template("login.html")

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("user.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM agriuser where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('home.html')
                    else:
                        flash("Invalid user credentials")
                        return render_template('login.html')

@app.route('/predictinfo')
def predictin():
   
   return render_template('matching.html')

@app.route("/check", methods=["POST"])
def check():
    selected_gesture = request.form.get("gesture")
    selected_meaning = request.form.get("meaning")
    correct_meaning = gestures.get(selected_gesture)

    if correct_meaning == selected_meaning:
        return jsonify({"result": "correct", "message": "Correct Match!"})
    else:
        return jsonify({"result": "wrong", "message": "Wrong Match!"})
    
@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/sudoku')
def sudoku():
    return render_template('sudoku_game.html')



@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)

