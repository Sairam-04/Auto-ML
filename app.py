from flask import Flask, g, session, redirect, request, render_template, url_for, send_file
import os
import pymongo
from decouple import config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import pandas_profiling
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from flask_ngrok import run_with_ngrok

app = Flask(__name__)
# run_with_ngrok(app)

app.secret_key = os.urandom(24)

MONGODB_URI = config('MONGODB_URI')
client = pymongo.MongoClient(MONGODB_URI)
db = client['automl']
users = db['users']
counter = db['counter']
cr_data = db['classification_regression']

@app.before_request
def before_request():
    g.user = None 
    if 'user' in session:
        g.user = session['user']

@app.route('/')
def index():
    data = []
    if g.user:
        for x in cr_data.find({"username":session["user"]}):
            data.append([
                x["model_id"],
                x["project_name"],
             x["date"],
            ])
    
        return render_template("index.html",data = data,username=session['user'])
    return redirect(url_for('login'))


@app.route('/login',methods=['GET', 'POST'])
def login():
    global invalid_user
    if request.method == 'POST':
        session.pop('user',None)
        user_list = users.find_one({"username":request.form['username']})
        if user_list:
            if request.form['password'] == user_list['password']:
                session['user'] = request.form['username']
                return redirect(url_for('index',username=session['user']))
            return render_template('login.html',invalid_user="Invalid Username or Password")
        return render_template('login.html',invalid_user="Invalid Username or Password")
    if request.args.get("account_created"):
        return render_template('login.html',msg2="Account created successfully")
    return render_template('login.html')

@app.route('/signup',methods=['GET', 'POST'])
def signup():
    global username_taken_msg
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        user_list = users.find_one({"username":username})
        if user_list:
            username_taken_msg = "Username already taken, try another one"
            return render_template('signup.html',username_taken_msg=username_taken_msg)
        users.insert_one({"username":username,"password":password})
        return redirect(url_for('login', account_created=True))
    return render_template('signup.html')

@app.route('/create-project', methods=["GET","POST"])
def createProject():
    username = session["user"]
    if request.method == "GET":
        return render_template("createproject.html", username = username)
    task_type = request.form["tasktype"]
    project_name = request.form["projectname"]
    now = datetime.now()
    date = now.strftime("%B %d, %Y")
    model_data = counter.find_one({"type":"model"})
    model_id = model_data['count'] + 1
    debug(model_id)
    counter.update_one({"type":"model"},{"$set":{"count":model_id}})
    cr_data.insert_one({"username":username,"model_id":model_id,"project_name":project_name,"deploy":"no", "date":date,"task_type":task_type})

    if task_type == "classification":
        return redirect(url_for('classify', model_id = model_id))
        # return render_template("classify.html", model_id = model_id)
    elif task_type == "regression":
        return redirect(url_for('regression', model_id = model_id))
        # return render_template("regression.html", model_id = model_id)

def generate_report(input, output):
    df1 = pd.read_csv(input)
    df2 = pd.read_csv(output)
    df = pd.concat([df1, df2], axis=1)
    report = pandas_profiling.ProfileReport(df)
    report.to_file("./templates/report.html")

@app.route("/report")
def report():
    if g.user:
        model_id = request.args.get("model_id")
        model_data = cr_data.find_one({"model_id":int(model_id)})
        if model_data["username"] != session['user']:
            return render_template("error.html", username = session['user'],msg = "Model not found with the ID")
        input_path = model_data["input_path"]
        output_path = model_data["output_path"]
        generate_report(input_path, output_path)
        return render_template("profiling-report.html", model_id = model_id, username = session['user'])
    return redirect(url_for('login'))

@app.route('/render-report')
def render_report():
    return render_template("report.html")

@app.route("/deploy", methods=["GET", "POST"])
def deploy():
    if g.user:
        if request.method == "GET":
            model_id = request.args.get("model_id")
            model_data = cr_data.find_one({"model_id":int(model_id)})
            deploy = model_data["deploy"]
            if deploy == "no":
                return render_template("deploy.html", model_id = model_id)
            return render_template("deploy.html", status = "true", model_id = model_id)
        model_id = request.args.get("model_id")
        status = request.args.get("status")
        cr_data.update_one({"model_id":int(model_id)},{"$set":{
            "deploy":status
            }})
        if status == "no":
            return render_template("deploy.html", model_id = model_id)
        else:
            return render_template("deploy.html", status = "true", model_id = model_id)
    return redirect(url_for('login'))

@app.route("/classify")
def classify():
    if g.user:
        model_id = request.args.get("model_id")
        return render_template("classify.html", username = session["user"], model_id = model_id)
    return redirect(url_for("index"))

@app.route("/regression")
def regression():
    if g.user:
        model_id = request.args.get("model_id")
        return render_template("regression.html", username = session["user"], model_id = model_id)
    return redirect(url_for("index"))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return render_template('login.html')

@app.route('/upload', methods=["POST", "GET"])
def upload():
    global classes
    if g.user:
        global path
        username = session['user']
        inputs = request.files["inputs"]
        fname = inputs.filename
        print(os.path)
        input_path = os.path.join('data',fname)
        inputs.save(input_path)

        outputs = request.files["outputs"]
        fname = outputs.filename
        print(os.path)
        output_path = os.path.join('data',fname)
        outputs.save(output_path)

        model_type = request.args.get("model_type")
        model_id = request.args.get("model_id")
        cr_data.update_one({"model_id":int(model_id)},{"$set":{
        "input_path":input_path,
        "output_path":output_path,
        }})

        #Return to the upload page 
        page = model_type + ".html"


        classes = "Data Description"


        return render_template(page, toast = "success", username=username, classes=classes, model_id = model_id, model_type = model_type)
    return redirect(url_for('login'))

@app.route("/train", methods=["POST"])
def train():
    # model_name means algorithm name
    model_name = request.form["model"]
    model_id = request.args.get("model_id")
    model_type = request.args.get("model_type")
    model_data = cr_data.find_one({"model_id":int(model_id)})
    input_data = model_data["input_path"]
    output_data = model_data["output_path"]
    model_train(input_data, output_data, model_id, model_name, model_type)
    return redirect(url_for("history",model_id=model_id))

def impute_null(data):
    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype == "object":
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].mean())
    return data

def encode_data(data):
    le = LabelEncoder()
    encodings = dict()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = le.fit_transform(data[col])
            keys = le.classes_
            values = le.transform(le.classes_)
            dictionary = dict(zip(keys, [x.item() for x in values]))
            encodings[col] = dictionary
    return data, encodings

def model_train(input_data, output_data, model_id, model_name, model_type):
    X = pd.read_csv(input_data)
    y = pd.read_csv(output_data)

    X = impute_null(X)
    X, encodings = encode_data(X)

    #Getting meta data
    parameters = [col for col in X.columns]
    inputs_count = len(parameters)
    output_name = y.columns[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    metrics = dict()
    if model_type == "classify":
        if model_name == "logistic":
            model = LogisticRegression()
        elif model_name == "decision_tree":
            model = DecisionTreeClassifier()
        elif model_name == "random_forest":
            model = RandomForestClassifier()
        elif model_name == "svm":
            model = SVC()
        else:
            model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        y_train = np.reshape(y_train, (y_train.shape[0], ))
        y_pred = np.reshape(y_pred, (y_pred.shape[0], ))
        acc = accuracy_score(y_train, y_pred)
        prec = precision_score(y_train, y_pred, average='weighted')
        rec = recall_score(y_train, y_pred, average='weighted')
        f1 = f1_score(y_train, y_pred, average='weighted')

        metrics["acc"] = acc.item()
        metrics["prec"] = prec.item()
        metrics["rec"] = rec.item()
        metrics["f1"] = f1.item()
        
    else:
        if model_name == "linear_reg":
            model = LinearRegression()
        elif model_name == "decision_tree_reg":
            model = DecisionTreeRegressor()
        elif model_name == "random_forest_reg":
            model = RandomForestRegressor()
        elif model_name == "svm_reg":
            model = SVR()
        else:
            model = KNeighborsRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_train, y_pred)
        metrics["mae"] = mae.item()
        metrics["mse"] = mse.item()
        metrics["rmse"] = rmse.item()
        metrics["r2"] = r2.item()

        
    
    model_path = "models/"+"model_"+str(model_id)+".sav"
    joblib.dump(model, model_path)
    cr_data.update_one({"model_id":int(model_id)},{"$set":{
        "model_file":model_path,
        "model_name":model_name, 
        "parameters":parameters,
        "inputs_count":inputs_count,
        "output_name": output_name,
        "encodings":encodings,
        "metrics":metrics
        }})
    return

def parse_encodings(encodings):
    params = list(encodings.keys())
    values = list(encodings.values())
    params_values = []
    for p,v in zip(params, values):
        params_values.append([p,v])
    return params_values

@app.route("/try_model", methods=["POST","GET"])
def try_model():
    if g.user:
        if request.method == "GET":
            model_id = request.args.get("model_id")
            model_data = cr_data.find_one({"model_id":int(model_id)})
            if model_data["username"] != session['user']:
                return render_template("error.html", username = session['user'],msg = "Model not found with the ID")
            inputs_data = model_data["parameters"]
            output_name = model_data["output_name"]
            try:
                encodings = model_data["encodings"]
                encodings_parsed = parse_encodings(encodings)
            except:
                encodings_parsed = None
            return render_template("try_model.html", inputs_data = inputs_data, output_name = output_name, model_id = model_id, encodings = encodings_parsed)
        model_id = request.args.get("model_id")
        model_data = cr_data.find_one({"model_id":int(model_id)})
        model_file = model_data["model_file"]
        values = list(request.form.values())
        values = [float(x) for x in values]
        model = joblib.load(model_file)
        prediction = model.predict([values])
        if model_data["username"] != session['user']:
            return render_template("error.html", username = session['user'],msg = "Model not found with the ID")
        inputs_data = model_data["parameters"]
        output_name = model_data["output_name"]
        try:
            encodings = model_data["encodings"]
            encodings_parsed = parse_encodings(encodings)
        except:
            encodings_parsed = None
        return render_template("try_model.html",prediction = str(prediction), inputs_data = inputs_data, output_name = output_name, model_id = model_id, encodings = encodings_parsed)
    return redirect(url_for('login'))

@app.route("/deployments", methods=["POST","GET"])
def deployments():
    if request.method == "GET":
        model_id = request.args.get("model_id")
        model_data = cr_data.find_one({"model_id":int(model_id)})
        if model_data["deploy"] == "no":
            return render_template("error.html",msg = "No deployment found")
        inputs_data = model_data["parameters"]
        output_name = model_data["output_name"]
        try:
            encodings = model_data["encodings"]
            encodings_parsed = parse_encodings(encodings)
        except:
            encodings_parsed = None
        return render_template("deployments.html", inputs_data = inputs_data, output_name = output_name, model_id = model_id, encodings = encodings_parsed)
    model_id = request.args.get("model_id")
    model_data = cr_data.find_one({"model_id":int(model_id)})
    model_file = model_data["model_file"]
    values = list(request.form.values())
    values = [float(x) for x in values]
    model = joblib.load(model_file)
    prediction = model.predict([values])
    if model_data["deploy"] == "no":
        return render_template("error.html",msg = "No deployment found")
    inputs_data = model_data["parameters"]
    output_name = model_data["output_name"]
    try:
        encodings = model_data["encodings"]
        encodings_parsed = parse_encodings(encodings)
    except:
        encodings_parsed = None
    return render_template("deployments.html",prediction = str(prediction), inputs_data = inputs_data, output_name = output_name, model_id = model_id, encodings = encodings_parsed)

@app.route("/history")
def history():
    data = []
    if g.user:
        for x in cr_data.find({"username":session["user"]}):
            data.append([
                x["model_id"],
                x["project_name"],
                x["date"]
            ])
    
        return render_template("history.html",data = data,username=session['user'])
    return redirect(url_for('login'))

@app.route("/metrics")
def metrics():
    if g.user:
        model_id = request.args.get("model_id")
        model_data = cr_data.find_one({"model_id":int(model_id)})
        metrics_data = model_data["metrics"]
        task_type = model_data["task_type"]
        model_type = task_type
        model_name = model_data["model_name"]
        if task_type == "classification":
            return render_template("metrics.html", model_id = model_id, classify="true", model_type = model_type, model_name = model_name, metrics_data = metrics_data)
        return render_template("metrics.html", model_id = model_id, model_type = model_type, model_name = model_name, metrics_data = metrics_data)
    return redirect(url_for('login'))

@app.route("/model-history")
def model_history():
    if g.user:
        model_id = request.args.get("model_id")
        model_data = cr_data.find_one({"model_id":int(model_id)})
        if model_data["username"] != session['user']:
            return render_template("error.html", username = session['user'],msg = "Model not found with the ID")
        project_name = model_data["project_name"]
        createdby = model_data["username"]
        model_type = model_data["task_type"]
        params = model_data["parameters"]
        output_name = model_data["output_name"]
        model_name = model_data["model_name"]
        date = model_data["date"]
        return render_template("model-history.html",date = date, model_name = model_name, username = session['user'], model_id = model_id, project_name = project_name, createdby = createdby, model_type = model_type, params = params, output_name = output_name)
    return redirect(url_for('login'))

@app.route("/download")
def download():
    model_id = request.args.get("model_id")
    data_type = request.args.get("data")
    model_data = cr_data.find_one({"model_id":int(model_id)})
    if model_data["username"] != session['user']:
        return render_template("error.html", username = session['user'],msg = "Model not found with the ID")
    if data_type == "input":
        return send_file(model_data["input_path"])
    elif data_type == "output":
        return send_file(model_data["output_path"])
    return send_file(model_data["model_file"])

@app.route('/sample-download')
def sample_download():
    if g.user:
        data = [
            "cancer_input.csv","cancer_output.csv",
            "drug_inputs.csv","drug_outputs.csv",
            "sales_input.csv","sales_output.csv",
            "wineQuality_inputs.csv","wineQuality_outputs.csv"
        ]
        data_id = request.args.get("id")
        path = "sample_data/" + data[int(data_id)]
        return send_file(path)
    return redirect(url_for('login'))

@app.route('/sample-data')
def sample_data():
    if g.user:
        ids = [
            ["Cancer Prediction",0,1],
            ["Drugs Classification",2,3],
            ["Sales Prediction",4,5],
            ["Wine Quality Prediction",6,7]
        ]
        return render_template("sample-data.html",data=ids, username = session['user'])
    return redirect(url_for('login'))

@app.route("/clear", methods=["POST", "GET"])
def clear():
    if request.method == "POST":
        counter.update_one({"type":"model"},{"$set":{"count":0}})
        cr_data.delete_many({})
        return redirect(url_for('login'))
    return render_template("cleardb.html")

def debug(s):
    print("*"*10)
    print(type(s))
    print(s)
    print("*"*10)


if __name__ == "__main__":
    print("running.....")
    app.run(port=5000)