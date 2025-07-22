import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
from flask import Flask,render_template,request,jsonify
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallbacksecret')

df = pd.read_csv('data/house.csv')

df.drop('id',axis=1,inplace=True)

df['Bedrooms'].fillna(df['Bedrooms'].mode()[0],inplace=True)
df['Baths'].fillna(df['Baths'].mode()[0],inplace=True)

undefine=df.isnull().sum()
print(undefine)

X=df[['Location','Area','Bedrooms','Baths']]
y=df['Price']

le = LabelEncoder()
X['Location']=le.fit_transform(X['Location'])

model = LinearRegression()
model.fit(X,y)

print('B0',model.intercept_)
print('B1,B2,B3,B4',model.coef_)

@app.route('/')
def home():
    return render_template('form.html')

location = None
area = None
badroomes = None
baths = None
loc_array = None
answer=None

@app.route('/submit', methods=['POST'])
def submit():
    global location, area, badroomes, baths, loc_array,answer
    location = request.form['location']
    loc_array = np.array([[location]])
    # Encode the location
    le = LabelEncoder()
    loc_array = le.fit_transform(loc_array.ravel())
    a=loc_array[0]
    area = request.form['area']
    badroomes = request.form['badroomes']
    baths = request.form['baths']
    all_data = np.array([[a,area,badroomes,baths]])
    numeric_data = all_data.astype(int)
    print(numeric_data)
    predict = model.predict(numeric_data)
    answer = round(predict[0],2)
    return  render_template('form.html',answer=answer)


@app.route('/show')
def show():
    # Reading global vars only — no need for global keyword here
    return (
        f"<b>Location:</b> {location}<br>"
        f"<b>Area:</b> {area}<br>"
        f"<b>Bedrooms:</b> {badroomes}<br>"
        f"<b>Baths:</b> {baths}<br>"
        f"<b>loc_array:</b> {loc_array[0]}"
        f"<b>loc_array:</b> {answer}"
    )

@app.route('/get',methods=['GET'])
def predicted_data():
    return jsonify({'prediction': float(answer) if answer is not None else "No prediction yet"})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv('DEBUG') == 'True')
