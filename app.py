from flask import Flask, render_template,request,url_for, jsonify
import pandas, pickle
import fuzzywuzzy
from fuzzywuzzy import process
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)



@app.route("/")
def index():
    loc=pandas.read_csv('E:/python/bangalorehouseprediction/Locations.csv')
    locations=loc.iloc[:,0].values
    return render_template("index.html",locations=locations)


@app.route("/price", methods=['GET','POST'])
def price():
    if request.method=='POST':
        loca= str(request.form["locat"])
        bath= int(request.form["bath"])
        bhk= int(request.form["bhk"])
        area= request.form["area"]
        balcony= int(request.form["balcony"])
        data=pandas.read_csv('E:/python/bangalorehouseprediction/FinalData.csv')
        df=data.groupby('location').agg('mean')
        if loca not in data['location'].unique():
            matches = fuzzywuzzy.process.extract(loca, data['location'].unique(),limit=20, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
            close_matches = [matches[0] for matches in matches if matches[1] >= 20]
            loc1=close_matches[0]
            pps=df.loc[loc1,'price_per_sqft']
        else:
            pps=df.loc[loca,'price_per_sqft']
        if(area==''):
            df2=data.groupby('bhk').agg('mean')
            area=df2.loc[bhk,'sqft_area']
        else:
            area=int(area)
        a={'location':[loca],
        'bath':[bath],
        'balcony':[balcony],
        'bhk':[bhk],
        'sqft_area':[area],
        'price_per_sqft':[pps]}
        b=pandas.DataFrame(a)
        randomforest=pickle.load(open('E:/python/bangalorehouseprediction/randomforest.sav', 'rb'))
        lasso=pickle.load(open('E:/python/bangalorehouseprediction/lasso.sav', 'rb'))
        ols=pickle.load(open('E:/python/bangalorehouseprediction/ols.sav', 'rb'))
        price=randomforest.predict(b)
        if price>=1000:
            price=(lasso.predict(b)+ols.predict(b))/2
        return jsonify({'price':round(price[0],2)})
 


if __name__ == '__main__':
    app.run(debug=True)

