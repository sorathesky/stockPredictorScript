import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    with open(filename) as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('/')[0]))
            prices.append(float(row[1]))
    return

def predict_prices(dates, prices, x):
    
    svr_lin = SVR(kernel= 'linear', C=1e3, gamma = 'auto')
    svr_poly = SVR(kernel='poly', C=1e3, degree = 2, gamma = 'auto')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma = 'auto')
    dates = np.reshape(dates,(len(dates), 1))
    prices = np.reshape(prices,(len(prices), 1))
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    

    plt.scatter(dates, prices, color='black', label="Data")
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='LLLinear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Ploynomial model')
    plt.ylabel('Date')
    plt.xlabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    x = np.reshape(x,(-1, 1))
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


get_data('AAPL.csv')

predicted_price = predict_prices(dates, prices, 29)
print(predicted_price)
