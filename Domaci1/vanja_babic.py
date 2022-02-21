# -*- coding: utf-8 -*-
"""
@author: babic
"""

#%% Biblioteke

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr

import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing

#%% Ucitavanje podataka 

df = pd.read_csv('SeoulBikeData.csv', encoding='latin1')
print(df.head())
print(df.shape)

y = df['Rented Bike Count'] #izlazna promenljiva za linearnu regresiju

#%% Preimenovanje naziva kolona radi lakseg rada

df.rename(columns={'Rented Bike Count':'RentedBikeCount', 'Temperature(°C)':'Temperature', 'Wind speed (m/s)':'WindSpeed(m/s)', 'Visibility (10m)':'Visibility(10m)', 'Snowfall (cm)':'Snowfall(cm)', 'Dew point temperature(°C)':'DewPointTemperature', 'Solar Radiation (MJ/m2)':'SolarRadiation(MJ/m2)', 'Functioning Day':'FunctioningDay'}, inplace=True)

#%% Izbacivanje obelezja Date i ubacivanje novih obelezja

df['Date']= pd.to_datetime(df['Date']) #konverzija u DateTime tip podatka
print(df.dtypes)

df['Mesec'] = df['Date'].dt.month #kreiranje obelezja Mesec
df['DanUNedelji'] = df['Date'].dt.dayofweek #kreiranje obelezja DanUNedelji

#%% Provera iz kojih godina su podaci, obzirom da po zadatku nadalje ne cuvamo podatke o godinama

df['Godina'] = df['Date'].dt.year
print(df['Godina'].unique())
df_godina = df.set_index('Godina').groupby('Godina').count()

#%% Izbacivanje obelezja

df.drop(['Date'], inplace= True, axis = 1) #izbacivanje obelezja Date, po zadatku
df.drop(['Godina'], inplace= True, axis = 1) #izbacivanje obelezja Godina, po zadatku

print(df.dtypes)
print(df.shape)

#%% Analiza nedostajucih podataka

print(df.isnull().sum()) #u bazi nema nedostajucih podataka

#%% Analiza obeležja

stat = df.describe()

print((df['Snowfall(cm)'] == 0).sum())
print((df['Snowfall(cm)'] == 0).sum()/len(df['Rainfall(mm)'])) #Oko 95% vrednosti je 0 za kolicinu snega

print((df['Rainfall(mm)'] == 0).sum())
print((df['Rainfall(mm)'] == 0).sum()/len(df['Rainfall(mm)'])) #Oko 94% vrednosti je 0 za kolicinu kise

print((df['SolarRadiation(MJ/m2)'] == 0).sum())
print((df['SolarRadiation(MJ/m2)'] == 0).sum()/len(df['SolarRadiation(MJ/m2)'])) #Oko 50% vrednosti je 0 za kolicinu kise

df_season = df.set_index('Seasons')
plt.figure()
plt.boxplot([df_season.loc['Autumn','Temperature'], df_season.loc['Winter','Temperature'], df_season.loc['Summer','Temperature'], df_season.loc['Spring','Temperature']]) 
plt.ylabel('Temperatura')
plt.xlabel('Godisnje doba')
plt.xticks([1, 2, 3, 4], ["Jesen", "Zima", "Leto", "Prolece"])
plt.grid()

vreme = df.filter(['Seasons', 'Temperature', 'Humidity(%)', 'WindSpeed(m/s)', 'Visibility(10m)', 'Rainfall(mm)', 'Snowfall(cm)'], axis=1)
df_godisnjeDoba = vreme.set_index('Seasons').groupby('Seasons')
df_godisnjeDoba_desc = df_godisnjeDoba.describe()

#%% Analiza obelezja RentedBikeCount i vizuelizacija promene te promenljive 

y_stat = y.describe()

#da li postoji sat tokom radnog dana a da nije iznajmljen ni jedan bicikl
df_functioningDay = df.groupby('FunctioningDay').min() 
print(df_functioningDay['RentedBikeCount'])

#prosecan broj iznajmljenih bicikala po mesecu
df_meseciMean = df.set_index('Mesec').groupby('Mesec').mean().sort_values('RentedBikeCount')
plt.figure()
fig = plt.figure(figsize = (15, 10)) 
plt.bar(df_meseciMean.index, df_meseciMean['RentedBikeCount'], width=0.8)
plt.ylabel('Prosečan broj iznajmljenih bicikala')
plt.xlabel('Mesec')
plt.show()

df_danUNedelji = df.set_index('DanUNedelji') 
plt.figure()
plt.boxplot([df_danUNedelji.loc[0,'RentedBikeCount'], df_danUNedelji.loc[1,'RentedBikeCount'], df_danUNedelji.loc[2,'RentedBikeCount'], df_danUNedelji.loc[3,'RentedBikeCount'], df_danUNedelji.loc[4,'RentedBikeCount'], df_danUNedelji.loc[5,'RentedBikeCount'], df_danUNedelji.loc[6,'RentedBikeCount']]) 
plt.ylabel('Broj iznajmljenih bicikala')
plt.xlabel('Dan u nedelji')
plt.xticks([1, 2, 3, 4, 5, 6, 7], ["Ponedeljak", "Utorak", "Sreda", "Četvrtak", "Petak", "Subota", "Nedelja"])
plt.grid()
plt.show()

df_praznik=df.loc[df['Holiday']=='Holiday', 'RentedBikeCount']
df_nijepraznik=df.loc[df['Holiday']=='No Holiday', 'RentedBikeCount']
plt.hist(df_praznik, bins=np.arange(0,500,20), alpha=0.5, label='Praznik', density=True)
plt.hist(df_nijepraznik, bins=np.arange(0,500,20), alpha=0.3, label='Nije praznik', density=True)
plt.legend()
plt.show()

rbn_meseci = df.groupby(by=['Hour']).mean()['RentedBikeCount']
plt.plot(np.arange(1, 25, 1), rbn_meseci, 'b')
plt.ylabel('Prosečan broj iznajmljenih bicikala')
plt.xlabel('Sat u toku dana')
plt.show()

#%% Korelacija

df['Seasons'].replace({"Winter":1, "Spring":2, "Summer":3, "Autumn":4}, inplace = True)
df['Holiday'].replace({"No Holiday":0, "Holiday":1}, inplace = True)

corr = df.corr()
f = plt.figure(figsize=(12, 9))
sb.heatmap(corr, annot=True);

df.plot.scatter(x='DewPointTemperature', y='Temperature', c="red")

#%% Linearna regresija

df['FunctioningDay'].replace({"No":0, "Yes":1}, inplace = True)

x = df.copy(deep=True)
x['FunctioningDay'] = x['FunctioningDay'].replace(0, np.nan)

x.dropna(inplace = True, axis=0)

x = x.drop(['FunctioningDay'], axis=1)
y = x['RentedBikeCount']

x = x.drop(['RentedBikeCount'], axis=1)

#%%

def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y, y_predicted) # np.mean((y_test-y_predicted)**2)
    mae = mean_absolute_error(y, y_predicted) # np.mean(np.abs(y_test-y_predicted))
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))

#%% Podela obelezja na trening i test skup

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

#%% Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn

# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)

# Obuka
first_regression_model.fit(x_train, y_train)

# Testiranje
y_predicted = first_regression_model.predict(x_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
print("koeficijenti: ", first_regression_model.coef_)

#%% Selekcija obelezja unazad

import statsmodels.api as sm
X = sm.add_constant(x_train)
model = sm.OLS(y_train, X.astype(float)).fit()
print(model.summary())

x_train = x_train.drop(['Mesec'], axis =1 )
x_test = x_test.drop(['Mesec'], axis =1 )
x = x.drop(['Mesec'], axis =1 )

X = sm.add_constant(x_train)
model = sm.OLS(y_train, X.astype(float)).fit()
print(model.summary())

x_train = x_train.drop(['Visibility(10m)'], axis =1 )
x_test = x_test.drop(['Visibility(10m)'], axis =1 )
x = x.drop(['Visibility(10m)'], axis =1 )

X = sm.add_constant(x_train)
model = sm.OLS(y_train, X.astype(float)).fit()
print(model.summary())

x_train = x_train.drop(['Snowfall(cm)'], axis =1 )
x_test = x_test.drop(['Snowfall(cm)'], axis =1 )
x = x.drop(['Snowfall(cm)'], axis =1 )

X = sm.add_constant(x_train)
model = sm.OLS(y_train, X.astype(float)).fit()
print(model.summary())

x_train = x_train.drop(['DewPointTemperature'], axis =1 )
x_test = x_test.drop(['DewPointTemperature'], axis =1 )
x = x.drop(['DewPointTemperature'], axis =1 )

X = sm.add_constant(x_train)
model = sm.OLS(y_train, X.astype(float)).fit()
print(model.summary())

#%% Standardizacija obelezja (svodjenje na sr.vr. 0 i varijansu 1)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
x_train_std = pd.DataFrame(x_train_std)
x_test_std = pd.DataFrame(x_test_std)
x_train_std.columns = list(x.columns)
x_test_std.columns = list(x.columns)
x_train_std.head()

#%% Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn

# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)

# Obuka
first_regression_model.fit(x_train, y_train)

# Testiranje
y_predicted = first_regression_model.predict(x_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
print("koeficijenti: ", first_regression_model.coef_)

#%%

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train)
x_inter_test = poly.transform(x_test)

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...+d1x1^2+d2x2^2+...+dnxn^2

# Inicijalizacija
regression_model_degree = LinearRegression()

# Obuka modela
regression_model_degree.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_degree.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_degree.coef_)),regression_model_degree.coef_)
print("koeficijenti: ", regression_model_degree.coef_)

#%%

poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train)
x_inter_test = poly.transform(x_test)
print(poly.get_feature_names())
# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...+d1x1^2+d2x2^2+...+dnxn^2

# Inicijalizacija
regression_model_degree = LinearRegression()

# Obuka modela
regression_model_degree.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_degree.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_degree.coef_)),regression_model_degree.coef_)
print("koeficijenti: ", regression_model_degree.coef_)

#%%
"""
poly = PolynomialFeatures(degree=4, interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train)
x_inter_test = poly.transform(x_test)

# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...+d1x1^2+d2x2^2+...+dnxn^2

# Inicijalizacija
regression_model_degree = LinearRegression()

# Obuka modela
regression_model_degree.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_degree.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_degree.coef_)),regression_model_degree.coef_)
print("koeficijenti: ", regression_model_degree.coef_)
"""
#%% Ridge regresija

# Inicijalizacija
ridge_model = Ridge(alpha=20)

# Obuka modela
ridge_model.fit(x_inter_train, y_train)

# Testiranje
y_predicted = ridge_model.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
print("koeficijenti: ", ridge_model.coef_)

#%% Lasso regresija

# Model initialization
lasso_model = Lasso(alpha=0.01)

# Fit the data(train the model)
lasso_model.fit(x_inter_train, y_train)

# Predict
y_predicted = lasso_model.predict(x_inter_test)

# Evaluation
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


#ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
print("koeficijenti: ", lasso_model.coef_)


