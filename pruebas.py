import numpy as np
import pandas as pd
import time
import json
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.stats.multicomp as mc
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

##########################################################
##########################################################
### Funciones para pruebas estadísticas y preprocesado ###

# Función que cambia laborable a binario
# Recibe el dataframe y una lista con los días festivos que contiene
# Devuelve el dataframe actualizado
def laborable_binario(df, festivos):
    df['laborable'] = 1

    df.loc[df['dia'] == 'sabado', 'laborable'] = 0
    df.loc[df['dia'] == 'domingo', 'laborable'] = 0

    for f in festivos:
        df.loc[df['fecha'] == f, 'laborable'] = 0

    return df

# Función que cambia laborable a las categorías laborable, viernes, sabado, domingo y festivo
# Recibe el dataframe y una lista con los días festivos que contiene
# Devuelve el dataframe actualizado
def laborable_lvsdf(df, festivos):
    df['laborable'] = 'laborable'

    df.loc[df['dia'] == 'viernes', 'laborable'] = 'viernes'
    df.loc[df['dia'] == 'sabado', 'laborable'] = 'sabado'
    df.loc[df['dia'] == 'domingo', 'laborable'] = 'domingo'

    for f in festivos:
        df.loc[df['fecha'] == f, 'laborable'] = 'festivo'

    return df

# Función que cambia laborable a las categorías laborable, festivo y prefestivo
# Recibe el dataframe y dos listas con los días festivos y prefestivos que contiene
# Devuelve el dataframe actualizado
def laborable_lfp(df, festivos, prefestivos):
    df['laborable'] = 'laborable'

    df.loc[df['dia'] == 'viernes', 'laborable'] = 'prefestivo'
    df.loc[df['dia'] == 'sabado', 'laborable'] = 'prefestivo'
    df.loc[df['dia'] == 'domingo', 'laborable'] = 'festivo'

    for f in festivos:
        df.loc[df['fecha'] == f, 'laborable'] = 'festivo'

    for f in prefestivos:
        df.loc[df['fecha'] == f, 'laborable'] = 'prefestivo'

    return df

# Función que cambia laborable a las categorías laborable, festivo, contando los prefestivos como laborables
# Recibe el dataframe y una lista con los días festivos que contiene
# Devuelve el dataframe actualizado
def laborable_lf(df, festivos):
    df['laborable'] = 'laborable'

    df.loc[df['dia'] == 'sabado', 'laborable'] = 'festivo'
    df.loc[df['dia'] == 'domingo', 'laborable'] = 'festivo'

    for f in festivos:
        df.loc[df['fecha'] == f, 'laborable'] = 'festivo'

    return df

# Función que aplica clustering al dataframe
# Recibe el dataframe, el número de clusters y sobre qué variables se quiere aplicar
# Devuelve el dataframe actualizado
def aplicar_clustering(df, n_clusters, variables):
    clustering = KMeans(n_clusters=n_clusters).fit(df[variables])
    KMeans()
    df['grupo'] = clustering.labels_

    return df

# Calcula el test de Fischer sobre los grupos generados por clustering teniendo en cuenta laborable
# Recibe el dataframe
# Devuelve el p-value obtenido
def t_fischer(df):
    data_crosstab = pd.crosstab(df['grupo'], df['laborable'], margins=False)
    oddsratio, pvalue = stats.fisher_exact(data_crosstab)

    return pvalue

# Calcula el chi2 sobre los grupos generados por clustering teniendo en cuenta laborable
# Recibe el dataframe
# Devuelve el p-value obtenido
def t_chi2(df):
    data_crosstab = pd.crosstab(df['grupo'], df['laborable'], margins=False)
    g, p, dof, expctd = chi2_contingency(data_crosstab, lambda_="log-likelihood")

    return p

# Dibuja un boxplot
# Recibe el dataframe y las variables con las que hacerlo
def boxplot(df, variable_x, variable_y):
    bp = df.boxplot(column=variable_y, by=variable_x, grid=False)
    plt.show()

# Calcula el test de Student para las categorías que se le pasan como parámetro
# Recibe las listas con los datos para cada categoría
# Devuelve el p-value
def t_student(categoria1, categoria2):
    t, p = stats.ttest_ind(categoria1, categoria2)

    return p

# Calcula ANOVA para las categorías que se le pasan como parámetro
# Recibe las listas con los datos para cada categoría
# Devuelve el p-value y la tabla de la comparación
def t_anova(df, variable, *argv):
    F, p = stats.f_oneway(*argv)

    comp = mc.MultiComparison(df['uso'], df[variable])
    tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method="bonf")

    return p, tbl

# Calcula la correlación de Pearson para las variables del dataframe
# Recibe el dataframe y las variables para las que se quiere calcular la correlación
# Devuelve el valor de correlación
def corr_pearson(df, variable1, variable2):
    corr, p = pearsonr(df[variable1], df[variable2])

    return corr, p

# Dibuja una gráfica con la media del uso en el eje y
# Recibe el dataframe y la otra variable
def grafica_uso_y(df, variable):
    x = df[variable]
    x = list(x.drop_duplicates())
    x.sort()

    y = [df.loc[df[variable] == e, 'uso'].mean() for e in x]

    if variable == 'Wind':
        x = [e*3.6 for e in x]
    elif variable == 'Temperature':
        x = [e-273.15 for e in x]

    plt.plot(x, y)

    uds = ''
    if variable == 'Temperature':
        uds = ' (ºC)'
    elif variable == 'Wind':
        uds = ' (km/h)'
    elif variable == 'Precipitation':
        uds = ' (m)'

    plt.xlabel(variable + uds)

    plt.ylabel('uso (%)')
    plt.show()

# Dibuja una gráfica con la media de la variable para cada valor de uso
# Recibe el dataframe y la otra variable
def grafica_uso_x(df, variable):
    x = df['uso']
    x = list(x.drop_duplicates())
    x.sort()

    y = [df.loc[df['uso'] == e, variable].mean() for e in x]

    plt.plot(x, y)
    plt.xlabel('uso (%)')
    plt.ylabel(variable)
    plt.show()

# Dibuja una gráfica con la media de la variable y en el eje y y la variable x en el eje x
# Recibe el dataframe y las variables
def grafica(df, variableX, variableY):
    x = df[variableX]
    x = list(x.drop_duplicates())
    x.sort()

    y = [df.loc[df[variableX] == e, variableY].mean() for e in x]

    plt.plot(x, y)
    plt.xlabel(variableX)
    plt.ylabel(variableY)
    plt.show()

# Convierte una columna del dataframe a one hot encoding
# Recibe el dataframe, la variable que se quiere convertir y el prefijo que se quiere usar
# Devuelve el dataframe actualizado
def one_hot(df, variable, prefijo):
    df_one_hot = pd.get_dummies(df[variable], prefix=prefijo)
    df = df.join(df_one_hot)

    return df

# Normaliza las variables y las añade al dataframe con el sufijo "_norm"
# Recibe el dataframe y la lista de variables a normalizar
# Devuelve el dataframe actualizado
def normalizar(df, variables):
    scaler = MinMaxScaler()

    for v in variables:
        df[v + '_norm'] = scaler.fit_transform(df[[v]])

    return df

# Convierte una variable periódica a su seno (_sin) y coseno (_cos) para utilizarla en un modelo de predicción
# Recibe el dataframe, la variable y el periodo que sigue (p.e. 24 en el caso de las horas)
# Devuelve el dataframe actualizado
def var_periodica(df, variable, periodo):
    df[variable + '_sin'] = np.sin(df[variable] * (2. * np.pi / periodo))
    df[variable + '_cos'] = np.cos(df[variable] * (2. * np.pi / periodo))

    return df

# Elimina outliers mediante z score
# Recibe el dataframe y la variable que se quiere eliminar outliers
# Devuelve el dataframe sin las filas que son outliers
def zscore(df, variable):
    z = np.abs(stats.zscore(df[variable]))
    df['z'] = z
    df = df.loc[df['z'] < 3]
    df = df.drop(columns='z')

    return df

######################################################
######################################################
### Funciones para el entrenamiento de los modelos ###

# Entrena con regresión lineal y devuelve el MSE y el R2 en entrenamiento y test
# Recibe las variables y el target tanto para entrenamiento como para test
# Devuelve el MSE y el R2 de train y test
def regr_mlr(X_train, X_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    y_train_pred = regr.predict(X_train)

    mse_t = mean_squared_error(y_train, y_train_pred)
    r2_t = r2_score(y_train, y_train_pred)

    mse_v = mean_squared_error(y_test, y_pred)
    r2_v = r2_score(y_test, y_pred)

    return mse_t, r2_t, mse_v, r2_v

# Entrena con regresión polinómica y devuelve el MSE y el R2 en entrenamiento y test
# Recibe las variables y el target tanto para entrenamiento como para test y el grado (si no se incluye se elige grado 4 por defecto)
# Devuelve el MSE y el R2 de train y test
def regr_pr(X_train, X_test, y_train, y_test, grado=4):
    poly_reg = PolynomialFeatures(degree=grado)
    regr = linear_model.LinearRegression()
    regr.fit(poly_reg.fit_transform(X_train), y_train)
    y_pred = regr.predict(poly_reg.fit_transform(X_test))

    y_train_pred = regr.predict(poly_reg.fit_transform(X_train))

    mse_t = mean_squared_error(y_train, y_train_pred)
    r2_t = r2_score(y_train, y_train_pred)

    mse_v = mean_squared_error(y_test, y_pred)
    r2_v = r2_score(y_test, y_pred)

    return mse_t, r2_t, mse_v, r2_v

# Entrena con SVM y devuelve el MSE y el R2 en entrenamiento y test
# Recibe las variables y el target tanto para entrenamiento como para test
# Devuelve el MSE y el R2 de train y test
def regr_svm(X_train, X_test, y_train, y_test):
    regr = SVR()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    y_train_pred = regr.predict(X_train)

    mse_t = mean_squared_error(y_train, y_train_pred)
    r2_t = r2_score(y_train, y_train_pred)

    mse_v = mean_squared_error(y_test, y_pred)
    r2_v = r2_score(y_test, y_pred)

    return mse_t, r2_t, mse_v, r2_v

# Entrena con Random Forest y devuelve el MSE y el R2 en entrenamiento y test
# Recibe las variables y el target tanto para entrenamiento como para test, así como los diferentes parámetros (si no se incluyen se eligen los por defecto)
# Devuelve el MSE y el R2 de train y test
def regr_rf(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, max_features='auto'):
    regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    y_train_pred = regr.predict(X_train)

    mse_t = mean_squared_error(y_train, y_train_pred)
    r2_t = r2_score(y_train, y_train_pred)

    mse_v = mean_squared_error(y_test, y_pred)
    r2_v = r2_score(y_test, y_pred)

    return mse_t, r2_t, mse_v, r2_v

# Entrena con AdaBoost y devuelve el MSE y el R2 en entrenamiento y test
# Recibe las variables y el target tanto para entrenamiento como para test
# Devuelve el MSE y el R2 de train y test
def regr_ab(X_train, X_test, y_train, y_test):
    regr = AdaBoostRegressor()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    y_train_pred = regr.predict(X_train)

    mse_t = mean_squared_error(y_train, y_train_pred)
    r2_t = r2_score(y_train, y_train_pred)

    mse_v = mean_squared_error(y_test, y_pred)
    r2_v = r2_score(y_test, y_pred)

    return mse_t, r2_t, mse_v, r2_v

# Entrena con XGBoost y devuelve el MSE y el R2 en entrenamiento y test
# Recibe las variables y el target tanto para entrenamiento como para test, así como los diferentes parámetros (si no se incluyen se eligen los por defecto)
# Devuelve el MSE y el R2 de train y test
def regr_xgb(X_train, X_test, y_train, y_test, max_depth=6, gamma=0, min_child_weight=1, subsample=1, colsample_bytree=1):
    regr = XGBRegressor(max_depth=max_depth, gamma=gamma, min_child_weight=min_child_weight,
                        subsample=subsample, colsample_bytree=colsample_bytree)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    y_train_pred = regr.predict(X_train)

    mse_t = mean_squared_error(y_train, y_train_pred)
    r2_t = r2_score(y_train, y_train_pred)

    mse_v = mean_squared_error(y_test, y_pred)
    r2_v = r2_score(y_test, y_pred)

    return mse_t, r2_t, mse_v, r2_v

#################################
#################################
######## Ejemplos de uso ########
a = pd.read_csv('dataframe.csv')

### Laborable
a = laborable_binario(a, ['24/12/2019', '25/12/2019', '01/01/2020', '06/01/2020', '12/10/2020', '02/11/2020'])
a = laborable_lvsdf(a, ['24/12/2019', '25/12/2019', '01/01/2020', '06/01/2020', '12/10/2020', '02/11/2020'])
a = laborable_lfp(a, ['25/12/2019', '01/01/2020', '06/01/2020', '12/10/2020', '02/11/2020'],
                  ['22/12/2019', '24/12/2019', '31/12/2019', '05/01/2020', '11/10/2020', '01/11/2020'])
a = laborable_lf(a, ['24/12/2019', '25/12/2019', '01/01/2020', '06/01/2020', '12/10/2020', '02/11/2020'])

### Clustering
a = aplicar_clustering(a, 5, ['uso', 'Temperature', 'Wind', 'Precipitation'])
pvalue = t_fischer(a)
pvalue = t_chi2(a)

### Boxplot, test de Student y ANOVA
boxplot(a, 'laborable', 'uso')
pvalue = t_student(a.loc[a['laborable'] == 1, 'uso'], a.loc[a['laborable'] == 0, 'uso'])
pvalue, tbl = t_anova(a, 'laborable', a.loc[a['laborable'] == 'festivo', 'uso'],
                      a.loc[a['laborable'] == 'prefestivo', 'uso'], a.loc[a['laborable'] == 'laborable', 'uso'])

### Correlación y gráficas
corr, pvalue = corr_pearson(a, 'uso', 'CO')
grafica_uso_y(a, 'Temperature')
grafica_uso_x(a, 'CO')

### One hot encoding, normalización y variables periódicas
a = one_hot(a, 'laborable', 'l')
a = normalizar(a, ['CO', 'NO', 'NO2', 'SO2', 'O3', 'PM10', 'C6H6'])
a = var_periodica(a, 'hora', 24)

### Eliminar outliers
a = zscore(a, 'porcentajes')

########################################
########################################
### Código común a todas las pruebas ###

# Lectura de datos
dfSevilla = pd.read_csv('dfTrainSevilla.csv')
dfBCN = pd.read_csv('dfTrainBCN.csv')

# Preprocesado
festivosSevilla = ['25/12/2019', '01/01/2020', '06/01/2020', '12/10/2020', '02/11/2020']
prefestivosSevilla = ['22/12/2019', '24/12/2019', '31/12/2019', '05/01/2020', '11/10/2020', '01/11/2020']
dfSevilla = laborable_lfp(dfSevilla, festivosSevilla, prefestivosSevilla)
dfSevilla = one_hot(dfSevilla, 'laborable', 'l')
dfSevilla = var_periodica(dfSevilla, 'hora', 24)

festivosBCN = ['19/04/2019', '11/09/2019', '25/12/2019', '25/12/2020', '26/12/2020', '01/01/2021']
dfBCN = laborable_lf(dfBCN, festivosBCN)
dfBCN = one_hot(dfBCN, 'laborable', 'l')
dfBCN = var_periodica(dfBCN, 'hora', 24)

ciudades = ['Sevilla', 'Barcelona']

lX = {'Sevilla': [dfSevilla[['hora_sin', 'hora_cos',
                             'l_laborable', 'l_festivo', 'l_prefestivo',
                             'Temperature', 'Wind', 'Precipitation']],
                  dfSevilla[['hora_sin', 'hora_cos',
                             'l_laborable', 'l_festivo', 'l_prefestivo',
                             'Temperature', 'Wind', 'Precipitation',
                             'CO', 'NO', 'NO2', 'SO2', 'O3', 'PM10', 'C6H6']]],
      'Barcelona': [dfBCN[['hora_sin', 'hora_cos',
                           'l_laborable', 'l_festivo',
                           'Temperature', 'Wind', 'Precipitation']],
                    dfBCN[['hora_sin', 'hora_cos',
                           'l_laborable', 'l_festivo',
                           'Temperature', 'Wind', 'Precipitation',
                           'CO', 'NO', 'NO2', 'SO2', 'O3', 'PM10']]]}

ly = {'Sevilla': dfSevilla['uso'],
      'Barcelona': dfBCN['uso']}

########################################
########################################
### Prueba con todos los algoritmos ###

algoritmo = [regr_mlr, regr_pr, regr_svm, regr_rf, regr_ab, regr_xgb]

n_rep = 100

dfr = pd.DataFrame(columns=['algoritmo', 'ciudad', 'gases',
                            'R2 TRAIN', 'MSE TRAIN', 'R2 TEST', 'MSE TEST', 'time'])

for func, name in zip(algoritmo, ['MLR', 'PR', 'SVM', 'RF', 'AB', 'XGB']):
    for ciudad in ciudades:
        y = ly[ciudad]
        for X, g in zip(lX[ciudad], ['no', 'si']):
            mses_t = []
            r2s_t = []
            mses_v = []
            r2s_v = []

            start_time = time.time()
            for i in range(0, n_rep):
                print(ciudad, g, i)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                mse_t, r2_t, mse_v, r2_v = func(X_train, X_test, y_train, y_test)

                mses_t.append(mse_t)
                r2s_t.append(r2_t)
                mses_v.append(mse_v)
                r2s_v.append(r2_v)
            seconds = time.time() - start_time

            dfr = dfr.append({'algoritmo': 'RF', 'ciudad': ciudad, 'gases': g,
                              'R2 TRAIN': np.mean(r2s_t), 'MSE TRAIN': np.mean(mses_t),
                              'R2 TEST': np.mean(r2s_v), 'MSE TEST': np.mean(mses_v),
                              'time': seconds}, ignore_index=True)

# Guardar los resultados (R2, MSE) y el tiempo en un fichero
dfr.to_csv('resultadosPruebas.csv')
print(dfr)

########################################
########################################
### Pruebas con regresión polinómica ###

grados = [2, 3, 4, 7]

n_rep = 100

dfr = pd.DataFrame(columns=['algoritmo', 'ciudad', 'gases',
                            'grado',
                            'R2 TRAIN', 'MSE TRAIN', 'R2 TEST', 'MSE TEST', 'time'])

for grado in grados:
    for ciudad in ciudades:
        y = ly[ciudad]
        for X, g in zip(lX[ciudad], ['no', 'si']):
            mses_t = []
            r2s_t = []
            mses_v = []
            r2s_v = []

            start_time = time.time()
            for i in range(0, n_rep):
                print(ciudad, g, grado, i)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                mse_t, r2_t, mse_v, r2_v = regr_pr(X_train, X_test, y_train, y_test, grado=grado)

                mses_t.append(mse_t)
                r2s_t.append(r2_t)
                mses_v.append(mse_v)
                r2s_v.append(r2_v)
            seconds = time.time() - start_time

            dfr = dfr.append({'algoritmo': 'PR', 'ciudad': ciudad, 'gases': g,
                              'grado': grado,
                              'R2 TRAIN': np.mean(r2s_t), 'MSE TRAIN': np.mean(mses_t),
                              'R2 TEST': np.mean(r2s_v), 'MSE TEST': np.mean(mses_v),
                              'time': seconds}, ignore_index=True)

# Guardar los resultados (R2, MSE) y el tiempo en un fichero
dfr.to_csv('resultadosPruebasPR.csv')
print(dfr)

#################################
#################################
### Pruebas con Random Forest ###

lmf = ['auto', 0.5, 0.3]
lmd = [None, 10, 7, 5]
lne = [100, 500, 1000]

n_rep = 100

dfr = pd.DataFrame(columns=['algoritmo', 'ciudad', 'gases',
                            'max_features', 'max_depth', 'n_estimators',
                            'R2 TRAIN', 'MSE TRAIN', 'R2 TEST', 'MSE TEST', 'time'])

for ciudad in ciudades:
    y = ly[ciudad]
    for X, g in zip(lX[ciudad], ['no', 'si']):
        for mf in lmf:
            for md in lmd:
                for ne in lne:
                    mses_t = []
                    r2s_t = []
                    mses_v = []
                    r2s_v = []

                    start_time = time.time()
                    for i in range(0, n_rep):
                        print(ciudad, g, mf, md, ne, i)

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                        mse_t, r2_t, mse_v, r2_v = regr_rf(X_train, X_test, y_train, y_test,
                                                           max_features=mf, max_depth=md, n_estimators=ne)

                        mses_t.append(mse_t)
                        r2s_t.append(r2_t)
                        mses_v.append(mse_v)
                        r2s_v.append(r2_v)
                    seconds = time.time() - start_time

                    dfr = dfr.append({'algoritmo': 'PR', 'ciudad': ciudad, 'gases': g,
                                      'max_features': mf, 'max_depth': md, 'n_estimators': ne,
                                      'R2 TRAIN': np.mean(r2s_t),'MSE TRAIN': np.mean(mses_t),
                                      'R2 TEST': np.mean(r2s_v), 'MSE TEST': np.mean(mses_v),
                                      'time': seconds}, ignore_index=True)

# Guardar los resultados (R2, MSE) y el tiempo en un fichero
dfr.to_csv('resultadosPruebasRF.csv')
print(dfr)

###########################
###########################
### Pruebas con XGBoost ###

lmd = [6, 3, 2]
lgm = [0, 10, 15, 30, 50]
lmcw = [1, 5, 10]
lss = [1, 0.5]
lcsbt = [1, 0.5]

n_rep = 100

dfr = pd.DataFrame(columns=['algoritmo', 'ciudad', 'gases',
                            'max_depth', 'gamma', 'min_child_weight', 'subsample', 'colsample_bytree',
                            'R2 TRAIN', 'MSE TRAIN', 'R2 TEST', 'MSE TEST', 'time'])

for ciudad in ciudades:
    y = ly[ciudad]
    for X, g in zip(lX[ciudad], ['no', 'si']):
        for md in lmd:
            for gm in lgm:
                for mcw in lmcw:
                    for ss in lss:
                        for csbt in lcsbt:
                            mses_t = []
                            r2s_t = []
                            mses_v = []
                            r2s_v = []

                            start_time = time.time()
                            for i in range(0, n_rep):
                                print(ciudad, g, md, gm, mcw, ss, csbt, i)

                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                                mse_t, r2_t, mse_v, r2_v = regr_xgb(X_train, X_test, y_train, y_test,
                                                                    max_depth=md, gamma=gm, min_child_weight=mcw,
                                                                    subsample=ss, colsample_bytree=csbt)

                                mses_t.append(mse_t)
                                r2s_t.append(r2_t)
                                mses_v.append(mse_v)
                                r2s_v.append(r2_v)
                            seconds = time.time() - start_time

                            dfr = dfr.append({'algoritmo': 'PR', 'ciudad': ciudad, 'gases': g,
                                              'max_depth': md, 'gamma': gm, 'min_child_weight': mcw,
                                              'subsample': ss, 'colsample_bytree': csbt,
                                              'R2 TRAIN': np.mean(r2s_t), 'MSE TRAIN': np.mean(mses_t),
                                              'R2 TEST': np.mean(r2s_v), 'MSE TEST': np.mean(mses_v),
                                              'time': seconds}, ignore_index=True)

# Guardar los resultados (R2, MSE) y el tiempo en un fichero
dfr.to_csv('resultadosPruebasXGB.csv')
print(dfr)

###########################
###########################
##### Pruebas finales #####

configSevillaRF = {'n_estimators': 500, 'max_features': 0.5, 'max_depth': None}
configBCNXGB = {'max_depth': 6, 'gamma': 50, 'min_child_weight': 10, 'subsample': 1, 'colsample_bytree': 1}

dfSevilla_Train = pd.read_csv('dfTrainSevilla.csv')
dfSevilla_Test = pd.read_csv('dfTestSevilla.csv')
dfBCN_Train = pd.read_csv('dfTrainBCN.csv')
dfBCN_Test = pd.read_csv('dfTestBCN.csv')

dfr = pd.DataFrame(columns=['ciudad', 'R2 Train', 'MSE Train', 'R2 Test', 'MSE Test'])

print('Sevilla')

X_train_Sevilla = dfSevilla_Train[['hora_sin', 'hora_cos',
                                   'l_laborable', 'l_festivo', 'l_prefestivo',
                                   'Temperature', 'Wind', 'Precipitation',
                                   'CO', 'NO', 'NO2', 'SO2', 'O3', 'PM10', 'C6H6']]

y_train_Sevilla = dfSevilla_Train['uso']

X_test_Sevilla = dfSevilla_Test[['hora_sin', 'hora_cos',
                                 'l_laborable', 'l_festivo', 'l_prefestivo',
                                 'Temperature', 'Wind', 'Precipitation',
                                 'CO', 'NO', 'NO2', 'SO2', 'O3', 'PM10', 'C6H6']]

y_test_Sevilla = dfSevilla_Test['uso']

mse_t_Sevilla, r2_t_Sevilla, mse_v_Sevilla, r2_v_Sevilla = regr_rf(X_train_Sevilla, X_test_Sevilla,
                                                                   y_train_Sevilla, y_test_Sevilla,
                                                                   n_estimators=configSevillaRF['n_estimators'],
                                                                   max_depth=configSevillaRF['max_depth'],
                                                                   max_features=configSevillaRF['max_features'])

dfr = dfr.append({'ciudad': 'Sevilla', 'R2 Train': r2_t_Sevilla, 'MSE Train': mse_t_Sevilla,
                  'R2 Test': r2_v_Sevilla, 'MSE Test': mse_v_Sevilla}, ignore_index=True)

print('Barcelona')

X_train_BCN = dfBCN_Train[['hora_sin', 'hora_cos',
                           'l_laborable', 'l_festivo',
                           'Temperature', 'Wind', 'Precipitation',
                           'CO', 'NO', 'NO2', 'SO2', 'O3', 'PM10']]

y_train_BCN = dfBCN_Train['uso']

X_test_BCN = dfBCN_Test[['hora_sin', 'hora_cos',
                         'l_laborable', 'l_festivo',
                         'Temperature', 'Wind', 'Precipitation',
                         'CO', 'NO', 'NO2', 'SO2', 'O3', 'PM10']]

y_test_BCN = dfBCN_Test['uso']

mse_t_BCN, r2_t_BCN, mse_v_BCN, r2_v_BCN = regr_xgb(X_train_BCN, X_test_BCN, y_train_BCN, y_test_BCN,
                                                    max_depth=configBCNXGB['max_depth'], gamma=configBCNXGB['gamma'],
                                                    min_child_weight=configBCNXGB['min_child_weight'],
                                                    subsample=configBCNXGB['subsample'],
                                                    colsample_bytree=configBCNXGB['colsample_bytree'])

dfr = dfr.append({'ciudad': 'Barcelona', 'R2 Train': r2_t_BCN, 'MSE Train': mse_t_BCN,
                  'R2 Test': r2_v_BCN, 'MSE Test': mse_v_BCN}, ignore_index=True)

dfr.to_csv('resultadosPruebasFinales.csv')
print(dfr)
