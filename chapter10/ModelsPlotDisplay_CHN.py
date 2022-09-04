import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV


def OLS(in_X_train, in_y_train, in_X_oos, in_oos_data):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    reg = model.fit(in_X_train, in_y_train)

    yhat = reg.predict(in_X_oos)

    inner_oos_data = in_oos_data.copy()
    inner_oos_data['rethat'] = yhat

    datalr = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
    datalr.columns = ['Dates', 'stkcd', 'y', 'yhat']
    datalr['model'] = 'ols'

    coef_ser = pd.Series(reg.coef_, index=in_X_train.columns)
    return datalr, coef_ser


def EN(in_X_train, in_y_train, in_X_test, in_y_test, in_X_oos, in_test_data, in_oos_data):
    from sklearn.linear_model import ElasticNet
    inner_test_data = in_test_data.copy()
    alphas = [1e-6,1e-5,1e-4, 1e-3, 1e-2]
    ret_test = []
    ret_oos = []
    coef_ser = {}
    for a in alphas:
        elasticnet = ElasticNet(l1_ratio=0.5,
                                alpha=a,
                                fit_intercept=True,
                                normalize=False,
                                max_iter=1e4,
                                tol=1e-4,
                                copy_X=True,
                                random_state=123,
                                selection='cyclic')
        elasticnet.fit(in_X_train, in_y_train)

        Ytest_en = elasticnet.predict(in_X_test)
        inner_test_data['rethat'] = Ytest_en
        res1 = 1 - np.sum((inner_test_data['y'] - inner_test_data['rethat']) ** 2) / np.sum(inner_test_data['y'] ** 2)
        ret_test.append(res1)

        Yoos_en = elasticnet.predict(in_X_oos)
        inner_oos_data = in_oos_data.copy()
        inner_oos_data['rethat'] = Yoos_en

        dataen = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
        dataen.columns = ['Dates', 'stkcd', 'y', 'yhat']
        dataen['model'] = 'EN'
        dataen['AlphaValue'] = a
        coef_ser[a] = pd.Series(elasticnet.coef_, index=in_X_train.columns)
        ret_oos.append(dataen)

    loc_max = ret_test.index(max(ret_test))
    a_max = alphas[loc_max]
    elasticnet_best = ElasticNet(l1_ratio=0.5,
                                 alpha=a_max,
                                 fit_intercept=True,
                                 normalize=False,
                                 max_iter=1e4,
                                 tol=1e-4,
                                 copy_X=True,
                                 random_state=123,
                                 selection='cyclic')
    elasticnet_best.fit(in_X_train, in_y_train)
    yhat = elasticnet_best.predict(in_X_oos)

    inner_oos_data = in_oos_data.copy()
    inner_oos_data['rethat'] = yhat

    dataen = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
    dataen.columns = ['Dates', 'stkcd', 'y', 'yhat']
    return pd.concat(ret_oos), dataen, coef_ser


def Ridge_method(in_X_train, in_y_train, in_X_test, in_y_test, in_X_oos, in_test_data, in_oos_data):
    from sklearn.linear_model import Ridge
    inner_test_data = in_test_data.copy()

    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2]
    ret_test = []
    ret_oos = []
    coef_ser = {}
    for a in alphas:
        ridge = Ridge(
            alpha=a,
            fit_intercept=True,
            normalize=False,
            max_iter=1e4,
            tol=1e-4,
            copy_X=True,
            random_state=None)
        ridge.fit(in_X_train, in_y_train)

        Ytest_ridge = ridge.predict(in_X_test)
        inner_test_data['rethat'] = Ytest_ridge
        res1 = 1 - np.sum((inner_test_data['y'] - inner_test_data['rethat']) ** 2) / np.sum(inner_test_data['y'] ** 2)
        ret_test.append(res1)

        Yoos_ridge = ridge.predict(in_X_oos)
        inner_oos_data = in_oos_data.copy()
        inner_oos_data['rethat'] = Yoos_ridge

        data_ridge = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
        data_ridge.columns = ['Dates', 'stkcd', 'y', 'yhat']
        data_ridge['model'] = 'ridge'
        data_ridge['AlphaValue'] = a

        coef_ser[a] = pd.Series(ridge.coef_, index=in_X_train.columns)
        ret_oos.append(data_ridge)

    loc_max = ret_test.index(max(ret_test))
    a_max = alphas[loc_max]
    ridge_best = Ridge(
        alpha=a_max,
        fit_intercept=True,
        normalize=False,
        max_iter=1e4,
        tol=1e-4,
        copy_X=True,
        random_state=None)
    ridge_best.fit(in_X_train, in_y_train)
    yhat = ridge_best.predict(in_X_oos)

    inner_oos_data = in_oos_data.copy()
    inner_oos_data['rethat'] = yhat

    data_ridge = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
    data_ridge.columns = ['Dates', 'stkcd', 'y', 'yhat']
    return pd.concat(ret_oos), data_ridge, coef_ser


def Lasso_method(in_X_train, in_y_train, in_X_test, in_y_test, in_X_oos, in_test_data, in_oos_data):
    from sklearn.linear_model import Lasso

    inner_test_data = in_test_data.copy()
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2]
    ret_test = []
    ret_oos = []
    coef_ser = {}
    for a in alphas:
        lasso = Lasso(
            alpha=a,
            fit_intercept=True,
            normalize=False,
            max_iter=1e4,
            tol=1e-4,
            copy_X=True,
            random_state=None)
        lasso.fit(in_X_train, in_y_train)

        Ytest_lasso = lasso.predict(in_X_test)
        inner_test_data['rethat'] = Ytest_lasso
        res1 = 1 - np.sum((inner_test_data['y'] - inner_test_data['rethat']) ** 2) / np.sum(inner_test_data['y'] ** 2)
        ret_test.append(res1)

        Yoos_lasso = lasso.predict(in_X_oos)
        inner_oos_data = in_oos_data.copy()
        inner_oos_data['rethat'] = Yoos_lasso

        data_lasso = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
        data_lasso.columns = ['Dates', 'stkcd', 'y', 'yhat']
        data_lasso['model'] = 'lasso'
        data_lasso['AlphaValue'] = a

        coef_ser[a] = pd.Series(lasso.coef_, index=in_X_train.columns)
        ret_oos.append(data_lasso)

    loc_max = ret_test.index(max(ret_test))
    a_max = alphas[loc_max]
    lasso_best = Lasso(
        alpha=a_max,
        fit_intercept=True,
        normalize=False,
        max_iter=1e4,
        tol=1e-4,
        copy_X=True,
        random_state=None)
    lasso_best.fit(in_X_train, in_y_train)
    yhat = lasso_best.predict(in_X_oos)

    inner_oos_data = in_oos_data.copy()
    inner_oos_data['rethat'] = yhat

    data_lasso = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
    data_lasso.columns = ['Dates', 'stkcd', 'y', 'yhat']
    return pd.concat(ret_oos), data_lasso, coef_ser


def PCA_method(in_X_train, in_y_train, in_X_test, in_y_test, in_X_oos, in_test_data, in_oos_data):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    import matplotlib.gridspec as gridspec

    in_X_train = in_X_train.iloc[:, :20]
    in_X_test = in_X_test.iloc[:, :20]
    in_X_oos = in_X_oos.iloc[:, :20]
    kf_10 = KFold(n_splits=5, shuffle=True, random_state=123)
    mse = []
    ret_oos = []

    importance_dict = {}
    exp_ratio_dict = {}

    for i in range(5, 11):
        pca = PCA(n_components=i)
        pca.fit(in_X_train)

        importance = list(pca.explained_variance_ratio_)
        exp_ratio = pd.Series(pca.explained_variance_ratio_).cumsum()

        importance_dict[i] = importance
        exp_ratio_dict[i] = exp_ratio

        X_pca = pca.transform(in_X_train)
        Y = np.matrix(in_y_train).reshape(-1, 1)
        X_pca = np.matrix(X_pca).reshape(-1, np.size(X_pca, 1))
        reg = LinearRegression()
        score = -1 * cross_val_score(reg, X_pca, Y, cv=kf_10, scoring='neg_mean_squared_error', verbose=0).mean()
        mse.append(score)

        model1 = LinearRegression()
        lin_reg = model1.fit(X_pca, Y)

        Xoos_pca = pca.transform(in_X_oos)
        Xoos_pca = np.matrix(Xoos_pca).reshape(-1, np.size(Xoos_pca, 1))
        yhat = lin_reg.predict(Xoos_pca)
        inner_oos_data = in_oos_data.copy()
        inner_oos_data['rethat'] = yhat

        data_pca = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
        data_pca.columns = ['Dates', 'stkcd', 'y', 'yhat']
        data_pca['model'] = 'pca'
        data_pca['n'] = i
        ret_oos.append(data_pca)

    optimal_comp = mse.index(min(mse)) + 1

    pca_best = PCA(n_components=optimal_comp)
    pca_best.fit(in_X_train)
    X_pca = pca_best.transform(in_X_train)
    Y = np.matrix(in_y_train).reshape(-1, 1)
    X_pca = np.matrix(X_pca).reshape(-1, np.size(X_pca, 1))
    model1 = LinearRegression()
    lin_reg = model1.fit(X_pca, Y)
    Xoos_pca = pca_best.transform(in_X_oos)
    Xoos_pca = np.matrix(Xoos_pca).reshape(-1, np.size(Xoos_pca, 1))
    yhat = lin_reg.predict(Xoos_pca)

    inner_oos_data = in_oos_data.copy()
    inner_oos_data['rethat'] = yhat

    data_pca = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
    data_pca.columns = ['Dates', 'stkcd', 'y', 'yhat']

    return pd.concat(ret_oos), data_pca, importance_dict, exp_ratio_dict


def PLS_method(in_X_train, in_y_train, in_X_test, in_y_test, in_X_oos, in_test_data, in_oos_data):
    from sklearn.cross_decomposition import PLSRegression, PLSCanonical
    kf_10 = KFold(n_splits=5, shuffle=True, random_state=123)
    mse = []
    ret_test = []
    ret_oos = []
    coef_ser = {}
    for k in range(1, 9):
        pls = PLSRegression(n_components=k, scale=False, copy=True)
        score = -1 * cross_val_score(pls, in_X_train, in_y_train, cv=kf_10, scoring='neg_mean_squared_error',
                                     verbose=0).mean()
        mse.append(score)
        pls.fit(in_X_train, in_y_train)

        Yoos_pls = pls.predict(in_X_oos)
        inner_oos_data = in_oos_data.copy()
        inner_oos_data['rethat'] = Yoos_pls

        data_pls = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
        data_pls.columns = ['Dates', 'stkcd', 'y', 'yhat']
        data_pls['model'] = 'pls'
        data_pls['n'] = k

        coef_ser[k] = pd.Series(pls.coef_[:, 0], index=in_X_train.columns)
        ret_oos.append(data_pls)

    optimal_comp = mse.index(min(mse)) + 1
    pls_best = PLSRegression(n_components=optimal_comp, scale=False, copy=True)
    pls_best.fit(in_X_train, in_y_train)
    yhat = pls_best.predict(in_X_oos)

    inner_oos_data = in_oos_data.copy()
    inner_oos_data['rethat'] = yhat

    inner_oos_data = in_oos_data.copy()
    inner_oos_data['rethat'] = yhat

    data_pls = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
    data_pls.columns = ['Dates', 'stkcd', 'y', 'yhat']

    return pd.concat(ret_oos), data_pls, coef_ser


from keras.layers import LeakyReLU as leakyrelu
from keras import regularizers
import tensorflow as tf


def create_model(learn_rate=0.001, momentum=0.4, layer=3, dv=0.2):  ###0.4relu
    # create model
    model1 = Sequential()
    model1.add(Dense(units=256,
                     input_dim=np.size(X_train, 1),
                     kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(1e-2)
                     ))

    model1.add(BatchNormalization())

    model1.add(Activation(gelu))

    model1.add(Dropout(dv))
    if layer >= 2:
        for i in range(layer - 1):
            layer_i = 2 ** (i + 1)
            model1.add(Dense(units=int(256 / layer_i), kernel_initializer='random_normal',
                             kernel_regularizer=regularizers.l2(1e-2)))
            model1.add(BatchNormalization())
            model1.add(Activation(gelu))
            model1.add(Dropout(dv))
        model1.add(Dense(1))
    else:
        model1.add(Dense(1))
    # Compile model
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model1.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model1


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, TimeDistributed

from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
import math as m
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers


def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf


def create_model(input_size, learn_rate=0.001, momentum=0.4, layer=3, dv=0.2):  ###0.4relu
    # create model
    model1 = Sequential()
    model1.add(Dense(units=input_size,
                     input_dim=input_size,
                     kernel_initializer='random_normal', kernel_regularizer=regularizers.l2(1e-2)
                     ))

    model1.add(BatchNormalization())

    model1.add(Activation(gelu))

    model1.add(Dropout(dv))
    if layer >= 2:
        for i in range(layer - 1):
            layer_i = 2 ** (i + 1)
            model1.add(Dense(units=int(256 / layer_i), kernel_initializer='random_normal',
                             kernel_regularizer=regularizers.l2(1e-2)))
            model1.add(BatchNormalization())
            model1.add(Activation(gelu))
            model1.add(Dropout(dv))
        model1.add(Dense(1))
    else:
        model1.add(Dense(1))
    # Compile model
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model1.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model1


def NN(in_X_train, in_y_train, in_X_test, in_y_test, in_X_oos, in_test_data, in_oos_data):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=2,
                                  min_lr=0.000000001,
                                  min_delta=0.000000,
                                  verbose=-1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=-1)
    callbacks = [early_stopping]

    ret_oos = []
    for l in range(1, 4):
        model1 = create_model(input_size=in_X_train.shape[1], layer=l)
        train_history = model1.fit(x=in_X_train,
                                   y=in_y_train, validation_data=(in_X_test, in_y_test),
                                   epochs=50, batch_size=200, verbose=0, callbacks=callbacks)
        nn_pred = model1.predict(in_X_oos)
        inner_oos_data = in_oos_data.copy()
        inner_oos_data['rethat'] = nn_pred
        data_nn = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
        data_nn.columns = ['Dates', 'stkcd', 'y', 'yhat']
        data_nn['model'] = 'NN'
        data_nn['layers'] = l

        ret_oos.append(data_nn)
    return pd.concat(ret_oos)


def RandomForest_method(in_X_train, in_y_train, in_X_test, in_y_test, in_X_oos, in_test_data, in_oos_data):
    from sklearn.ensemble import RandomForestRegressor
    inner_test_data = in_test_data.copy()

    n_es = [2, 3, 4, 5, 6]
    ret_test = []
    ret_oos = []
    coef_ser = {}
    for n in n_es:
        rf = RandomForestRegressor(
            n_estimators=n)
        rf.fit(in_X_train, in_y_train)

        Ytest_rf = rf.predict(in_X_test)
        inner_test_data['rethat'] = Ytest_rf
        res1 = 1 - np.sum((inner_test_data['y'] - inner_test_data['rethat']) ** 2) / np.sum(inner_test_data['y'] ** 2)
        ret_test.append(res1)

        Yoos_rf = rf.predict(in_X_oos)
        inner_oos_data = in_oos_data.copy()
        inner_oos_data['rethat'] = Yoos_rf

        data_rf = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
        data_rf.columns = ['Dates', 'stkcd', 'y', 'yhat']
        data_rf['model'] = 'rf'
        data_rf['n_estimators'] = n

        ret_oos.append(data_rf)

    loc_max = ret_test.index(max(ret_test))
    ne_max = n_es[loc_max]
    rf_best = RandomForestRegressor(
        n_estimators=ne_max)

    rf_best.fit(in_X_train, in_y_train)
    yhat = rf_best.predict(in_X_oos)

    inner_oos_data = in_oos_data.copy()
    inner_oos_data['rethat'] = yhat

    data_rf = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
    data_rf.columns = ['Dates', 'stkcd', 'y', 'yhat']
    return pd.concat(ret_oos), data_rf, coef_ser


def GBDT_method(in_X_train, in_y_train, in_X_test, in_y_test, in_X_oos, in_test_data, in_oos_data):
    from sklearn.ensemble import GradientBoostingRegressor
    inner_test_data = in_test_data.copy()

    n_es = [2, 3, 4, 5, 6]
    ret_test = []
    ret_oos = []
    coef_ser = {}
    for n in n_es:
        gbdt = GradientBoostingRegressor(
            n_estimators=n)
        gbdt.fit(in_X_train, in_y_train)

        Ytest_gbdt = gbdt.predict(in_X_test)
        inner_test_data['rethat'] = Ytest_gbdt
        res1 = 1 - np.sum((inner_test_data['y'] - inner_test_data['rethat']) ** 2) / np.sum(inner_test_data['y'] ** 2)
        ret_test.append(res1)

        Yoos_gbdt = gbdt.predict(in_X_oos)
        inner_oos_data = in_oos_data.copy()
        inner_oos_data['rethat'] = Yoos_gbdt

        data_gbdt = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
        data_gbdt.columns = ['Dates', 'stkcd', 'y', 'yhat']
        data_gbdt['model'] = 'gbdt'
        data_gbdt['n_estimators'] = n

        ret_oos.append(data_gbdt)

    loc_max = ret_test.index(max(ret_test))
    ne_max = n_es[loc_max]
    gbdt_best = GradientBoostingRegressor(
        n_estimators=ne_max)

    gbdt_best.fit(in_X_train, in_y_train)
    yhat = gbdt_best.predict(in_X_oos)

    inner_oos_data = in_oos_data.copy()
    inner_oos_data['rethat'] = yhat

    data_gbdt = inner_oos_data[['Dates', 'stkcd', 'y', 'rethat']]
    data_gbdt.columns = ['Dates', 'stkcd', 'y', 'yhat']
    return pd.concat(ret_oos), data_gbdt, coef_ser
