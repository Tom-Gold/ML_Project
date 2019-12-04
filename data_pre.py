import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


label = 'birth_type_CS'
label_col = ['induced_csec_assis', 'not_induced_csec_assis', 'regular']
norm_cols = ['Afger_5_', 'infant_wight', 'age', 'birth_week', 'birth_+days']
cols = ['AB', 'CS', 'EP', 'G', 'LC', 'P', 'infant_wight', 'age', 'birth_week', 'birth_days', 'birth_start_type_STRIPPING', 'birth_start_type_Spontaneous', 'birth_start_type_induce', 'birth_start_type_neck', 'birth_start_type_other', 'birth_start_type_stripping', 'birth_start_type_Foley_bulb', 'birth_start_type_PROSTO', 'birth_type_CS', 'Amniotic_fluid_BLOOD', 'Amniotic_fluid_NONE_OBSERVED', 'Amniotic_fluid_LOW_AMOUNT', 'Amniotic_fluid_MAS_THIN', 'Amniotic_fluid_MAS_THICK', 'Amniotic_fluid_MAS', 'Amniotic_fluid_MAS_STAIN', 'Amniotic_fluid_CLEAR', 'Amniotic_fluid_Polyhydramnios', 'position_Breech__Presentation', 'position_Brow_Presentation', 'position_Compound_Presentation', 'position_Face_Presentation', 'position_Footling_Presentation', 'position_Transverse', 'position_Vertex', 'position_Vertex_L.O.A', 'position_Vertex_L.O.P', 'position_Vertex_-_L.O.T', 'position_Vertex_-_O.A._or_O.T', 'position_Vertex_-_OT', 'position_Vertex_-_P.O.P.', 'position_Vertex_-_R.O.A', 'position_Vertex_-_R.O.P', 'position_Vertex_-_R.O.T']

min_cols = ['AB', 'CS', 'EP', 'G', 'LC', 'P', 'infant_wight', 'age', 'birth_week', 'birth_days','position_Breech__Presentation', 'position_Brow_Presentation', 'position_Compound_Presentation', 'position_Face_Presentation', 'position_Footling_Presentation', 'position_Transverse', 'position_Vertex', 'position_Vertex_L.O.A', 'position_Vertex_L.O.P', 'position_Vertex_-_L.O.T', 'position_Vertex_-_O.A._or_O.T', 'position_Vertex_-_OT', 'position_Vertex_-_P.O.P.', 'position_Vertex_-_R.O.A', 'position_Vertex_-_R.O.P', 'position_Vertex_-_R.O.T']

def train_test_val_split(df_raw):
    train, test = train_test_split(df_raw, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')
    return train, test, val


def generate_dataset(df_raw, norm_with_sk=False):

    df, pred_data = my_train_test_split(df_raw, 0.01)

    X, y = x_y_split(df)

    pred_x, pred_y = x_y_split(pred_data)

    X_T_train, X_test, y_T_train, y_test = train_test_split(X, y, test_size=0.1)

    newdata = upsample(X_T_train, y_T_train)
    
    X_train, y_train = x_y_split(newdata)

    print("X_train shape: ", X_train.shape, "X_test shape: ", X_test.shape)
    print("y_train shape: ", y_train.shape, "y_test shape: ", y_test.shape)
    print("pred_x shape: ", pred_x.shape, "pred_y shape: ", pred_y.shape)
    print("Test Samples: ", X_test.shape[0], " Test Labels Ratio: ", (y_test.sum()/X_test.shape[0]))
    print("Train Samples: ", X_train.shape[0], " Train Labels Ratio: ", (y_train.sum()/X_train.shape[0]))  

    return X_train, X_test, y_train, y_test, pred_x, pred_y


def print_df_info(df):
    print("Data shape: ", df.shape)
#     print(df.head(3))  
#     print(df.describe())
    

def set_df_features(df, feat, label=label_col):
    y_df = df[label].copy()
    x_df = df[feat].copy()
    res = pd.concat([x_df,y_df],axis=1)
    return res
    
    
def file_to_dataframe(file="one_hot_data_clean.xlsx", dtype='float32', with_norm=False):
    path = os.getcwd()
    df = pd.read_excel(path + "/Data/" + file)
    df_raw = df[cols]
#     df_raw = df[df.birth_start_type_CS == 0]
#     features = df_raw.columns.tolist()
    
    if with_norm: 
        df_raw[norm_cols] = df_raw[norm_cols].astype(dtype)
        df_raw = sk_norm_data(df_raw)
    
    print_df_info(df_raw)
    return df_raw


def sk_norm_data(df):
    all_features = df.columns.tolist()
    cols_to_norm = []
    for col in all_features:
        max_v = df[col].max()
        if max_v > 1:
            print("col ", col, " val ", max_v)
            cols_to_norm.append(col)
    feat = df[cols_to_norm].astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    feat = scaler.fit_transform(feat.values)
    # scaler = sc.fit(feat.values)
    # feat = scaler.transform(feat.values)
    df[cols_to_norm] = feat
    return df


#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#     return X_train, X_test

# def sk_norm_data(X_train, X_test):
#     sc = StandardScaler()
#     x_tr = X_train[norm_cols]
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#     return X_train, X_test


def norm_data(df_raw):
    df = pd.DataFrame(dtype='float32')
    for col in total_cols:
        if col in norm_cols:
            df[col] = df_raw.loc[:, col] / df_raw.loc[:, col].max()
        else:
            df[col] = df_raw[col]
    return df


def my_train_test_split(df, ratio=0.1):
    test = df.sample(frac=ratio)
    train = df.drop(test.index)
    return train, test


def x_y_split(df, label=label_col):
    y_data = df[label].copy()
    x_data = df.drop(columns=label)
    # x_data = df[features]
    return x_data, y_data


# def split_to_datasets():
#     df = file_to_dataframe()
#     train, test = train_test_split(df)
#     x_tr, y_tr = x_y_split(train)
#     x_ts, y_ts = x_y_split(test)
#     return x_tr, y_tr, x_ts, y_ts


def split_to_datasets_with_val():
    df = file_to_dataframe()
    train, test_val = my_train_test_split(df, 0.2)
    test, val = my_train_test_split(test_val, 0.5)
    x_tr, y_tr = x_y_split(train)
    x_val, y_val = x_y_split(val)
    x_ts, y_ts = x_y_split(test)
    return x_tr, y_tr, x_ts, y_ts, x_val, y_val


def get_input_fn(data_set, num_epochs=None, n_batch=128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in features}),
        y=pd.Series(data_set[label_col].values),
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle)

# def upsample(x, y):
#     X = pd.concat([X_train, y_train], axis=1)

# # separate minority and majority classes
# not_fraud = X[X.Class==0]
# fraud = X[X.Class==1]

# # upsample minority
# fraud_upsampled = resample(fraud,
#                           replace=True, # sample with replacement
#                           n_samples=len(not_fraud), # match number in majority class
#                           random_state=27) # reproducible results

# # combine majority and upsampled minority
# upsampled = pd.concat([not_fraud, fraud_upsampled])

# # check new class counts
# upsampled.Class.value_counts()

def upsample(x, y):
    X = pd.concat([x, y], axis=1)
    no_csec = X[X.birth_type_CS == 0]
    csec = X[X.birth_type_CS > 0]

    csec_upsampled = resample(csec,replace=True, n_samples=len(no_csec), random_state=27)
    newdata = pd.concat([no_csec, csec_upsampled])
    return newdata

def upsample_df(X):
#     X = pd.concat([x, y], axis=1)
    no_csec = X[X.regular == 1]
    csec = X[X.regular == 0]

    csec_upsampled = resample(csec,replace=True, n_samples=len(no_csec), random_state=27)
    newdata = pd.concat([no_csec, csec_upsampled])
    return newdata


def find_correlations(df, feat_amount=10, label = 'birth_type_CS'):
    result_df_corr = df.corr()
    result_df_corr = abs(result_df_corr[label])
#     print(result_df_corr[label])
#     print(result_df_corr)
    result_df_corr = result_df_corr.sort_values(ascending=False)
    result_df_corr = result_df_corr[1:feat_amount]
    result = result_df_corr.index.tolist()
    return result


def find_K_Best(df, K=10):
    X, y = x_y_split(df)
    #apply SelectKBest class to extract top K best features
    bestfeatures = SelectKBest(score_func=chi2, k=K)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Featues','Score']  #naming the dataframe columns
    feat = featureScores.nlargest(K,'Score')
    print(feat)  #print K best features
    res = feat['Featues'].values.tolist()
    return res


def find_tree_best(df, top = 10):
    X, y = x_y_split(df)
    #Extra Tree Classifier for extracting the top 10 features for the dataset.
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(top).plot(kind='barh')
    plt.show()
    res = feat_importances.nlargest(top).index.tolist()
    return res