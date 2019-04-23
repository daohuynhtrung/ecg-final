import numpy as np
from sklearn.decomposition import PCA, FastICA

def reduceDemensionPCA(data, n_components):
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(data)
    dataRestored = pca.inverse_transform(principalComponents)
    return dataRestored

def reduceDemensionICA(data, n_components):
    ica = FastICA(n_components=n_components)
    ica.fit(data)
    principalComponents = ica.fit_transform(data)
    dataRestored = ica.inverse_transform(principalComponents)
    return dataRestored