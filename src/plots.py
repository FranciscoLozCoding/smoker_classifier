import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances(model, n_features, feature_names):
    indices = np.argsort(model.feature_importances_)[::-1]
    names_sorted = [feature_names[i] for i in indices]
    
    plt.barh(range(n_features)[::-1], model.feature_importances_[indices], align='center')
    plt.yticks(np.arange(n_features)[::-1], names_sorted)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()