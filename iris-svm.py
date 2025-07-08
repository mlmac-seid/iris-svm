# imports
import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
import seaborn as sns
import pandas as pd

# 3d figures
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# creating animations
import matplotlib.animation
from IPython.display import HTML

# styling additions
from IPython.display import HTML
style = '''
    <style>
        div.info{
            padding: 15px;
            border: 1px solid transparent;
            border-left: 5px solid #dfb5b4;
            border-color: transparent;
            margin-bottom: 10px;
            border-radius: 4px;
            background-color: #fcf8e3;
            border-color: #faebcc;
        }
        hr{
            border: 1px solid;
            border-radius: 5px;
        }
    </style>'''
HTML(style)

import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x="sepal_width", y="sepal_length", z='petal_width',
                    color="species", template="simple_white")
fig.update_traces(marker={'size': 4})

from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
colors = sns.color_palette(as_cmap=True)
cmap = ListedColormap(sns.color_palette()[0:2])
df = df.loc[df['species'] != 'versicolor']
x_train = df[['sepal_length', 'sepal_width']]
y_train = df['species']
y_train = y_train.replace('setosa', 0)
y_train = y_train.replace('virginica', 1)
y_train = np.array(y_train)

model = SVC(kernel='linear', C=1e10)
model.fit(x_train.values, y_train);

model.support_vectors_

setosa = df.loc[df['species'] == 'setosa']
setosa = setosa[['sepal_length', 'sepal_width']]
setosa = np.array(setosa)

virginica = df.loc[df['species'] == 'virginica']
virginica = virginica[['sepal_length', 'sepal_width']]
virginica = np.array(virginica)

# plotting function
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none', edgecolors='k');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(setosa[:,0], setosa[:,1], s=50, cmap=cmap)
plt.scatter(virginica[:,0], virginica[:,1], s=50, cmap=cmap)
plot_svc_decision_function(model);

hard_svm = SVC(kernel='linear', C=1e10)
hard_svm.fit(x_train.values, y_train)

soft_svm = SVC(kernel='linear', C=1)
soft_svm.fit(x_train.values, y_train)

very_soft_svm = SVC(kernel='linear', C=1e-2)
very_soft_svm.fit(x_train.values, y_train);



# setup plot
fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharex=True,sharey=True, figsize=(15,5))
ax1.set_title(f'C={hard_svm.C}')
ax2.set_title(f'C={soft_svm.C}')
ax3.set_title(f'C={very_soft_svm.C}')
ax1.scatter(setosa[:,0], setosa[:,1], s=50, cmap=cmap)
ax1.scatter(virginica[:,0], virginica[:,1], s=50, cmap=cmap)
ax2.scatter(setosa[:,0], setosa[:,1], s=50, cmap=cmap)
ax2.scatter(virginica[:,0], virginica[:,1], s=50, cmap=cmap)
ax3.scatter(setosa[:,0], setosa[:,1], s=50, cmap=cmap)
ax3.scatter(virginica[:,0], virginica[:,1], s=50, cmap=cmap)
plot_svc_decision_function(hard_svm,ax=ax1);
plot_svc_decision_function(soft_svm,ax=ax2);
plot_svc_decision_function(very_soft_svm,ax=ax3);