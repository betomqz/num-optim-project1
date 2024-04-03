import numpy as np
import pandas as pd
from myqp import myqp_intpoint_proy
import matplotlib.pyplot as plt

# Formatos de gráficas
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
plt.style.use('seaborn-whitegrid')
plt.rc('font', size=16)

myred = "#F95252"
myblue = "#2ab0ff"

# Cargar base de datos
df = pd.read_csv('wine.data')

A = df[df['class'] == 1]
B = df[df['class'] == 2]
C = df[df['class'] == 3]

X_A = A.drop(columns=['class'])
y_A = A['class']
X_B = B.drop(columns=['class'])
y_B = B['class']
X_C = C.drop(columns=['class'])
y_C = C['class']

# -------------------------------------------------------- SEPARAR A y B
# s y r son el número de filas de A y B, respectivamente
# n es el número de columnas que tienen A y B y es el tamaño de nuestra w
(s, n) = X_A.shape
r = X_B.shape[0]

# Agregamos la variable de la beta
n += 1

Q = np.eye(n)
Q[-1][-1] = 0
c = np.zeros(n)
F = np.concatenate((-X_A, X_B))
aux = np.concatenate((-np.ones(s),np.ones(r)))
F = np.c_[F, aux]
d = np.ones(s+r)

w_hat,mu,z,iter = myqp_intpoint_proy(Q, F, c, d, verbose=True)

w = w_hat[:-1]
beta = w_hat[-1]

# valor de la función objetivo sin tomar en cuenta a beta
print(np.dot(w,w))

# beta
print(beta)

# Todos tendrían que ser <= -1
AT_w = np.dot(X_A, w) + beta

# Todos tendrían que ser >= 1
BT_w = np.dot(X_B, w) + beta

#### Graficar
fig, ax = plt.subplots(figsize=(12,8))
len_A = np.arange(len(AT_w))
ax.scatter(len_A+1, AT_w, c=myred, label=r"$\mathbb{A}^Tw+\beta$")
ax.scatter(np.arange(len(AT_w), len(BT_w)+len(AT_w))+1, BT_w, color=myblue, label=r"$\mathbb{B}^Tw+\beta$")
plt.vlines(len_A+1, ymin = AT_w, ymax = -1, colors=myred)
plt.vlines(np.arange(len(AT_w), len(BT_w)+len(AT_w))+1, ymin = 1, ymax = BT_w, colors=myblue)
ax.legend()
ax.set_title("Verificación de restricciones")
#plt.savefig('outputs/AvsB.pdf', format='pdf')
plt.show()

# -------------------------------------------------------- SEPARAR C y B
# s y r son el número de filas de C y B, respectivamente
# n es el número de columnas que tienen C y B y es el tamaño de nuestra w
(s, n) = X_C.shape
r = X_B.shape[0]

# Agregamos la variable de la beta
n += 1

Q = np.eye(n)
Q[-1][-1] = 0
c = np.zeros(n)
F = np.concatenate((-X_C, X_B))
aux = np.concatenate((-np.ones(s),np.ones(r)))
F = np.c_[F, aux]
d = np.ones(s+r)

w_hat,mu,z,iter = myqp_intpoint_proy(Q, F, c, d, verbose=True)

w = w_hat[:-1]
beta = w_hat[-1]

# valor de la función objetivo sin tomar en cuenta a beta
print(np.dot(w,w))

# beta
print(beta)

# Todos tendrían que ser <= -1
CT_w = np.dot(X_C, w) + beta

# Todos tendrían que ser >= 1
BT_w = np.dot(X_B, w) + beta

# graficar
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
plt.style.use('seaborn-whitegrid')
plt.rc('font', size=16)

mygreen = "#3AC08E"
myblue = "#2ab0ff"

fig, ax = plt.subplots(figsize=(12,8))
len_C = np.arange(len(CT_w))
ax.scatter(len_C+1, CT_w, c=mygreen, label=r"$\mathbb{C}^Tw+\beta$")
ax.scatter(np.arange(len(CT_w), len(BT_w)+len(CT_w))+1, BT_w, color=myblue, label=r"$\mathbb{B}^Tw+\beta$")
plt.vlines(len_C+1, ymin = CT_w, ymax = -1, colors=mygreen)
plt.vlines(np.arange(len(CT_w), len(BT_w)+len(CT_w))+1, ymin = 1, ymax = BT_w, colors=myblue)
ax.legend()
ax.set_title("Verificación de restricciones")
#plt.savefig('outputs/CvsB.pdf', format='pdf')
plt.show()

# -------------------------------------------------------- SEPARAR A y C
# s y r son el número de filas de A y C, respectivamente
# n es el número de columnas que tienen A y C y es el tamaño de nuestra w
(s, n) = X_A.shape
r = X_C.shape[0]

# Agregamos la variable de la beta
n += 1

Q = np.eye(n)
Q[-1][-1] = 0
c = np.zeros(n)
F = np.concatenate((-X_A, X_C))
aux = np.concatenate((-np.ones(s),np.ones(r)))
F = np.c_[F, aux]
d = np.ones(s+r)

w_hat,mu,z,iter = myqp_intpoint_proy(Q, F, c, d, verbose=True)

w = w_hat[:-1]
beta = w_hat[-1]

# valor de la función objetivo sin tomar en cuenta a beta
print(np.dot(w,w))

# beta
print(beta)

# Todos tendrían que ser <= -1
AT_w = np.dot(X_A, w) + beta

# Todos tendrían que ser >= 1
CT_w = np.dot(X_C, w) + beta

# graficar
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
plt.style.use('seaborn-whitegrid')
plt.rc('font', size=16)

fig, ax = plt.subplots(figsize=(12,8))
len_A = np.arange(len(AT_w))
ax.scatter(len_A+1, AT_w, c=myred, label=r"$\mathbb{A}^Tw+\beta$")
ax.scatter(np.arange(len(AT_w), len(CT_w)+len(AT_w))+1, CT_w, color=mygreen, label=r"$\mathbb{C}^Tw+\beta$")
plt.vlines(len_A+1, ymin = AT_w, ymax = -1, colors=myred)
plt.vlines(np.arange(len(AT_w), len(CT_w)+len(AT_w))+1, ymin = 1, ymax = CT_w, colors=mygreen)
ax.legend()
ax.set_title("Verificación de restricciones")
# plt.savefig('outputs/AvsC.pdf', format='pdf')
plt.show()