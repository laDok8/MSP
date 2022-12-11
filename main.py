#!/usr/bin/env python
# coding: utf-8

# ## 1. Úkol

# In[ ]:


# zadáni 4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib import cm


# Import dat, kde poslední sloupec tvoří mnou zjištěné hodnoty.

# In[ ]:


# data 1
city = np.array(['big', 'big', 'medium', 'medium', 'medium', 'small', 'small', 'small'])
cat = np.array(
    ['Praha', 'Brno', 'Znojmo', 'Tišnov', 'Rokytnice nad Jizerou', 'Jablunkov', 'Dolní Věstonice', 'student'])
count = np.array([1327, 915, 681, 587, 284, 176, 215, 21])
winter = np.array([510, 324, 302, 257, 147, 66, 87, 11])
summer = np.array([352, 284, 185, 178, 87, 58, 65, 4])
change = np.array([257, 178, 124, 78, 44, 33, 31, 4])
indifferent = np.array([208, 129, 70, 74, 6, 19, 32, 2])


# Pro zjištění rovnosti procentuálního zastoupení obyvatel měst použijeme test dobré shody.
# $H_0 : p_1 = p_2 ... = p_8$

# In[ ]:


def chi_test(str, cat, count, dof=1):
    #print("got: ", cat.astype(int), "\nexp: ", (([cat.sum() / count.sum()] * len(count)) * count).astype(int), '\n')
    print(str, stats.chisquare(cat, f_exp=([cat.sum() / count.sum()] * len(count)) * count, ddof=dof))


chi_test("winter", winter, count)
chi_test("summer", summer, count)
chi_test("change", change, count)
print("critical value: ", stats.chi2.ppf(0.95, 6))


# Pro zimní čas zamítamé $H_0$ (p < 0.05), tudíž se percentuální zastoupení mezi městy liší.
# Pro změnu času a letní čas $H_0$ nezamítamé.

# Pro zjištění rovnosti procentuálního zastoupení mezi různě velkými městy (velké, střední, malé) použijeme obdobný test nad seskupenými daty.

# In[ ]:


# group by city
city_group = pd.DataFrame({'city': city, 'count': count, 'winter': winter, 'indifferent': indifferent})
city_group = city_group.groupby('city').sum()

chi_test("group_winter", city_group['winter'], city_group['count'])
chi_test("group_indifferent", city_group['indifferent'], city_group['count'])


# $H_0$ zamítáme jak pro zimní čas, tak pro nerozhodnuté obyvatele.

# Pro odhadnutí z které skupiny student pocházi (velké, střední, malé) použijeme opět $\chi^2$ test. Očekávané hodnoty pro příslušnost studenta tvoři bodový odhad z daných skupin.

# 

# In[ ]:


city_group = pd.DataFrame(
    {'city': city, 'cat': cat, 'count': count, 'winter': winter, 'summer': summer, 'change': change,
     'indifferent': indifferent})
city_group = city_group[city_group['cat'] != 'student']
city_group = city_group.drop('cat', axis=1)

city_group = city_group.groupby(['city']).sum()
city_group.loc['student'] = [count[-1], winter[-1], summer[-1], change[-1], indifferent[-1]]
# separate count column
counts = city_group['count']
city_group = city_group.drop('count', axis=1)

# chi_test student to X cities
student = city_group.loc['student'].values

chi_test("student=big city", student, city_group.loc['big'].values, dof=0)
chi_test("student=medium city", student, city_group.loc['medium'].values, dof=0)
chi_test("student=small city", student, city_group.loc['small'].values, dof=0)


# Ze získaných dat nelze vyvodit příslušnost studenta do kterékoliv skupiny (p> 0.05 pro každou skupinu).
# Závěr nejde vyvodit nejspíše kvůli podobnosti kategorii a malému vzorku mých dat, avšak největší podobnost se ukazuje ke středně velkým městům.
# 

# ## 2. Úkol

# import dat

# In[ ]:


# data2
x = np.array(
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 2.22, 2.22, 2.22, 2.22, 2.22, 2.22, 2.22, 4.44, 4.44, 4.44, 4.44, 4.44,
     4.44, 4.44, 6.67, 6.67, 6.67, 6.67, 6.67, 6.67, 6.67, 8.89, 8.89, 8.89, 8.89, 8.89, 8.89, 8.89, 11.11, 11.11,
     11.11, 11.11, 11.11, 11.11, 11.11, 13.33, 13.33, 13.33, 13.33, 13.33, 13.33, 13.33, 15.56, 15.56, 15.56, 15.56,
     15.56, 15.56, 15.56, 17.78, 17.78, 17.78, 17.78, 17.78, 17.78, 17.78, 20.00, 20.00, 20.00, 20.00, 20.00, 20.00,
     20.00])
y = np.array(
    [0.00, 1.67, 3.33, 5.00, 6.67, 8.33, 10.00, 0.00, 1.67, 3.33, 5.00, 6.67, 8.33, 10.00, 0.00, 1.67, 3.33, 5.00, 6.67,
     8.33, 10.00, 0.00, 1.67, 3.33, 5.00, 6.67, 8.33, 10.00, 0.00, 1.67, 3.33, 5.00, 6.67, 8.33, 10.00, 0.00, 1.67,
     3.33, 5.00, 6.67, 8.33, 10.00, 0.00, 1.67, 3.33, 5.00, 6.67, 8.33, 10.00, 0.00, 1.67, 3.33, 5.00, 6.67, 8.33,
     10.00, 0.00, 1.67, 3.33, 5.00, 6.67, 8.33, 10.00, 0.00, 1.67, 3.33, 5.00, 6.67, 8.33, 10.00],
    dtype='float64')
z = np.array(
    [3.51678419592411, 0.1849332370251, -6.99049059346124, -20.8617205435173, -42.18343555059, -64.1081000325912,
     -97.0458957178397, 9.94388450504941, 9.8986323460927, 0.00208487443735393, -11.9988247882815, -34.3924062967239,
     -57.8057880895009, -88.5027847946918, 27.2265088682819, 24.4284903127453, 18.2145118228463, 1.94336256143575,
     -16.7727492407465, -42.0967616131069, -71.1967634077031, 56.3446765711985, 52.378628730395, 44.868666552744, 4.56,
     11.1141991787491, -14.9444563191628, -46.3430249668268, 92.5713346195499, 89.3713833654496, 81.3928266038351,
     65.624785460706, 44.8871370272759, 23.7631895332308, -8.64772334749607, 139.152701188226, 135.142403459449,
     125.462388521822, 112.482276630433, 94.9110448098603, 67.9275602260165, 39.4220662346769, 195.442542098054,
     192.351096305394, 185.89808996746, 169.086590035809, 150.799896551493, 126.327402037799, 95.0588038056046,
     260.907635326248, 257.874160634941, 251.576673225094, 235.967305190209, 217.589979963901, 192.992167354398,
     160.733689709696, 337.506810454201, 334.670521354979, 324.663539748041, 313.044243941327, 293.633223156008,
     269.455800707765, 235.257507397646, 423.522399438186, 421.287561597445, 412.428044955087, 397.834243660025,
     382.450519904685, 355.996853248479, 322.553027375103])


# Pomocné funkce pro vizualizaci regrese.

# In[ ]:


x1 = np.linspace(0, 20, 100)
y1 = np.linspace(0, 10, 100)
x1, y1 = np.meshgrid(x1, y1)


def plot_3d(x, y, z, x1, y1, z1):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x1, y1, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
    # add data points
    ax.scatter(x, y, z, c='r', marker='x')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # rotate view
    ax.view_init(15, 75)
    plt.show()


# Vypočtená statistika pro obecnou kvadratickou regresi

# In[ ]:


# alpha = .05 is default
F = sm.add_constant(np.column_stack((x, y, x ** 2, y ** 2, x * y)))
modelres = sm.OLS(z, F).fit()
print(modelres.summary(xname=['const', 'x', 'y', 'x^2', 'y^2', 'xy']))
print()
print('R^2', modelres.rsquared)

z1 = modelres.predict(sm.add_constant(
    np.column_stack((x1.ravel(), y1.ravel(), x1.ravel() ** 2, y1.ravel() ** 2, x1.ravel() * y1.ravel())))).reshape(
    x1.shape)

plot_3d(x, y, z, x1, y1, z1)


# Pro redukci modelu využijeme T-test na nulovou hodnotu koeficientu.
# Koeficient můžeme při analýze zanedbat při zamítnuti nulové hypotézy rovnosti parametrů ((P>|t|) > 0.05).

# In[ ]:


print(modelres.t_test("x5 = 0"))  #xy

F = sm.add_constant(np.column_stack((x, y, x ** 2, y ** 2)))
modelres = sm.OLS(z, F).fit()

print(modelres.t_test("x2 = 0"))  #y


# Výsledný regresní model: $Z = B_1 + B_2X + B_3X^2 + B_4Y^2$.
# 2b) Hodnoty koeficientů jsou vyznačeny v tabulce pod sloupcem "coef".
# 95% interval spolehlivosti je vyznačen v posledních 2 sloupcích [0.025      0.975].

# In[ ]:


F = sm.add_constant(np.column_stack((x, x ** 2, y ** 2)))
modelres = sm.OLS(z, F).fit()
print(modelres.summary(xname=['const', 'x', 'x^2', 'y^2']))
print()
print('R^2', modelres.rsquared)
print("rozptyl závislé proměnné", modelres.mse_resid)

z1 = modelres.predict(sm.add_constant(
    np.column_stack((x1.ravel(), x1.ravel() ** 2, y1.ravel() ** 2)))).reshape(
    x1.shape)
plot_3d(x, y, z, x1, y1, z1)


# Pro test rovnosti na nulu dvou koeficientů zárověn jsem zvolil F-test s $B_0$ a $B_1$.
# $H_0$: $B_0 = B_1 = 0$
# 

# In[ ]:


print(modelres.f_test("const = x1 = 0"))


# Na hladině významnosti $\alpha = 0.05$ ((P>|t|) < 0.05) zamítáme $H_0$, koeficienty se tudíž zároveň liší od nuly.

# 

# Pro test rovnosti dvou koeficientů jsem zvolil T-test mezi $B_0$ a $B_1$.
# $H_0$: $B_0 = B_1$
# 

# In[ ]:


print(modelres.t_test("const = x1 "))


# Na hladině významnosti $\alpha = 0.05$ ((P>|t|) < 0.05) zamítáme $H_0$, koeficienty se tudíž liší.
