# zadani 4
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm

#data 1
city = np.array(['big', 'big', 'medium', 'medium', 'medium', 'small', 'small', 'small'])
cat = np.array(['Praha', 'Brno', 'Znojmo', 'Tišnov', 'Rokytnice nad Jizerou', 'Jablunkov', 'Dolní Věstonice', 'student'])
count = np.array([1327, 915, 681, 587, 284, 176, 215, 21])
winter = np.array([510, 324, 302, 257, 147, 66, 87, 11])
summer = np.array([352, 284, 185, 178, 87, 58, 65, 4])
change = np.array([257, 178, 124, 78, 44, 33, 31, 4])
indifferent = np.array([208, 129, 70, 74, 6, 19, 32, 2])

def chi_test(cat, count):
    print("got: ",cat.astype(int), "\nexp: ", (([cat.sum() / count.sum()] * len(count)) * count).astype(int),'\n')
    print(stats.chisquare(cat, f_exp=([cat.sum() / count.sum()] * len(count)) * count))

#chi_test(winter, count) # <0.05 => zamitame H0, je rozdil mezi mesty
#chi_test(summer, count)
#chi_test(change, count)
#print(stats.chisquare([1,15,0,15,6,15,44,30])),print(stats.chisquare([15]*10)) TRASH
#print("critical value: ", stats.chi2.ppf(0.95, 7))

# group by city
city_group = pd.DataFrame({'city': city, 'count': count, 'winter': winter, 'indifferent': indifferent})
city_group = city_group.groupby('city').sum()
#print(city_group)
#chi_test(city_group['winter'], city_group['count'])

# group by city
city_group = pd.DataFrame({'city': city, 'cat': cat, 'count': count, 'winter': winter, 'summer': summer, 'change': change, 'indifferent': indifferent})
city_group = city_group[city_group['cat'] != 'student']
city_group = city_group.drop('cat', axis=1)

city_group = city_group.groupby(['city']).sum()
city_group.loc['student'] = [21, 11, 4, 4, 2]
# separate count column
counts = city_group['count']
city_group = city_group.drop('count', axis=1)


#print(city_group)

def chi_test2(cat, exp):
    print("got: ",cat.astype(int), "\nexp: ", exp.astype(int),'\n')
    print(stats.chisquare(cat, f_exp=exp))
#chi_test2 student to big city
exp = np.array([city_group.loc['big'] / counts['big'] * counts['student']])
#exp to array
exp = np.array([exp[0][0], exp[0][1], exp[0][2], exp[0][3]])
print(exp)

chi_test2(city_group.loc['student'], exp)







exit(0)

# data2
x = np.array(['0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '2.22', '2.22', '2.22', '2.22', '2.22', '2.22', '2.22', '4.44', '4.44', '4.44', '4.44', '4.44', '4.44', '4.44', '6.67', '6.67', '6.67', '6.67', '6.67', '6.67', '6.67', '8.89', '8.89', '8.89', '8.89', '8.89', '8.89', '8.89', '11.11', '11.11', '11.11', '11.11', '11.11', '11.11', '11.11', '13.33', '13.33', '13.33', '13.33', '13.33', '13.33', '13.33', '15.56', '15.56', '15.56', '15.56', '15.56', '15.56', '15.56', '17.78', '17.78', '17.78', '17.78', '17.78', '17.78', '17.78', '20.00', '20.00', '20.00', '20.00', '20.00', '20.00', '20.00'], dtype='float64')
y = np.array(['0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00', '0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00', '0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00', '0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00', '0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00', '0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00', '0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00', '0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00', '0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00', '0.00', '1.67', '3.33', '5.00', '6.67', '8.33', '10.00'], dtype='float64')
z = np.array(['4.56', '-0.99', '-10.31', '-21.32', '-40.17', '-63.45', '-94.17', '9.5', '8.33', '0.42', '-13.32', '-33.5', '-56.82', '-89.45', '28.83', '26.2', '17.08', '3.25', '-16.63', '-40.82', '-71.68', '54.14', '52.16', '44.52',     '29.63', '8.93', '-15.99', '-44.98', '92.64', '89.58', '80.04', '68.89', '49.6', '22.64', '-9.2', '138.35',     '135.44', '125.32', '113.13', '94.98', '68.49', '39.02', '194.73', '192.12', '184.08', '171.76', '148.17',     '127.48', '94.54', '260.47', '258.41', '249.98', '234.49', '215.88', '192.3', '159.45', '339.77', '333.69', '326.48', '311.62', '293.13', '268.29', '237.57', '423.51', '421.43', '414.2', '399.43', '377.86', '354.77', '324.31'], dtype='float64')

F = np.column_stack((x, y, x**2, y**2, x*y))
F = sm.add_constant(F)
modelres = sm.OLS(z, F).fit()
print(modelres.summary())

#plot prediction model
x1 = np.linspace(0, 20, 100)
y1 = np.linspace(0, 10, 100)
x1, y1 = np.meshgrid(x1, y1)
z1 = modelres.predict(sm.add_constant(np.column_stack((x1.ravel(), y1.ravel(), x1.ravel()**2, y1.ravel()**2, x1.ravel()*y1.ravel())))).reshape(x1.shape)
fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x1, y1, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#add data points
ax.scatter(x, y, z, c='r', marker='x')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#rotate view
ax.view_init(15, 75)
plt.show()

#print(modelres.f_test("const = x1 = 0"))