import pandas as pd
#Kullanılacak veri setini ekleme
dataset = pd.read_excel('eye_tracker_data.xlsx')

print(dataset.shape)

X = dataset.iloc[:,[4,6,7]]
y = dataset['PicStatus']

#Verilerimizi eğitim ve test olarak ayırma işlemi. Veri setinin %75'ini eğitim %25'unu test olarak ayırdık
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

#Normalizasyon işlemi

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#scikit-learn kütüphanesinden aldığımız KNeighborsClassifier sınıfını kullanıp knn objesini oluşturuyoruz ve sınıflandırma işlemini gerçekleştiriyoruz
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
#Eğitim setleri kullanılarak öğrenme işlemi yapıyoruz
knn.fit(X_train, y_train)

#Oluşturduğumuz model için eğitim ve test sonuçlarını yüzdesel olarak yazdırıyoruz
print('K-NN sınıflandırıcısının eğitim setindeki doğruluk oranı: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('K-NN sınıflandırıcısının test setindeki doğruluk oranı: {:.2f}'
     .format(knn.score(X_test, y_test)))

#scikit-learn kütüphanesinden aldığımız LogisticRegression sınıfını kullanıp logreg objesini oluşturuyoruz ve sınıflandırma işlemini gerçekleştiriyoruz
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#Eğitim setleri kullanılarak öğrenme işlemi yapıyoruz
logreg.fit(X_train, y_train)

#Oluşturduğumuz model için eğitim ve test sonuçlarını yüzdesel olarak yazdırıyoruz
print('Lojistik Regresyon sınıflandırıcısının eğitim setindeki doğruluk oranı: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Lojistik Regresyon sınıflandırıcısının test setindeki doğruluk oranı: {:.2f}'
     .format(logreg.score(X_test, y_test)))

#scikit-learn kütüphanesinden aldığımız DecisionTreeClassifier sınıfını kullanıp clf objesini oluşturuyoruz ve sınıflandırma işlemini gerçekleştiriyoruz
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
#Eğitim setleri kullanılarak öğrenme işlemi yapıyoruz
clf.fit(X_train, y_train)

#Oluşturduğumuz model için eğitim ve test sonuçlarını yüzdesel olarak yazdırıyoruz
print('Karar Ağacı sınıflandırıcısının eğitim setindeki doğruluk oranı: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Karar Ağacı sınıflandırıcısının test setindeki doğruluk oranı: {:.2f}'
     .format(clf.score(X_test, y_test)))

#scikit-learn kütüphanesinden aldığımız LinearDiscriminantAnalysis sınıfını kullanıp lda objesini oluşturuyoruz ve sınıflandırma işlemini gerçekleştiriyoruz
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
#Eğitim setleri kullanılarak öğrenme işlemi yapıyoruz
lda.fit(X_train, y_train)

#Oluşturduğumuz model için eğitim ve test sonuçlarını yüzdesel olarak yazdırıyoruz
print('Doğrusal Ayrımcılık Analizi eğitim setindeki doğruluk oranı: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Doğrusal Ayrımcılık Analizi sınıflandırıcısının test setindeki doğruluk oranı: {:.2f}'
     .format(lda.score(X_test, y_test)))

#scikit-learn kütüphanesinden aldığımız GaussianNB sınıfını kullanıp gnb objesini oluşturuyoruz ve sınıflandırma işlemini gerçekleştiriyoruz
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#Eğitim setleri kullanılarak öğrenme işlemi yapıyoruz
gnb.fit(X_train, y_train)

#Oluşturduğumuz model için eğitim ve test sonuçlarını yüzdesel olarak yazdırıyoruz
print('Naive Bayes sınıflandırıcısının eğitim setindeki doğruluk oranı: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Naive Bayes sınıflandırıcısının test setindeki doğruluk oranı: {:.2f}'
     .format(gnb.score(X_test, y_test)))

#scikit-learn kütüphanesinden aldığımız SVC sınıfını kullanıp svm objesini oluşturuyoruz ve sınıflandırma işlemini gerçekleştiriyoruz
from sklearn.svm import SVC
svm = SVC()
#Eğitim setleri kullanılarak öğrenme işlemi yapıyoruz
svm.fit(X_train, y_train)

#Oluşturduğumuz model için eğitim ve test sonuçlarını yüzdesel olarak yazdırıyoruz
print('Destek Vektör Makinesi sınıflandırıcısının eğitim setindeki doğruluk oranı: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Destek Vektör Makinesi sınıflandırıcısının test setindeki doğruluk oranı: {:.2f}\n'
     .format(svm.score(X_test, y_test)))

# Hata matrisi ve sınıflandırma raporları için gerekli kütüphanelerin tanımlanması
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# X_test setimizi kullanarak oluşturduğumuz model için tamin yürütelim
# Elde ettiğimiz set(..._y_pred) ile hedef (y_test) test setimizi karşılaştıralım
# Hata matris ve sınıflandırma raporları
k_y_pred = knn.predict(X_test)
print("KNeighborsClassifier sınıflandırması için matris hata oranları\n")
print(confusion_matrix(y_test, k_y_pred))
print("KNeighborsClassifier sınıflandırma raporu\n")
print(classification_report(y_test, k_y_pred))

l_y_pred = logreg.predict(X_test)
print("LogisticRegression  için matris hata oranları\n")
print(confusion_matrix(y_test, l_y_pred))
print("LogisticRegression sınıflandırma raporu\n")
print(classification_report(y_test, l_y_pred))

c_y_pred = clf.predict(X_test)
print("DecisionTreeClassifier  için matris hata oranları\n")
print(confusion_matrix(y_test, c_y_pred))
print("DecisionTreeClassifier sınıflandırma raporu\n")
print(classification_report(y_test, c_y_pred))

lda_y_pred = lda.predict(X_test)
print("LinearDiscriminantAnalysis  için matris hata oranları\n")
print(confusion_matrix(y_test, lda_y_pred))
print("LinearDiscriminantAnalysis sınıflandırma raporu\n")
print(classification_report(y_test, lda_y_pred))

g_y_pred = gnb.predict(X_test)
print("GaussianNB  için matris hata oranları\n")
print(confusion_matrix(y_test, g_y_pred))
print("GaussianNB sınıflandırma raporu\n")
print(classification_report(y_test, g_y_pred))

s_y_pred = svm.predict(X_test)
print("Support Vector Machine  için matris hata oranları\n")
print(confusion_matrix(y_test, s_y_pred))
print("Support Vector Machine sınıflandırma raporu\n")
print(classification_report(y_test, s_y_pred))


