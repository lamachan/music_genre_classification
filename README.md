# Music genre classification
## Cel projektu
Projekt *"Rozpoznawanie gatunków muzycznych z wykorzystaniem technik uczenia maszynowego"* realizowany jest w ramach przedmiotu *Projekt indywidualny* na 4. semestrze studiów inżynierskich *Informatyka stosowana* na *Wydziale Elektrycznym Politechniki Warszawskiej*.

Celem projektu jest stworzenie aplikacji dokonującej klasyfikacji gatunków utworów muzycznych. W ramach projektu należy stworzyć własną bazę utworów, dokonać ekstrakcji cech oraz na ich podstawie wytrenować i wdrożyć model uczenia maszynowego.
## Działanie aplikacji
### Front-end
Zaimplementowano prostą aplikację okienkową umożliwiającą klasyfikację utworu wgranego przez użytkownika jako plik dźwiękowy. Możliwe jest wgranie pliku w formacie *.mp3* lub *.wav*. Przed wgraniem pliku należy wybrać 1 z 3 dostępnych modeli z listy rozwijanej (`Logistic Regression`, `Decision Tree` lub `Random Forest`). Po ok. 6 sekundach wyświetlane jest przewidywanie gatunku.
### Back-end
Z wgranego pliku wycinane jest 5 20-sekundowych próbek (po kolei, od początku utworu) i ekstrahowane są cechy (76 ostatecznie wybranych do klasyfikacji na etapie analizy eksploracyjnej danych). W wyniku tego powstaje ramka danych o rozmiarze 5 x 76. Za pomocą przelicznika dopasowanego do zbioru uczącego, cechy są standaryzowane. Następnie ramka przekazywana jest do wybranego modelu. Każda z 5 próbek otrzymuje etykietę (gatunek). Jako ostateczna etykieta wybierany jest gatunek o największej liczbie wystąpień (głosowanie większościowe). W przypadku takiej samej liczby głosów na więcej niż 1 gatunek, etykieta wybierana jest według porządku alfabetycznego. Obsługiwane gatunki: `choir`, `classical`, `electronic`, `folk`, `jazz`, `metal`, `pop`, `rap`, `reggae`, `rock`.

## Struktura repozytorium
* *__dane__* - katalog z plikami tekstowymi zawierającymi zbiór danych:
	* __music_data1.csv__ - plik tekstowy ze zbiorem danych po etapie ekstrakcji danych
	* __music_data2.csv__ - plik tekstowy ze zbiorem danych po etapie eksploracyjnej analizy danych
* __librosa_testing.ipynb__ - testowanie funkcji biblioteki *librosa*
* __feature_extraction.ipynb__ - ekstrakcja danych
* __data_mining.ipynb__ - eksploracyjna analiza danych
* __classification_models.ipynb__ - uczenie i ocena modeli
* *__classification_app__* - katalog z plikami dotyczącymi aplikacji okienkowej:
	* *__pickles__* - katalog z plikami utrwalającymi typu *.pkl*:
		* __scaler.pkl__ - przelicznik dokonujący standaryzacji
		* __decision_tree_model.pkl__ - model drzewa decyzyjnego
		* __logistic_regression_model.pkl__ - model regresji logistycznej
		* __random_forest_model.pkl__ - model lasu losowego
	* __feature_extraction.py__ - moduł zawierający funkcje ekstrahujące cechy z pliku dźwiękowego
	* __app.py__ - logika aplikacji i GUI
	* __classification_app_gui.png__ - zrzut ekranu GUI aplikacji
