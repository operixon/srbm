SRBM
====

Opis
----

Implementacja SRBM w oparciu o pracę <B>Sparse deep belief net models for visual area V2<B> autorstwa <B>Chaitanya Ekanadham</B>.

Funkcjonalności
---------------

Obecna wersja posiada :
1. Testową implementację algorytmu
2. Podpięty zbiór trenujący z bazy MINST
3. Proste GUI

Uruchomienie z testu
--------------------

Do uruchomienia testowego aplikacji można użyć testu org.wit.srbm.SrbmNetworkNGTest#testLearning().

Samodzielne uruchomienie sieci
------------------------------

``` Java
        SRBM algorithm = new SRBM();
        algorithm.train();
```

Konfiguracja
------------

Do konfiguracji parametrów sieci służy klasa Configuration.

Obsługa
-------

Po uruchomieniu testu pokaże się okno z graficzną reprezentacją kolumn macieży wag oraz paczka ucząca.
Na wyjściu konsoli będzie widać numer epoki, współczynnik błędu i czasy wykonania poszczgólnych kroków sieci.

