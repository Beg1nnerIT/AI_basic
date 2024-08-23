import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

smak = ctrl.Antecedent(np.arange(0, 11, 1), 'smak')
pikantnosc = ctrl.Antecedent(np.arange(0, 11, 1), 'pikantnosc')
konsystencja = ctrl.Antecedent(np.arange(0, 11, 1), 'konsystencja')
aromat = ctrl.Antecedent(np.arange(0, 11, 1), 'aromat')
slodkosc = ctrl.Antecedent(np.arange(0, 11, 1), 'slodkosc')
przydatnosc = ctrl.Consequent(np.arange(0, 11, 1), 'przydatnosc')

# Definicja funkcji przynależności
smak['bardzo_niedobry'] = fuzz.trimf(smak.universe, [0, 0, 5])
smak['dobry'] = fuzz.trimf(smak.universe, [0, 5, 10])
smak['bardzo_dobry'] = fuzz.trimf(smak.universe, [5, 10, 10])

pikantnosc['niska'] = fuzz.trimf(pikantnosc.universe, [0, 0, 5])
pikantnosc['umiarkowana'] = fuzz.trimf(pikantnosc.universe, [0, 5, 10])
pikantnosc['wysoka'] = fuzz.trimf(pikantnosc.universe, [5, 10, 10])

konsystencja['rzadka'] = fuzz.trimf(konsystencja.universe, [0, 0, 5])
konsystencja['srednia'] = fuzz.trimf(konsystencja.universe, [0, 5, 10])
konsystencja['gesta'] = fuzz.trimf(konsystencja.universe, [5, 10, 10])

aromat['slaby'] = fuzz.trimf(aromat.universe, [0, 0, 5])
aromat['umiarkowany'] = fuzz.trimf(aromat.universe, [0, 5, 10])
aromat['silny'] = fuzz.trimf(aromat.universe, [5, 10, 10])

slodkosc['niska'] = fuzz.trimf(slodkosc.universe, [0, 0, 5])
slodkosc['srednia'] = fuzz.trimf(slodkosc.universe, [0, 5, 10])
slodkosc['wysoka'] = fuzz.trimf(slodkosc.universe, [5, 10, 10])

przydatnosc['niewlasciwa'] = fuzz.trimf(przydatnosc.universe, [0, 0, 5])
przydatnosc['umiarkowana'] = fuzz.trimf(przydatnosc.universe, [0, 5, 10])
przydatnosc['wysoka'] = fuzz.trimf(przydatnosc.universe, [5, 10, 10])

# Zdefiniuj reguły
regula1 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['wysoka'] & konsystencja['rzadka'] & aromat['slaby'] & slodkosc['niska'], przydatnosc['wysoka'])
regula2 = ctrl.Rule(smak['dobry'] & pikantnosc['umiarkowana'] & konsystencja['srednia'] & aromat['umiarkowany'] & slodkosc['srednia'], przydatnosc['umiarkowana'])
regula3 = ctrl.Rule(smak['bardzo_niedobry'] | pikantnosc['niska'] | konsystencja['gesta'] | aromat['silny'] | slodkosc['wysoka'], przydatnosc['niewlasciwa'])
regula4 = ctrl.Rule(smak['dobry'] & pikantnosc['niska'] & konsystencja['rzadka'] & aromat['slaby'] & slodkosc['niska'], przydatnosc['niewlasciwa'])
regula5 = ctrl.Rule(smak['bardzo_niedobry'] & pikantnosc['niska'] & konsystencja['rzadka'] & aromat['slaby'] & slodkosc['niska'], przydatnosc['niewlasciwa'])
regula6 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['umiarkowana'] & konsystencja['srednia'] & aromat['slaby'] & slodkosc['niska'], przydatnosc['umiarkowana'])
regula7 = ctrl.Rule(smak['dobry'] & pikantnosc['niska'] & konsystencja['rzadka'] & aromat['umiarkowany'] & slodkosc['niska'], przydatnosc['niewlasciwa'])
regula8 = ctrl.Rule(smak['bardzo_niedobry'] & pikantnosc['wysoka'] & konsystencja['rzadka'] & aromat['silny'] & slodkosc['wysoka'], przydatnosc['niewlasciwa'])
regula9 = ctrl.Rule(smak['dobry'] & pikantnosc['umiarkowana'] & konsystencja['rzadka'] & aromat['umiarkowany'] & slodkosc['niska'], przydatnosc['wysoka'])
regula10 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['wysoka'] & konsystencja['gesta'] & aromat['silny'] & slodkosc['wysoka'], przydatnosc['wysoka'])
regula11 = ctrl.Rule(smak['dobry'] & pikantnosc['niska'] & konsystencja['srednia'] & aromat['umiarkowany'] & slodkosc['srednia'], przydatnosc['wysoka'])
regula12 = ctrl.Rule(smak['bardzo_niedobry'] | pikantnosc['niska'] | konsystencja['rzadka'] | aromat['slaby'] | slodkosc['niska'], przydatnosc['niewlasciwa'])
regula13 = ctrl.Rule(smak['dobry'] & pikantnosc['umiarkowana'] & konsystencja['rzadka'] & aromat['slaby'] & slodkosc['wysoka'], przydatnosc['umiarkowana'])
regula14 = ctrl.Rule(smak['bardzo_niedobry'] & pikantnosc['wysoka'] & konsystencja['rzadka'] & aromat['slaby'] & slodkosc['niska'], przydatnosc['niewlasciwa'])
regula15 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['umiarkowana'] & konsystencja['srednia'] & aromat['slaby'] & slodkosc['niska'], przydatnosc['wysoka'])
regula16 = ctrl.Rule(smak['dobry'] & pikantnosc['niska'] & konsystencja['rzadka'] & aromat['umiarkowany'] & slodkosc['niska'], przydatnosc['niewlasciwa'])
regula17 = ctrl.Rule(smak['bardzo_niedobry'] & pikantnosc['wysoka'] & konsystencja['rzadka'] & aromat['silny'] & slodkosc['wysoka'], przydatnosc['niewlasciwa'])
regula18 = ctrl.Rule(smak['dobry'] & pikantnosc['umiarkowana'] & konsystencja['rzadka'] & aromat['umiarkowany'] & slodkosc['niska'], przydatnosc['wysoka'])
regula19 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['wysoka'] & konsystencja['gesta'] & aromat['silny'] & slodkosc['wysoka'], przydatnosc['wysoka'])
regula20 = ctrl.Rule(smak['dobry'] & pikantnosc['niska'] & konsystencja['srednia'] & aromat['umiarkowany'] & slodkosc['srednia'], przydatnosc['wysoka'])
regula21 = ctrl.Rule(smak['bardzo_niedobry'] | pikantnosc['niska'] | konsystencja['rzadka'] | aromat['slaby'] | slodkosc['niska'], przydatnosc['niewlasciwa'])
regula22 = ctrl.Rule(smak['dobry'] & pikantnosc['umiarkowana'] & konsystencja['rzadka'] & aromat['slaby'] & slodkosc['wysoka'], przydatnosc['umiarkowana'])
regula23 = ctrl.Rule(smak['bardzo_niedobry'] & pikantnosc['wysoka'] & konsystencja['rzadka'] & aromat['slaby'] & slodkosc['niska'], przydatnosc['niewlasciwa'])
regula24 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['umiarkowana'] & konsystencja['srednia'] & aromat['slaby'] & slodkosc['niska'], przydatnosc['wysoka'])
regula25 = ctrl.Rule(smak['dobry'] & pikantnosc['niska'] & konsystencja['rzadka'] & aromat['umiarkowany'] & slodkosc['niska'], przydatnosc['niewlasciwa'])
regula26 = ctrl.Rule(smak['bardzo_niedobry'] & pikantnosc['wysoka'] & konsystencja['rzadka'] & aromat['silny'] & slodkosc['wysoka'], przydatnosc['niewlasciwa'])
regula27 = ctrl.Rule(smak['dobry'] & pikantnosc['umiarkowana'] & konsystencja['rzadka'] & aromat['umiarkowany'] & slodkosc['niska'], przydatnosc['wysoka'])
regula28 = ctrl.Rule(smak['bardzo_dobry'] & pikantnosc['wysoka'] & konsystencja['gesta'] & aromat['silny'] & slodkosc['wysoka'], przydatnosc['wysoka'])
regula29 = ctrl.Rule(smak['dobry'] & pikantnosc['niska'] & konsystencja['srednia'] & aromat['umiarkowany'] & slodkosc['srednia'], przydatnosc['wysoka'])
regula30 = ctrl.Rule(smak['bardzo_niedobry'] | pikantnosc['niska'] | konsystencja['rzadka'] | aromat['slaby'] | slodkosc['niska'], przydatnosc['niewlasciwa'])

system_oceny = ctrl.ControlSystem([regula1, regula2, regula3, regula4, regula5, regula6, regula7, regula8, regula9, regula10, regula11, regula12, regula13, regula14, regula15, regula16, regula17, regula18, regula19, regula20, regula21, regula22, regula23, regula24, regula25, regula26, regula27, regula28, regula29, regula30])  # Include all 30 rules
oslodkosc_potrawy = ctrl.ControlSystemSimulation(system_oceny)

# Krok 3: Implementuj wnioskowanie rozmyte
def ocen_potrawe(smak_val, pikantnosc_val, konsystencja_val, aromat_val, slodkosc_val):
    oslodkosc_potrawy.input['smak'] = smak_val
    oslodkosc_potrawy.input['pikantnosc'] = pikantnosc_val
    oslodkosc_potrawy.input['konsystencja'] = konsystencja_val
    oslodkosc_potrawy.input['aromat'] = aromat_val
    oslodkosc_potrawy.input['slodkosc'] = slodkosc_val
    oslodkosc_potrawy.compute()
    return oslodkosc_potrawy.output['przydatnosc']


def generuj_zestawy_testowe(liczba_zestawow=10):
    zestawy_testowe = []
    for _ in range(liczba_zestawow):
        smak_val = np.random.uniform(0, 10)
        pikantnosc_val = np.random.uniform(0, 10)
        konsystencja_val = np.random.uniform(0, 10)
        aromat_val = np.random.uniform(0, 10)
        slodkosc_val = np.random.uniform(0, 10)

        zestaw = {
            'smak': smak_val,
            'pikantnosc': pikantnosc_val,
            'konsystencja': konsystencja_val,
            'aromat': aromat_val,
            'slodkosc': slodkosc_val,
        }
        zestawy_testowe.append(zestaw)

    return zestawy_testowe

def testuj_zestawy_testowe(zestawy_testowe):
    for i, zestaw in enumerate(zestawy_testowe, 1):
        wynik_oceny = ocen_potrawe(zestaw['smak'], zestaw['pikantnosc'], zestaw['konsystencja'], zestaw['aromat'], zestaw['slodkosc'])
        print(f"Zestaw testowy {i}:")
        print(f"Oceniony smak: {zestaw['smak']}, Oceniona pikantność: {zestaw['pikantnosc']}, Oceniona konsystencja: {zestaw['konsystencja']}, Oceniony aromat: {zestaw['aromat']}, Oceniona słodkość: {zestaw['slodkosc']}")
        print(f"Ocena przydatności potrawy: {wynik_oceny:.2f}")
        print()

def interfejs_uzytkownika():
    print("Wybierz tryb:")
    print("1. Testy zestawów danych")
    print("2. Wprowadź własne dane")

    try:
        tryb = int(input("Wybór (1/2): "))
        if tryb == 1:
            liczba_zestawow = int(input("Podaj liczbę zestawów testowych do wygenerowania: "))
            zestawy_testowe = generuj_zestawy_testowe(liczba_zestawow)
            testuj_zestawy_testowe(zestawy_testowe)
        elif tryb == 2:
            smak_val = get_user_input("Podaj ocenę smaku (0-10): ")
            pikantnosc_val = get_user_input("Podaj ocenę pikantności (0-10): ")
            konsystencja_val = get_user_input("Podaj ocenę konsystencji (0-10): ")
            aromat_val = get_user_input("Podaj ocenę aromatu (0-10): ")
            slodkosc_val = get_user_input("Podaj ocenę słodkości (0-10): ")

            wynik_oceny = ocen_potrawe(smak_val, pikantnosc_val, konsystencja_val, aromat_val, slodkosc_val)

            print(f"Ocena przydatności potrawy: {wynik_oceny:.2f}")
        else:
            print("Nieprawidłowy wybór. Wybierz 1 lub 2.")
    except ValueError:
        print("Nieprawidłowy format. Wprowadź liczbę.")

def get_user_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if 0 <= value <= 10:
                return value
            else:
                print("Wartość powinna być w zakresie od 0 do 10. Spróbuj ponownie.")
        except ValueError:
            print("Nieprawidłowy format. Wprowadź liczbę.")

# Krok 6: Testowanie i ewaluacja
interfejs_uzytkownika()
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(smak.universe, fuzz.trimf(smak.universe, [0, 0, 5]), 'b', linewidth=1.5, label='Bardzo Niedobry')
ax0.plot(smak.universe, fuzz.trimf(smak.universe, [0, 5, 10]), 'g', linewidth=1.5, label='Dobry')
ax0.plot(smak.universe, fuzz.trimf(smak.universe, [5, 10, 10]), 'r', linewidth=1.5, label='Bardzo Dobry')
ax0.set_title('Smak')
ax0.legend()

# Funkcje przynależności dla pikantności
ax1.plot(pikantnosc.universe, fuzz.trimf(pikantnosc.universe, [0, 0, 5]), 'b', linewidth=1.5, label='Niska')
ax1.plot(pikantnosc.universe, fuzz.trimf(pikantnosc.universe, [0, 5, 10]), 'g', linewidth=1.5, label='Umiarkowana')
ax1.plot(pikantnosc.universe, fuzz.trimf(pikantnosc.universe, [5, 10, 10]), 'r', linewidth=1.5, label='Wysoka')
ax1.set_title('Pikantność')
ax1.legend()

# Funkcje przynależności dla konsystencji
ax2.plot(konsystencja.universe, fuzz.trimf(konsystencja.universe, [0, 0, 5]), 'b', linewidth=1.5, label='Rzadka')
ax2.plot(konsystencja.universe, fuzz.trimf(konsystencja.universe, [0, 5, 10]), 'g', linewidth=1.5, label='Średnia')
ax2.plot(konsystencja.universe, fuzz.trimf(konsystencja.universe, [5, 10, 10]), 'r', linewidth=1.5, label='Gęsta')
ax2.set_title('Konsystencja')
ax2.legend()

# Funkcje przynależności dla aromatu
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(aromat.universe, fuzz.trimf(aromat.universe, [0, 0, 5]), 'b', linewidth=1.5, label='Słaby')
ax0.plot(aromat.universe, fuzz.trimf(aromat.universe, [0, 5, 10]), 'g', linewidth=1.5, label='Umiarkowany')
ax0.plot(aromat.universe, fuzz.trimf(aromat.universe, [5, 10, 10]), 'r', linewidth=1.5, label='Silny')
ax0.set_title('Aromat')
ax0.legend()

# Funkcje przynależności dla słodyczy
ax1.plot(slodkosc.universe, fuzz.trimf(slodkosc.universe, [0, 0, 5]), 'b', linewidth=1.5, label='Niska')
ax1.plot(slodkosc.universe, fuzz.trimf(slodkosc.universe, [0, 5, 10]), 'g', linewidth=1.5, label='Średnia')
ax1.plot(slodkosc.universe, fuzz.trimf(slodkosc.universe, [5, 10, 10]), 'r', linewidth=1.5, label='Wysoka')
ax1.set_title('Słodycz')
ax1.legend()

# Funkcje przynależności dla przydatności
ax2.plot(przydatnosc.universe, fuzz.trimf(przydatnosc.universe, [0, 0, 5]), 'b', linewidth=1.5, label='Niewłaściwa')
ax2.plot(przydatnosc.universe, fuzz.trimf(przydatnosc.universe, [0, 5, 10]), 'g', linewidth=1.5, label='Umiarkowana')
ax2.plot(przydatnosc.universe, fuzz.trimf(przydatnosc.universe, [5, 10, 10]), 'r', linewidth=1.5, label='Wysoka')
ax2.set_title('Przydatność')
ax2.legend()

# ... (wykresy dla pozostałych zmiennych)

# Wykresy dla reguł wnioskowania
fig, ax = plt.subplots(figsize=(8, 4))
rules = ctrl.ControlSystemSimulation(system_oceny)
fuzzy_result = []

for smak_val, pikantnosc_val, konsystencja_val, aromat_val, slodkosc_val in zip(
        smak.universe, pikantnosc.universe, konsystencja.universe, aromat.universe, slodkosc.universe):
    rules.input['smak'] = smak_val
    rules.input['pikantnosc'] = pikantnosc_val
    rules.input['konsystencja'] = konsystencja_val
    rules.input['aromat'] = aromat_val
    rules.input['slodkosc'] = slodkosc_val
    rules.compute()
    fuzzy_result.append(rules.output['przydatnosc'])

ax.plot(smak.universe, fuzzy_result, 'b', linewidth=1.5, label='Przydatność')
ax.set_title('Wpływ zmiennych na przydatność potrawy')
ax.legend()

plt.tight_layout()
plt.show()
