**Zelle 1 führst du nicht im normalen Terminal aus**, sondern **direkt im Jupyter Notebook oder in Google Colab**.

Also diese Zeile:

```python
!pip install -q pandas numpy matplotlib seaborn scikit-learn tensorflow ucimlrepo
```

kommt in eine **Notebook-Zelle** und wird dort mit **Shift + Enter** ausgeführt.

---

## Falls du lokal mit Jupyter arbeitest

Dann ist die Reihenfolge so:

### 1. Erst im normalen Terminal / PowerShell / Ubuntu-Terminal:

```bash
cd dein/projektordner
jupyter notebook
```

oder:

```bash
jupyter lab
```

Danach öffnet sich Jupyter im Browser.

### 2. Im Browser erstellst du ein neues Notebook

Dann fügst du dort als erste Code-Zelle ein:

```python
!pip install -q pandas numpy matplotlib seaborn scikit-learn tensorflow ucimlrepo
```

Dann ausführst du die Zelle mit:

```text
Shift + Enter
```

---

## Falls du Google Colab nutzt

Dann öffnest du einfach dein Colab-Notebook und führst **Zelle 1 direkt dort** aus.

---

## Wichtig

Das Ausrufezeichen `!` bedeutet:

```python
!pip install ...
```

Jupyter soll diesen Befehl wie einen Terminal-Befehl ausführen.

Deshalb funktioniert das in einer Notebook-Zelle.

Im normalen Terminal würdest du stattdessen **ohne Ausrufezeichen** schreiben:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow ucimlrepo
```

Meine Empfehlung für dich: Nutze **Google Colab**, dann musst du dich weniger mit lokaler Installation beschäftigen.
