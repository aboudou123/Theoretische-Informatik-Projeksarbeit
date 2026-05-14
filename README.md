
---

# Schritt 0: Neues Notebook erstellen

## Lokal

Im Terminal:

```bash
mkdir student_dropout_project
cd student_dropout_project
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

pip install jupyter pandas numpy matplotlib seaborn scikit-learn tensorflow ucimlrepo
jupyter notebook
```

Dann im Browser ein neues Notebook erstellen.

## Google Colab

Öffne Google Colab, erstelle ein neues Notebook und führe direkt die Zellen unten aus.

---

# Zelle 1: Bibliotheken installieren

```python
!pip install -q pandas numpy matplotlib seaborn scikit-learn tensorflow ucimlrepo
```

---

# Zelle 2: Imports und Reproduzierbarkeit

```python
import os
import random
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

sns.set_theme(style="whitegrid")

print("Setup erfolgreich.")
print("TensorFlow-Version:", tf.__version__)
```

Nach dieser Zelle solltest du sehen:

```text
Setup erfolgreich.
TensorFlow-Version: ...
```

---

# Zelle 3: Datensatz laden

```python
dataset = fetch_ucirepo(id=697)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

df = X.copy()
df["Target"] = y

print("Datensatz erfolgreich geladen.")
print("Shape:", df.shape)

df.head()
```

Erwartung:

```text
Datensatz erfolgreich geladen.
Shape: (4424, 37)
```

Das bedeutet:

* 4424 Zeilen
* 36 Eingabefeatures
* 1 Zielvariable `Target`

---

# Zelle 4: Erste Übersicht

```python
print("Shape des Datensatzes:")
display(df.shape)

print("\nErste fünf Zeilen:")
display(df.head())

print("\nDatentypen und Non-Null-Werte:")
df.info()

print("\nDeskriptive Statistik:")
display(df.describe().T)
```

Hier prüfst du:

* Welche Spalten existieren?
* Sind alle Features numerisch?
* Gibt es offensichtliche Ausreißer?
* Welche Wertebereiche haben die Features?

---

# Zelle 5: Fehlende Werte prüfen

```python
missing_values = df.isnull().sum().sort_values(ascending=False)

print("Fehlende Werte pro Spalte:")
display(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("Es gibt keine fehlenden Werte im Datensatz.")
else:
    print("Es gibt fehlende Werte. Diese werden später im Preprocessing behandelt.")
```

Interpretation für die Arbeit:

```text
Der Datensatz enthält keine fehlenden Werte. Trotzdem wird im Preprocessing eine Imputation eingebaut, damit die Pipeline robust bleibt und auch bei zukünftigen Daten funktioniert.
```

---

# Zelle 6: Zielvariable analysieren

```python
target_counts = df["Target"].value_counts()
target_percentages = df["Target"].value_counts(normalize=True) * 100

target_distribution = pd.DataFrame({
    "Anzahl": target_counts,
    "Prozent": target_percentages.round(2)
})

display(target_distribution)

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Target", order=target_counts.index)
plt.title("Klassenverteilung der Zielvariable")
plt.xlabel("Klasse")
plt.ylabel("Anzahl")
plt.show()
```

Interpretation:

```text
Die Zielvariable besteht aus drei Klassen: Dropout, Enrolled und Graduate.
Da jede Person genau einer dieser Klassen zugeordnet wird, handelt es sich um ein multiklassiges Klassifikationsproblem.

Die Klassen sind nicht perfekt gleich verteilt. Besonders wichtig ist deshalb ein stratified Train/Test Split, damit die Klassenverteilung im Trainings- und Testset vergleichbar bleibt.
```

---

# Zelle 7: Datentypen analysieren

```python
dtype_summary = df.dtypes.value_counts()

print("Datentypen im Datensatz:")
display(dtype_summary)

print("\nSpalten und Datentypen:")
display(df.dtypes)
```

Interpretation:

```text
Die meisten Eingabefeatures sind numerisch codiert. Einige Spalten repräsentieren jedoch kategoriale Informationen, zum Beispiel Course, Marital status oder Application mode.
Diese dürfen nicht wie echte metrische Zahlen interpretiert werden.
Deshalb werden kategoriale Spalten später per One-Hot-Encoding verarbeitet.
```

---

# Zelle 8: Kategorische und numerische Spalten definieren

```python
target_column = "Target"

categorical_columns = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]

categorical_columns = [col for col in categorical_columns if col in df.columns]

numeric_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

print("Kategorische Features:", len(categorical_columns))
display(categorical_columns)

print("Numerische Features:", len(numeric_columns))
display(numeric_columns)
```

Warum das wichtig ist:

```text
Kategorische Zahlen wie Course oder Marital status haben keine natürliche mathematische Reihenfolge.
One-Hot-Encoding verhindert, dass das Modell eine künstliche Rangordnung lernt.
Numerische Features wie Noten, Alter oder makroökonomische Werte werden skaliert.
```

---

# Zelle 9: Histogramme numerischer Features

```python
df[numeric_columns].hist(figsize=(18, 14), bins=30)
plt.suptitle("Histogramme numerischer Features", fontsize=16)
plt.tight_layout()
plt.show()
```

Interpretation:

```text
Die Histogramme zeigen, wie die numerischen Features verteilt sind.
Einige Variablen sind annähernd kontinuierlich, zum Beispiel Admission grade oder Age at enrollment.
Andere Features zeigen starke Häufungen bei bestimmten Werten, was im Bildungsdatensatz plausibel ist.
```

---

# Zelle 10: Boxplots für ausgewählte wichtige Features

```python
important_numeric_features = [
    "Admission grade",
    "Age at enrollment",
    "Previous qualification (grade)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)"
]

important_numeric_features = [
    col for col in important_numeric_features
    if col in df.columns
]

for col in important_numeric_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="Target", y=col)
    plt.title(f"{col} nach Zielklasse")
    plt.xlabel("Zielklasse")
    plt.ylabel(col)
    plt.show()
```

Interpretation:

```text
Die Boxplots zeigen Unterschiede zwischen den Klassen.
Besonders akademische Leistungsmerkmale wie bestandene oder bewertete Curricular Units unterscheiden sich häufig deutlich zwischen Dropout und Graduate.
Das ist fachlich plausibel, weil Studienleistung ein starker Indikator für Studienerfolg ist.
```

---

# Zelle 11: Korrelationsanalyse

```python
df_encoded_for_corr = df.copy()

label_encoder_corr = LabelEncoder()
df_encoded_for_corr["Target_encoded"] = label_encoder_corr.fit_transform(df_encoded_for_corr["Target"])

correlation_matrix = df_encoded_for_corr.drop(columns=["Target"]).corr(numeric_only=True)

plt.figure(figsize=(18, 14))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.3
)
plt.title("Korrelationsmatrix aller numerischen Features")
plt.show()
```

Interpretation:

```text
Die Korrelationsmatrix zeigt lineare Zusammenhänge zwischen Features.
Starke positive oder negative Korrelationen können auf redundante Informationen hinweisen.
Bei Bildungsdaten sind hohe Zusammenhänge zwischen Features der ersten und zweiten Studiensemester erwartbar.
```

---

# Zelle 12: Korrelation mit Zielvariable

```python
target_correlations = (
    df_encoded_for_corr
    .drop(columns=["Target"])
    .corr(numeric_only=True)["Target_encoded"]
    .sort_values(key=abs, ascending=False)
)

display(target_correlations.head(15))

plt.figure(figsize=(10, 6))
target_correlations.head(15).sort_values().plot(kind="barh")
plt.title("Top 15 absolute Korrelationen mit der Zielvariable")
plt.xlabel("Korrelationskoeffizient")
plt.ylabel("Feature")
plt.show()
```

Wichtig für die mündliche Prüfung:

```text
Korrelation bedeutet nicht automatisch Kausalität.
Ein Feature kann stark mit der Zielvariable zusammenhängen, ohne direkt die Ursache für Studienabbruch oder Studienerfolg zu sein.
```

---

# Zelle 13: Train/Test Split

```python
X = df.drop(columns=["Target"])
y = df["Target"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Klassen-Mapping:")
for index, class_name in enumerate(label_encoder.classes_):
    print(index, "=", class_name)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("Trainingsdaten:", X_train.shape)
print("Testdaten:", X_test.shape)

print("\nKlassenverteilung im Trainingsset:")
display(pd.Series(y_train).value_counts(normalize=True).sort_index())

print("\nKlassenverteilung im Testset:")
display(pd.Series(y_test).value_counts(normalize=True).sort_index())
```

Interpretation:

```text
Der Split ist stratified. Dadurch bleibt die Klassenverteilung in Training und Test ähnlich.
Das ist bei unbalancierten Klassifikationsproblemen wichtig.
```

---

# Zelle 14: Preprocessing Pipeline

```python
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_columns),
        ("categorical", categorical_transformer, categorical_columns)
    ]
)

print("Preprocessing-Pipeline erstellt.")
```

Erklärung:

```text
Numerische Features werden mit dem Median imputiert und anschließend standardisiert.
Kategorische Features werden mit dem häufigsten Wert imputiert und danach per One-Hot-Encoding umgewandelt.
Das gesamte Preprocessing wird nur auf den Trainingsdaten gelernt, um Data Leakage zu vermeiden.
```

---

# Zelle 15: Hilfsfunktion für Evaluation

```python
def evaluate_model(model_name, y_true, y_pred):
    """
    Berechnet zentrale Klassifikationsmetriken für ein Modell.
    """
    results = {
        "Modell": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }
    return results


def plot_confusion_matrix(y_true, y_pred, title):
    """
    Visualisiert die Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(title)
    plt.show()
```

---

# Zelle 16: Klassisches ML-Modell — Random Forest

```python
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
)

start_time = time.time()

rf_pipeline.fit(X_train, y_train)

rf_training_time = time.time() - start_time

rf_predictions = rf_pipeline.predict(X_test)

print("Random Forest Training abgeschlossen.")
print("Trainingszeit:", round(rf_training_time, 2), "Sekunden")
```

Warum Random Forest?

```text
Random Forest ist für tabellarische Daten sehr geeignet.
Das Modell kann nichtlineare Zusammenhänge erkennen, ist robust gegenüber Ausreißern und liefert Feature Importances.
Außerdem ist es einfacher interpretierbar als ein neuronales Netz.
```

---

# Zelle 17: Random Forest Evaluation

```python
rf_results = evaluate_model(
    model_name="Random Forest",
    y_true=y_test,
    y_pred=rf_predictions
)

display(pd.DataFrame([rf_results]))

print("Classification Report - Random Forest:")
print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

plot_confusion_matrix(
    y_true=y_test,
    y_pred=rf_predictions,
    title="Confusion Matrix - Random Forest"
)
```

Metriken erklären:

```text
Accuracy misst den Anteil korrekt klassifizierter Fälle.
Precision zeigt, wie zuverlässig positive Vorhersagen einer Klasse sind.
Recall zeigt, wie viele echte Fälle einer Klasse gefunden wurden.
F1-Score kombiniert Precision und Recall.
Macro-Metriken behandeln alle Klassen gleich.
Weighted-Metriken berücksichtigen die Klassenhäufigkeiten.
```

---

# Zelle 18: Feature Importance Random Forest

```python
preprocessor_fitted = rf_pipeline.named_steps["preprocessor"]
rf_fitted = rf_pipeline.named_steps["model"]

feature_names = preprocessor_fitted.get_feature_names_out()
feature_importances = rf_fitted.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

display(importance_df.head(20))

plt.figure(figsize=(10, 8))
sns.barplot(
    data=importance_df.head(20),
    x="Importance",
    y="Feature"
)
plt.title("Top 20 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
```

Interpretation:

```text
Die Feature Importance zeigt, welche Variablen für die Modellentscheidung besonders relevant waren.
Bei diesem Datensatz sind häufig akademische Leistungsvariablen aus dem ersten und zweiten Semester besonders wichtig.
Das ist plausibel, weil Studienerfolg stark mit bisherigen Studienleistungen zusammenhängt.
```

---

# Zelle 19: Preprocessing für Deep Learning

```python
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()

X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
    X_train_processed,
    y_train,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("Train DL:", X_train_dl.shape)
print("Validation DL:", X_val_dl.shape)
print("Test DL:", X_test_processed.shape)
```

Wichtig:

```text
Das Deep-Learning-Modell bekommt exakt dieselben Trainings- und Testdaten wie der Random Forest.
Zusätzlich wird innerhalb der Trainingsdaten ein Validierungsset erzeugt, damit Early Stopping möglich ist.
```

---

# Zelle 20: Class Weights für Deep Learning

```python
classes = np.unique(y_train_dl)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_dl
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, class_weights_array)
}

print("Class Weights:")
display(class_weights)
```

Erklärung:

```text
Class Weights geben selteneren Klassen ein höheres Gewicht im Training.
Das ist sinnvoll, wenn die Zielklassen nicht gleich häufig vorkommen.
```

---

# Zelle 21: Eigenes Deep-Learning-Modell bauen

```python
input_dim = X_train_dl.shape[1]
num_classes = len(label_encoder.classes_)

dl_model = keras.Sequential([
    layers.Input(shape=(input_dim,)),

    layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.0001)
    ),
    layers.BatchNormalization(),
    layers.Dropout(0.30),

    layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.0001)
    ),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    layers.Dense(
        32,
        activation="relu"
    ),
    layers.Dropout(0.15),

    layers.Dense(
        num_classes,
        activation="softmax"
    )
])

dl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

dl_model.summary()
```

Erklärung:

```text
Das Modell ist ein vollständig selbst gebautes neuronales Netz für tabellarische Daten.
Dense Layers lernen nichtlineare Zusammenhänge.
ReLU wird verwendet, weil sie effizient ist und Vanishing Gradients reduziert.
Dropout und L2-Regularisierung reduzieren Overfitting.
Softmax erzeugt Wahrscheinlichkeiten für die drei Zielklassen.
Sparse Categorical Crossentropy ist passend, weil die Klassen als Integer codiert sind.
```

---

# Zelle 22: Deep Learning Training

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-5
)

start_time = time.time()

history = dl_model.fit(
    X_train_dl,
    y_train_dl,
    validation_data=(X_val_dl, y_val_dl),
    epochs=150,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

dl_training_time = time.time() - start_time

print("Deep Learning Training abgeschlossen.")
print("Trainingszeit:", round(dl_training_time, 2), "Sekunden")
```

Während dieser Zelle siehst du pro Epoche:

* `loss`
* `accuracy`
* `val_loss`
* `val_accuracy`

Wenn Early Stopping greift, stoppt das Training automatisch vor 150 Epochen.

---

# Zelle 23: Trainingskurven visualisieren

```python
history_df = pd.DataFrame(history.history)

plt.figure(figsize=(8, 5))
plt.plot(history_df["loss"], label="Training Loss")
plt.plot(history_df["val_loss"], label="Validation Loss")
plt.title("Loss-Kurve des Deep-Learning-Modells")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history_df["accuracy"], label="Training Accuracy")
plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy-Kurve des Deep-Learning-Modells")
plt.xlabel("Epoche")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```

Interpretation:

```text
Wenn Training Loss stark sinkt, aber Validation Loss steigt, deutet das auf Overfitting hin.
Wenn beide Kurven ähnlich verlaufen, generalisiert das Modell besser.
Early Stopping verhindert, dass zu lange trainiert wird.
```

---

# Zelle 24: Deep Learning Evaluation auf Testset

```python
dl_probabilities = dl_model.predict(X_test_processed)
dl_predictions = np.argmax(dl_probabilities, axis=1)

dl_results = evaluate_model(
    model_name="Deep Learning MLP",
    y_true=y_test,
    y_pred=dl_predictions
)

display(pd.DataFrame([dl_results]))

print("Classification Report - Deep Learning MLP:")
print(
    classification_report(
        y_test,
        dl_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

plot_confusion_matrix(
    y_true=y_test,
    y_pred=dl_predictions,
    title="Confusion Matrix - Deep Learning MLP"
)
```

Wichtig:

```text
Das Deep-Learning-Modell wird auf demselben Testset evaluiert wie der Random Forest.
Dadurch ist der Vergleich fair.
```

---

# Zelle 25: Modellvergleich

```python
comparison_df = pd.DataFrame([
    rf_results,
    dl_results
])

comparison_df["Trainingszeit_Sekunden"] = [
    rf_training_time,
    dl_training_time
]

display(comparison_df)

metrics_to_plot = [
    "Accuracy",
    "Precision_macro",
    "Recall_macro",
    "F1_macro",
    "F1_weighted"
]

comparison_plot_df = comparison_df.set_index("Modell")[metrics_to_plot]

comparison_plot_df.T.plot(kind="bar", figsize=(12, 6))
plt.title("Vergleich Random Forest vs. Deep Learning MLP")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(title="Modell")
plt.show()
```

Interpretation:

```text
Das bessere Modell ist nicht nur anhand der Accuracy zu beurteilen.
Bei unbalancierten Klassen sind besonders Macro-F1 und klassenspezifischer Recall wichtig.
Random Forest ist häufig stärker bei tabellarischen Daten und besser interpretierbar.
Das neuronale Netz kann nichtlineare Muster lernen, benötigt aber mehr Tuning und ist weniger transparent.
```

---

# Zelle 26: Automatische Entscheidungshilfe

```python
best_model_by_f1 = comparison_df.sort_values(
    by="F1_macro",
    ascending=False
).iloc[0]

print("Bestes Modell nach Macro-F1:")
print(best_model_by_f1["Modell"])
print("Macro-F1:", round(best_model_by_f1["F1_macro"], 4))
```

Diesen Satz kannst du in deiner Prüfung verwenden:

```text
Ich verwende Macro-F1 als wichtige Vergleichsmetrik, weil alle Klassen gleich stark berücksichtigt werden.
Das ist bei einem unbalancierten Mehrklassenproblem aussagekräftiger als reine Accuracy.
```

---

# Zelle 27: Wissenschaftliche Diskussion als Markdown in dein Notebook kopieren

```markdown
## Wissenschaftliche Diskussion

Die Ergebnisse zeigen, dass das Problem der Vorhersage von Studienabbruch und Studienerfolg als multiklassiges Klassifikationsproblem sinnvoll modelliert werden kann.

Eine besondere Stärke des Projekts ist, dass klassische Machine-Learning-Methoden und ein eigenes Deep-Learning-Modell auf demselben Testset verglichen wurden. Dadurch ist der Vergleich methodisch fair.

Der Random Forest eignet sich besonders gut für tabellarische Daten. Er kann nichtlineare Zusammenhänge erkennen, ist robust und liefert Feature Importances. Dadurch ist er besser interpretierbar als das neuronale Netz.

Das Deep-Learning-Modell wurde vollständig selbst erstellt. Es verwendet Dense Layers, ReLU-Aktivierungen, Dropout, L2-Regularisierung, Batch Normalization und eine Softmax-Ausgabe. Dadurch ist es für ein multiklassiges Klassifikationsproblem geeignet.

Eine Limitation ist, dass der Datensatz zwar viele akademische und demografische Variablen enthält, aber nicht alle möglichen Ursachen für Studienabbruch abbildet. Psychologische, soziale, finanzielle oder institutionelle Faktoren können ebenfalls relevant sein.

Außerdem muss beachtet werden, dass eine hohe Vorhersageleistung nicht automatisch Kausalität bedeutet. Wenn ein Feature wichtig ist, heißt das nicht zwingend, dass es die Ursache für Dropout ist.

In einem echten Unternehmen oder einer Hochschule müsste ein solches Modell sehr vorsichtig eingesetzt werden. Es sollte nicht zur automatischen Benachteiligung von Studierenden führen, sondern als Frühwarnsystem dienen, um Unterstützungsangebote gezielter bereitzustellen.
```

---

# Zelle 28: Finale Conclusion als Markdown

```markdown
## Fazit

In diesem Projekt wurde der Datensatz „Predict Students' Dropout and Academic Success“ analysiert, vorverarbeitet und mit zwei Modelltypen untersucht.

Es wurden ein klassisches Machine-Learning-Modell auf Basis eines Random Forests und ein selbst entwickeltes Deep-Learning-Modell trainiert. Beide Modelle wurden auf demselben stratified Testset evaluiert.

Die Ergebnisse zeigen, dass tabellarische Bildungsdaten gut mit klassischen Machine-Learning-Verfahren modelliert werden können. Besonders der Random Forest ist wegen seiner Robustheit, guten Performance und besseren Interpretierbarkeit eine sinnvolle Wahl.

Das neuronale Netz zeigt, dass auch Deep Learning für strukturierte Daten einsetzbar ist. Allerdings ist es komplexer, weniger interpretierbar und benötigt mehr Hyperparameterentscheidungen.

Für eine praktische Anwendung würde ich das Modell mit der besseren Macro-F1-Performance empfehlen. Zusätzlich sollte die Interpretierbarkeit berücksichtigt werden, da Entscheidungen im Bildungsbereich sensibel sind.
```

---

# Zelle 29: Präsentationsstruktur für 15 Minuten

```markdown
## 15-Minuten-Präsentation

### Folie 1: Titel
**Thema:** Predict Students' Dropout and Academic Success  
**Sprechtext:**  
„In meiner Portfolio-Arbeit untersuche ich, ob sich Studienabbruch und Studienerfolg anhand akademischer, demografischer und administrativer Merkmale vorhersagen lassen.“

---

### Folie 2: Problemdefinition
**Inhalt:**
- Multiklassige Klassifikation
- Zielklassen: Dropout, Enrolled, Graduate
- Relevanz für Hochschulen

**Sprechtext:**  
„Das Ziel ist nicht nur eine technische Klassifikation, sondern auch die Frage, wie Hochschulen gefährdete Studierende frühzeitig erkennen könnten.“

---

### Folie 3: Datensatz
**Inhalt:**
- UCI Machine Learning Repository
- 4424 Beobachtungen
- 36 Features
- Zielvariable: Target

**Sprechtext:**  
„Der Datensatz enthält verschiedene Merkmale wie Studienleistungen, Zulassungsdaten und soziodemografische Informationen.“

---

### Folie 4: Explorative Datenanalyse
**Inhalt:**
- Klassenverteilung
- Missing Values
- Feature-Verteilungen
- Korrelationen

**Sprechtext:**  
„Die EDA zeigt, dass keine fehlenden Werte vorhanden sind, aber die Klassen nicht vollständig gleichverteilt sind.“

---

### Folie 5: Preprocessing
**Inhalt:**
- Label Encoding der Zielvariable
- One-Hot-Encoding kategorialer Features
- StandardScaler für numerische Features
- Stratified Train/Test Split

**Sprechtext:**  
„Besonders wichtig war, kategorial codierte Zahlen nicht fälschlich als metrische Werte zu behandeln.“

---

### Folie 6: Random Forest
**Inhalt:**
- Klassisches ML-Modell
- 400 Entscheidungsbäume
- Class Weight balanced
- Feature Importance

**Sprechtext:**  
„Random Forest ist für tabellarische Daten gut geeignet, weil er nichtlineare Zusammenhänge modellieren kann und relativ robust ist.“

---

### Folie 7: Deep Learning Modell
**Inhalt:**
- Eigenes MLP
- Dense Layers
- ReLU
- Dropout
- L2-Regularisierung
- Softmax Output

**Sprechtext:**  
„Das neuronale Netz wurde vollständig selbst gebaut und verwendet keine pretrained Komponenten.“

---

### Folie 8: Evaluation
**Inhalt:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Sprechtext:**  
„Ich betrachte nicht nur Accuracy, sondern auch Macro-F1, weil die Klassen unterschiedlich häufig vorkommen.“

---

### Folie 9: Modellvergleich
**Inhalt:**
- Vergleichstabelle
- Interpretierbarkeit
- Trainingszeit
- Overfitting-Risiko

**Sprechtext:**  
„Das bessere Modell ist nicht automatisch das komplexere Modell. Gerade bei tabellarischen Daten sind klassische ML-Verfahren oft sehr stark.“

---

### Folie 10: Diskussion und Fazit
**Inhalt:**
- Stärken
- Limitationen
- Bias
- Einsatz in der Praxis

**Sprechtext:**  
„Ein solches Modell sollte in der Praxis nicht als automatisches Entscheidungssystem genutzt werden, sondern als unterstützendes Frühwarnsystem.“
```

---

# Zelle 30: Prüfungsfragen mit Musterantworten

```markdown
## Prüfungsfragen mit Musterantworten

### 1. Warum ist das ein Klassifikationsproblem?
Weil die Zielvariable diskrete Klassen besitzt: Dropout, Enrolled und Graduate.

### 2. Warum ist es eine multiklassige Klassifikation?
Weil es mehr als zwei mögliche Zielklassen gibt.

### 3. Warum wurde ein stratified Split verwendet?
Damit die Klassenverteilung in Trainings- und Testdaten ähnlich bleibt.

### 4. Was ist Data Leakage?
Data Leakage entsteht, wenn Informationen aus dem Testset während des Trainings verwendet werden.

### 5. Warum wird One-Hot-Encoding verwendet?
Weil kategoriale Features keine echte numerische Reihenfolge besitzen.

### 6. Warum werden numerische Features skaliert?
Skalierung bringt Features auf vergleichbare Größenordnungen, besonders wichtig für neuronale Netze.

### 7. Warum braucht Random Forest normalerweise keine Skalierung?
Entscheidungsbäume basieren auf Schwellenwerten und sind nicht distanzbasiert.

### 8. Warum wird trotzdem eine gemeinsame Pipeline verwendet?
Damit das Preprocessing sauber, reproduzierbar und frei von Data Leakage ist.

### 9. Was macht ein Random Forest?
Er kombiniert viele Entscheidungsbäume und aggregiert deren Vorhersagen.

### 10. Was ist der Vorteil von Random Forest?
Er ist robust, kann nichtlineare Zusammenhänge lernen und liefert Feature Importances.

### 11. Was ist ein Nachteil von Random Forest?
Er ist weniger interpretierbar als einfache Modelle wie logistische Regression.

### 12. Was bedeutet Accuracy?
Der Anteil aller korrekt klassifizierten Beispiele.

### 13. Was bedeutet Precision?
Von allen als Klasse X vorhergesagten Fällen: Wie viele waren tatsächlich Klasse X?

### 14. Was bedeutet Recall?
Von allen tatsächlichen Fällen der Klasse X: Wie viele wurden korrekt erkannt?

### 15. Was bedeutet F1-Score?
Der harmonische Mittelwert aus Precision und Recall.

### 16. Warum ist Macro-F1 wichtig?
Weil alle Klassen gleich gewichtet werden, unabhängig von ihrer Häufigkeit.

### 17. Warum kann Accuracy irreführend sein?
Bei unbalancierten Klassen kann ein Modell hohe Accuracy erreichen, obwohl es Minderheitsklassen schlecht erkennt.

### 18. Was zeigt eine Confusion Matrix?
Sie zeigt, welche Klassen korrekt und welche falsch klassifiziert wurden.

### 19. Warum wird ein neuronales Netz verwendet?
Um zu prüfen, ob ein nichtlineares Deep-Learning-Modell bessere Muster lernen kann.

### 20. Was ist ein Dense Layer?
Eine vollständig verbundene Schicht, in der jedes Neuron mit allen Eingaben verbunden ist.

### 21. Warum wird ReLU verwendet?
ReLU ist effizient und reduziert das Vanishing-Gradient-Problem.

### 22. Was macht Dropout?
Dropout deaktiviert während des Trainings zufällig Neuronen, um Overfitting zu reduzieren.

### 23. Was ist L2-Regularisierung?
Eine Strafkomponente für große Gewichte, die Overfitting reduziert.

### 24. Warum wird Softmax im Output Layer verwendet?
Softmax erzeugt Wahrscheinlichkeiten über mehrere Klassen.

### 25. Warum wird Sparse Categorical Crossentropy verwendet?
Weil die Zielklassen als Integer codiert sind.

### 26. Was macht Adam?
Adam ist ein Optimizer, der adaptive Lernraten verwendet.

### 27. Was ist Early Stopping?
Das Training wird beendet, wenn sich die Validierungsleistung nicht mehr verbessert.

### 28. Warum braucht man ein Validierungsset?
Zur Überwachung des Trainings und zur Erkennung von Overfitting.

### 29. Was ist Overfitting?
Das Modell lernt Trainingsdaten zu stark auswendig und generalisiert schlecht.

### 30. Was ist Underfitting?
Das Modell ist zu einfach und lernt die zugrunde liegenden Muster nicht ausreichend.

### 31. Warum sind Class Weights sinnvoll?
Sie gleichen unbalancierte Klassen aus, indem seltenere Klassen stärker gewichtet werden.

### 32. Was bedeutet Generalisierung?
Die Fähigkeit eines Modells, auf neuen unbekannten Daten gut zu funktionieren.

### 33. Warum ist Interpretierbarkeit wichtig?
Weil Entscheidungen im Bildungsbereich sensibel sind und nachvollziehbar sein sollten.

### 34. Was sind Feature Importances?
Sie zeigen, welche Features für die Entscheidungen eines Random Forest besonders relevant sind.

### 35. Bedeutet Feature Importance Kausalität?
Nein. Sie zeigt nur statistische Relevanz im Modell, keine Ursache-Wirkungs-Beziehung.

### 36. Warum sollte man Testdaten nur einmal final verwenden?
Damit die Testleistung eine realistische Schätzung für neue Daten bleibt.

### 37. Was ist Batch Normalization?
Eine Technik, die Aktivierungen innerhalb des Netzes stabilisiert und Training erleichtert.

### 38. Warum ist Deep Learning bei tabellarischen Daten nicht immer besser?
Klassische ML-Modelle sind bei strukturierten Daten oft effizienter und robuster.

### 39. Was wäre eine Verbesserung des Projekts?
Hyperparameter-Tuning, Cross-Validation, zusätzliche Features und Fairness-Analyse.

### 40. Wie würde man das Modell praktisch einsetzen?
Als Frühwarnsystem, nicht als automatisches Entscheidungssystem.
```

---

# Reihenfolge beim Ausführen

Du führst im Notebook genau diese Reihenfolge aus:

```text
Zelle 1  → Installation
Zelle 2  → Imports
Zelle 3  → Daten laden
Zelle 4  → Übersicht
Zelle 5  → Missing Values
Zelle 6  → Zielvariable
Zelle 7  → Datentypen
Zelle 8  → Spaltentypen definieren
Zelle 9  → Histogramme
Zelle 10 → Boxplots
Zelle 11 → Korrelationsmatrix
Zelle 12 → Zielkorrelationen
Zelle 13 → Train/Test Split
Zelle 14 → Preprocessing
Zelle 15 → Evaluation-Funktionen
Zelle 16 → Random Forest Training
Zelle 17 → Random Forest Evaluation
Zelle 18 → Feature Importance
Zelle 19 → Daten für Deep Learning
Zelle 20 → Class Weights
Zelle 21 → Deep Learning Architektur
Zelle 22 → Deep Learning Training
Zelle 23 → Trainingskurven
Zelle 24 → Deep Learning Evaluation
Zelle 25 → Modellvergleich
Zelle 26 → Bestes Modell bestimmen
Zelle 27 → Diskussion
Zelle 28 → Fazit
Zelle 29 → Präsentation
Zelle 30 → Prüfungsfragen
```

---

# Wichtig für deine Abgabe

Bei der Abgabe solltest du im Notebook **nicht nur Code**, sondern auch Markdown-Erklärungen haben.

Die wichtigsten Markdown-Sätze sind:

```text
Dieses Projekt behandelt ein multiklassiges Klassifikationsproblem, da die Zielvariable aus den Klassen Dropout, Enrolled und Graduate besteht.

Das Preprocessing wurde ausschließlich auf den Trainingsdaten gelernt, um Data Leakage zu vermeiden.

Random Forest wurde gewählt, weil das Modell für tabellarische Daten robust ist, nichtlineare Zusammenhänge erkennen kann und Feature Importances liefert.

Das Deep-Learning-Modell wurde vollständig selbst erstellt und verwendet keine pretrained Komponenten.

Beide Modelle wurden auf demselben Testset evaluiert, damit der Vergleich fair ist.

Macro-F1 ist besonders wichtig, weil die Zielklassen nicht perfekt gleichverteilt sind.
```
Super, freut mich sehr. Morgen machst du einfach Folgendes:

## Morgen wieder öffnen

### 1. PowerShell öffnen

Öffne wieder **PowerShell**.

### 2. In deinen Projektordner wechseln

```powershell
cd C:\Users\patri\student_dropout_project
```

### 3. Virtuelle Umgebung aktivieren

```powershell
.venv\Scripts\activate
```

Danach sollte links wieder ungefähr stehen:

```text
(.venv)
```

### 4. Jupyter starten

```powershell
jupyter notebook
```

oder:

```powershell
jupyter lab
```

### 5. Link im Browser öffnen

Jupyter zeigt dir wieder einen Link wie:

```text
http://localhost:8889/lab?token=...
```

oder:

```text
http://localhost:8888/lab?token=...
```

Diesen Link kopierst du in den Browser.

Dann öffnest du deine Datei:

```text
predict_students.ipynb
```

---

## Wichtig zu `localhost`

Dein alter Link:

```text
http://localhost:8889/lab/tree/predict_students.ipynb
```

funktioniert morgen **nur**, wenn Jupyter wieder auf demselben Port läuft und der Server aktiv ist.

Wenn Jupyter geschlossen ist, ist `localhost` nicht erreichbar.

Deshalb immer zuerst:

```powershell
cd C:\Users\patri\student_dropout_project
.venv\Scripts\activate
jupyter lab
```

Dann den neuen Link öffnen.

---

## Projekt richtig schließen

Wenn du fertig bist:

1. Notebook speichern: **Strg + S**
2. Browser-Tab schließen
3. In PowerShell bei laufendem Jupyter:

```text
Ctrl + C
```

Dann fragt Jupyter eventuell:

```text
Shutdown this Jupyter server?
```

Dann eingeben:

```text
y
```

und Enter drücken.

Danach ist alles sauber geschlossen.

