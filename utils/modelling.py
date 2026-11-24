from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score, recall_score, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


TARGET = "y3"
NUMERIC = [
    "HEIGHT", "SPEED", "DISTANCE", "BodyMass", "log_ImpactEnergy", "AC_MASS", "NUM_ENGS"
]
ORDINAL = [
    "SIZE", "NUM_SEEN", "NUM_STRUCK"
]
NOMINAL = [
    "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT",
    "TIME_OF_DAY", "FAAREGION", "SKY",
    "PRECIPITATION", "WARNED", "SpeciesGroup",
    "INCIDENT_MONTH"
]
STRIKE_PARTS = [
    "STR_RAD", "STR_WINDSHLD", "STR_NOSE",
    "STR_ENG1", "ING_ENG1",
    "STR_ENG2", "ING_ENG2",
    "STR_ENG3", "ING_ENG3",
    "STR_ENG4", "ING_ENG4",
    "STR_PROP", "STR_WING_ROT", "STR_FUSE",
    "STR_LG", "STR_TAIL", "STR_LGHTS", "STR_OTHER",
]


def load_data(file_path, target=TARGET):
    import pandas as pd
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    return df


def stratified_split(df, random_state=67, target=TARGET):
    train_val, test = train_test_split(df, test_size=0.2, stratify=df[target], random_state=random_state)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val[target], random_state=random_state)
    return train, val, test


def split_xy(df, target=TARGET):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def load_train_val_test(file_path, target=TARGET):
    df = load_data(file_path, target=target)
    train, val, test = stratified_split(df, target=target)
    Xtrain, ytrain = split_xy(train, target=target)
    Xval, yval = split_xy(val, target=target)
    Xtest, ytest = split_xy(test, target=target)
    return Xtrain, ytrain, Xval, yval, Xtest, ytest


ORDINAL_ORDER = [
    ["Small", "Medium", "Large", np.nan], # for size
    ["1", "2-10", "11-100", "More than 100", np.nan], # for num_seen
    ["1", "2-10", "11-100", "More than 100", np.nan], # for num_struck
]

def make_preprocessor(model="A"):
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler()
    )
    ordinal_transformer = make_pipeline(
        OrdinalEncoder(categories=ORDINAL_ORDER),
        SimpleImputer(strategy="most_frequent", add_indicator=True),
    )
    nominal_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)
    )

    transformers=[
        ("num", numeric_transformer, NUMERIC),
        ("ord", ordinal_transformer, ORDINAL),
        ("nom", nominal_transformer, NOMINAL)
    ]
    if model == "B":
        transformers.append(
            ("strike_parts", "passthrough", STRIKE_PARTS)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


# baseline -> DummyClassifier

def make_baseline_model():
    return make_pipeline(
        make_preprocessor(model="A"),
        DummyClassifier(strategy="most_frequent")
    )


LABELS = ["None", "Minor", "Subst/Destroyed"]


def evaluate_model(model, X, y, name, labels=LABELS):
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred, average="macro")
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred, average=None, zero_division=0)
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(
        classification_report(y, y_pred, target_names=labels, zero_division=0),
        f"Model: {name}\nF1 Macro: {f1:.4f}\nBalanced Accuracy: {balanced_accuracy:.4f}\nAccuracy: {accuracy:.4f}\nRecall: {recall}\nConfusion matrix (rows = true, columns = predicted):\n{cm}"
    )

    return {
        "model": name,
        "f1_macro": f1,
        "bal_acc": balanced_accuracy,
        "recall": recall,
        "accuracy": accuracy,
        "precision": precision_score(y, y_pred, average=None, zero_division=0),
        "f1": f1_score(y, y_pred, average=None, zero_division=0),
        "confusion_matrix": cm,
    }


def cv_summary(search, name):
    mean = search.cv_results_["mean_test_score"][search.best_index_]
    std = search.cv_results_["std_test_score"][search.best_index_]
    print(f"{name}: CV macro F1 = {mean:.4f} +/- {std:.4f}")
    return {"model": name, "cv_f1_mean": mean, "cv_f1_std": std}


def results_table(results, labels=LABELS):
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("recall", "precision", "f1", "confusion_matrix")}
        for metric in ("precision", "recall", "f1"):
            if metric in r:
                for label, value in zip(labels, r[metric]):
                    row[f"{metric}_{label}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion_matrices(models, X, y, path, labels=LABELS):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (title, model) in zip(axes, models.items()):
        cm = confusion_matrix(y, model.predict(X))
        shares = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{shares[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                        color="white" if shares[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks(range(len(labels)), labels, rotation=20)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def make_class_weights(y):
    counts = y.value_counts()
    return lambda g: {_class: (counts.max() / n) ** g for _class, n in counts.items()}
