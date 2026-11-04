import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

np.random.seed(42)
n = 300

data = pd.DataFrame({
    "Hour": np.random.randint(6, 23, n),
    "Weekend": np.random.randint(0, 2, n),
    "Event": np.random.randint(0, 2, n)
})

demand = (
    20
    + 25 * data["Hour"].between(9, 11)
    + 25 * data["Hour"].between(13, 16)
    + 15 * (data["Weekend"] == 0)
    + 25 * data["Event"]
    + np.random.normal(0, 8, n)
)

data["Availability"] = np.where(
    demand < 45,
    "High",
    np.where(demand < 70, "Medium", "Low")
)

print("========== ORIGINAL DATASET ==========")
print(data.head(10))
print("\nDataset Shape:", data.shape)

print("\n========== DATA PREPROCESSING ==========")

print("\nDataset Information:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())

for col in ["Hour", "Weekend", "Event"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].fillna(data[col].median())

data["Availability"] = (
    data["Availability"]
    .astype(str)
    .str.strip()
    .str.capitalize()
)

data = data[
    data["Availability"].isin(["High", "Medium", "Low"])
]

data = data[data["Hour"].between(6, 22)]

data = data[
    data["Weekend"].isin([0, 1]) &
    data["Event"].isin([0, 1])
]

print("\nDuplicate Records:", data.duplicated().sum())

data = data.drop_duplicates().reset_index(drop=True)

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

print("\nCleaned Dataset Shape:", data.shape)

print("\nCleaned Dataset:")
print(data.head())

print("\n========== FEATURE ENGINEERING ==========")

data["Peak_Hour"] = (
    data["Hour"].between(9, 11) |
    data["Hour"].between(13, 16)
).astype(int)

print(data.head())

data.to_csv(
    "Smart_Parking_Cleaned_Dataset.csv",
    index=False
)

print("\nCleaned dataset exported successfully!")

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

print("\nStatistical Summary:")
print(data.describe())

print("\nAvailability Distribution:")
print(data["Availability"].value_counts())

print("\nAvailability Percentage:")
print(
    data["Availability"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n========== OBJECTIVE 1 ==========")
print("Analysis of Overall Parking Availability")

availability_analysis = data["Availability"].value_counts()

print(availability_analysis)

print("\n========== OBJECTIVE 2 ==========")
print("Identification of Hours with Low Parking Availability")

low_by_hour = (
    data.assign(
        Low=(data["Availability"] == "Low")
    )
    .groupby("Hour")["Low"]
    .mean() * 100
)

print(low_by_hour.round(2))

print(
    "\nHour with Highest Low Availability:",
    low_by_hour.idxmax()
)

print("\n========== OBJECTIVE 3 ==========")
print("Weekday vs Weekend Parking Analysis")

weekend_analysis = pd.crosstab(
    data["Weekend"],
    data["Availability"],
    normalize="index"
) * 100

print(weekend_analysis.round(2))

print("\n========== OBJECTIVE 4 ==========")
print("Impact of Special Events on Parking")

event_analysis = pd.crosstab(
    data["Event"],
    data["Availability"],
    normalize="index"
) * 100

print(event_analysis.round(2))

print("\n========== OBJECTIVE 5 ==========")
print("Peak Hour Parking Analysis")

peak_analysis = pd.crosstab(
    data["Peak_Hour"],
    data["Availability"],
    normalize="index"
) * 100

print(peak_analysis.round(2))

counts = data["Availability"].value_counts().reindex(
    ["High", "Medium", "Low"],
    fill_value=0
)

fig, axes = plt.subplots(2, 3, figsize=(17, 10))

counts.plot(kind="bar", ax=axes[0, 0])

axes[0, 0].set_title("Parking Availability Distribution")
axes[0, 0].set_xlabel("Availability")
axes[0, 0].set_ylabel("Number of Records")
axes[0, 0].tick_params(axis="x", rotation=0)

axes[0, 1].plot(
    low_by_hour.index,
    low_by_hour.values,
    marker="o"
)

axes[0, 1].set_title("Low Availability by Hour")
axes[0, 1].set_xlabel("Hour")
axes[0, 1].set_ylabel("Low Availability (%)")
axes[0, 1].grid(True)

counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=axes[0, 2]
)

axes[0, 2].set_title("Parking Availability Percentage")
axes[0, 2].set_ylabel("")

weekend_analysis.plot(
    kind="bar",
    ax=axes[1, 0]
)

axes[1, 0].set_title("Weekday vs Weekend")
axes[1, 0].set_xlabel("Weekend (0=No, 1=Yes)")
axes[1, 0].set_ylabel("Percentage")
axes[1, 0].tick_params(axis="x", rotation=0)

event_analysis.plot(
    kind="bar",
    ax=axes[1, 1]
)

axes[1, 1].set_title("Special Event Impact")
axes[1, 1].set_xlabel("Event (0=No, 1=Yes)")
axes[1, 1].set_ylabel("Percentage")
axes[1, 1].tick_params(axis="x", rotation=0)

peak_analysis.plot(
    kind="bar",
    ax=axes[1, 2]
)

axes[1, 2].set_title("Peak Hour Parking Availability")
axes[1, 2].set_xlabel("Peak Hour (0=No, 1=Yes)")
axes[1, 2].set_ylabel("Percentage")
axes[1, 2].tick_params(axis="x", rotation=0)

plt.tight_layout()

plt.savefig(
    "Smart_Parking_EDA.png",
    dpi=300
)

plt.show()

print("\n========== MACHINE LEARNING MODELS ==========")

X = data[
    ["Hour", "Weekend", "Event", "Peak_Hour"]
]

y = data["Availability"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ])
}

results = []
trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    trained_models[name] = model

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1

    })

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "Smart_Parking_Model_Results.csv",
    index=False
)

print("\n========== CROSS VALIDATION ==========")

cv_results = []

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_results.append({

        "Model": name,
        "Average CV Accuracy": scores.mean(),
        "CV Standard Deviation": scores.std()

    })

cv_df = pd.DataFrame(cv_results)

print(cv_df.round(4).to_string(index=False))

cv_df.to_csv(
    "Smart_Parking_Cross_Validation.csv",
    index=False
)

best_model_name = (
    results_df.sort_values(
        by=["F1 Score", "Accuracy"],
        ascending=False
    ).iloc[0]["Model"]
)

best_model = trained_models[best_model_name]

best_predictions = best_model.predict(X_test)

print("\n========== BEST FIT MODEL ==========")

print("Best Model:", best_model_name)

print(
    "Best Model Accuracy:",
    round(
        accuracy_score(y_test, best_predictions) * 100,
        2
    ),
    "%"
)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["High", "Medium", "Low"]
)

plt.figure(figsize=(7, 5))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.colorbar()

plt.xticks(
    [0, 1, 2],
    ["High", "Medium", "Low"]
)

plt.yticks(
    [0, 1, 2],
    ["High", "Medium", "Low"]
)

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted Availability")
plt.ylabel("Actual Availability")

plt.tight_layout()

plt.savefig(
    "Smart_Parking_Confusion_Matrix.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"] * 100
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "Model_Accuracy_Comparison.png",
    dpi=300
)

plt.show()

if best_model_name in ["Decision Tree", "Random Forest"]:

    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\n========== FEATURE IMPORTANCE ==========")

    print(importance)

    plt.figure(figsize=(8, 5))

    importance.plot(kind="bar")

    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.tight_layout()

    plt.savefig(
        "Feature_Importance.png",
        dpi=300
    )

    plt.show()

print("\n========== PARKING PREDICTION ==========")

try:

    hour = int(
        input("Enter hour (6 to 22): ")
    )

    weekend = input(
        "Is it a weekend? (yes/no): "
    ).strip().lower()

    event = input(
        "Is there a special event? (yes/no): "
    ).strip().lower()

    if not 6 <= hour <= 22:

        print("Enter an hour from 6 to 22.")

    elif weekend not in ["yes", "no"]:

        print("Please answer yes or no for weekend.")

    elif event not in ["yes", "no"]:

        print("Please answer yes or no for event.")

    else:

        peak_hour = int(
            (9 <= hour <= 11) or
            (13 <= hour <= 16)
        )

        new_data = pd.DataFrame({

            "Hour": [hour],

            "Weekend": [
                1 if weekend == "yes" else 0
            ],

            "Event": [
                1 if event == "yes" else 0
            ],

            "Peak_Hour": [peak_hour]

        })

        result = best_model.predict(new_data)[0]

        print("\nPredicted Parking Availability:", result)

        if result == "High":

            print(
                "Recommendation: Parking is likely to be easily available."
            )

        elif result == "Medium":

            print(
                "Recommendation: Parking availability may be limited."
            )

        else:

            print(
                "Recommendation: Parking demand is likely to be high."
            )

except ValueError:

    print("Please enter a valid hour.")

print("\n================================================")
print("          FINAL PROJECT CONCLUSION")
print("================================================")

print("\nProject: Smart Parking Availability Predictor")

print("Total Dataset Records:", len(data))

print("Total Features:", len(X.columns))

print("Machine Learning Models:", len(models))

print("Best Performing Model:", best_model_name)

print(
    "Best Model Accuracy:",
    round(
        accuracy_score(y_test, best_predictions) * 100,
        2
    ),
    "%"
)

print(
    "Best Model Precision:",
    round(
        precision_score(
            y_test,
            best_predictions,
            average="macro",
            zero_division=0
        ) * 100,
        2
    ),
    "%"
)

print(
    "Best Model Recall:",
    round(
        recall_score(
            y_test,
            best_predictions,
            average="macro",
            zero_division=0
        ) * 100,
        2
    ),
    "%"
)

print(
    "Best Model F1 Score:",
    round(
        f1_score(
            y_test,
            best_predictions,
            average="macro",
            zero_division=0
        ) * 100,
        2
    ),
    "%"
)

print("\nKEY FINDINGS:")

print("1. Parking availability is influenced by time of day.")
print("2. Weekday and weekend patterns can be compared.")
print("3. Special events can influence parking demand.")
print("4. Peak hours can be identified through data analysis.")
print("5. Machine learning can classify parking availability.")

print("\nRECOMMENDATIONS:")

print("1. Monitor parking availability during peak hours.")
print("2. Plan parking resources before special events.")
print("3. Use predictive models to support parking management.")
print("4. Improve parking information systems.")

print("\nGenerated Project Files:")

print("1. Smart_Parking_Cleaned_Dataset.csv")
print("2. Smart_Parking_Model_Results.csv")
print("3. Smart_Parking_Cross_Validation.csv")
print("4. Smart_Parking_EDA.png")
print("5. Smart_Parking_Confusion_Matrix.png")
print("6. Model_Accuracy_Comparison.png")
print("7. Feature_Importance.png (if applicable)")

print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY!")

print("================================================")