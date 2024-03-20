import pandas as pd
from joblib import load

if __name__ == "__main__":
    test_df = pd.read_csv("datasets/test.csv")
    features_df = pd.read_csv("datasets/features.csv")
    test_merged = pd.merge(test_df, features_df, on=['lat', 'lon'], how='left')
    X_test = test_merged.drop(columns=['id'])
    selector = load(open('selector.sav', 'rb'))
    X_test_filtered = selector.transform(X_test)
    loaded_model = load(open('best_model.sav', "rb"))
    test_predictions = loaded_model.predict(X_test_filtered)
    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "score": test_predictions
    })
    submission_df.to_csv("submission.csv", index=False)
