import pandas as pd
import joblib
from econml.cate_interpreter import SingleTreeCateInterpreter

from data_extraction.utils import load_excel


def prepare_data_with_scaler(filepath, feature_names, scaler):
    try:
        df = load_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = load_excel(filepath)
    df = df.fillna(df.median(numeric_only=True))
    X = df[feature_names]
    X_normalized = pd.DataFrame(scaler.transform(X), columns=feature_names)
    return df, X_normalized

def interpret_causal_forest(data_path="../data/fm_dem_sat_merged.xlsx", model_path="../models/causal_forest_full.pkl"):
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["features"]

    df, X_normalized = prepare_data_with_scaler(data_path, feature_names, scaler)

    interpreter = SingleTreeCateInterpreter(max_depth=3)
    interpreter.interpret(model, X_normalized)

    dot_str = interpreter.export_graphviz(
        feature_names=feature_names,
        treatment_names=[model_bundle.get("treatment_col", "treatment")],
        rounded=True,
        precision=3,
    )
    print(dot_str)
    print("\n---")
    print("Vizualize the string here:")
    print("https://dreampuf.github.io/GraphvizOnline/")

if __name__ == "__main__":
    interpret_causal_forest()
