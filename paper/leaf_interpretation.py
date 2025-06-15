import pandas as pd
import joblib
from econml.cate_interpreter import SingleTreeCateInterpreter

from data_extraction.utils import load_excel


def prepare_data_with_scaler(
    filepath="../data/fm_dem_sat_merged.xlsx",
    feature_names=None,
    scaler=None
):
    """
    Load data, impute missing, select given features,
    and apply provided scaler to normalize them.
    """
    try:
        df = load_excel(filepath)
    except FileNotFoundError:
        from data_extraction.merge_fee_morbidity_demographics import merge_fm_dm_sat
        merge_fm_dm_sat()
        df = load_excel(filepath)

    # Impute missing values with median
    df = df.fillna(df.median(numeric_only=True))
    # Extract and normalize features
    X = df[feature_names]
    X_normalized = pd.DataFrame(scaler.transform(X), columns=feature_names)
    return df, X_normalized


def interpret_causal_forest(
    data_path="../data/fm_dem_sat_merged.xlsx",
    model_path="../models/causal_forest_full_honest.pkl"
):
    """
    Load trained causal forest bundle, prepare data,
    and interpret with a single-tree CATE interpreter.
    """
    # Load model bundle
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["features"]

    # Prepare data and normalize
    df, X_norm = prepare_data_with_scaler(data_path, feature_names, scaler)

    # Monkey-patch: alias predict to const_marginal_effect
    # so SingleTreeCateInterpreter can call it
    model.const_marginal_effect = model.predict

    # Interpret with a shallow single-tree approximation
    interpreter = SingleTreeCateInterpreter(max_depth=3)
    interpreter.interpret(model, X_norm)

    # Export to Graphviz dot format
    dot_str = interpreter.export_graphviz(
        feature_names=feature_names,
        treatment_names=[bundle.get("treatment_col", "treatment")],
        rounded=True,
        precision=3,
    )

    # Output the dot string and visualization link
    print(dot_str)
    print("\n---")
    print("Visualize via: https://dreampuf.github.io/GraphvizOnline/")


if __name__ == "__main__":
    interpret_causal_forest()
