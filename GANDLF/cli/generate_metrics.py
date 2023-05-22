import pandas as pd
from GANDLF.parseConfig import parseConfig

# input: 1 csv with 3 columns: subjectid, prediction, ground_truth
# input: gandlf config
# output: metrics.csv (based on the config)


def _parse_gandlf_csv(fpath):
    df, _ = parseTrainingCSV(fpath, train=False)
    df = df.drop_duplicates()
    for _, row in df.iterrows():
        if "Label" in row:
            yield row["SubjectID"], row["Channel_0"], row["Label"]
        else:
            yield row["SubjectID"], row["Channel_0"]


def generate_metrics_dict(input_csv: str, config: str) -> dict:
    """
    This function generates metrics from the input csv and the config.

    Args:
        input_csv (str): The input CSV.
        config (str): The input yaml config.

    Returns:
        dict: The metrics dictionary.
    """
    input_df = pd.read_csv(input_csv)

    required_columns = ["subjectid", "prediction", "ground_truth"]
    for column in required_columns:
        assert (
            column in input_df.columns
        ), f"The input csv should have a column named {column}"

    config_dict = parseConfig(config)