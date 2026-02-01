import pandas as pd

FEATURE_COLUMNS = [
    'RhythmScore',
    'AudioLoudness',
    'VocalContent',
    'AcousticQuality',
    'InstrumentalScore',
    'LivePerformanceLikelihood',
    'MoodScore',
    'TrackDurationMs',
    'Energy'
]

TARGET_COLUMN = "BeatsPerMinute"

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and returns feature matrix X. Adds Bias = 1
    """
    x = df[FEATURE_COLUMNS].copy()
    x["Bias"] = 1
    return x

def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) from a training dataframe.
    """
    x = prepare_features(df)
    y = df[TARGET_COLUMN]
    return x, y