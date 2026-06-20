"""
data_loader.py
---------------
Loads CSV/Excel files into a pandas DataFrame and performs basic
sanity checks before handing off to the agent pipeline.
"""

from pathlib import Path
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame.

    Args:
        file_path: path to the uploaded file.

    Returns:
        pandas.DataFrame

    Raises:
        ValueError: if the file extension is unsupported.
        FileNotFoundError: if the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if suffix == ".csv":
        # Try a couple of common encodings before giving up.
        for encoding in ("utf-8", "latin1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV file with utf-8 or latin1 encoding.")

    # .xlsx / .xls
    return pd.read_excel(path)


def basic_dataset_info(df: pd.DataFrame) -> dict:
    """
    Quick shape/column summary used by the UI right after upload.
    """
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
