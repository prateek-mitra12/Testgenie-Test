import pandas as pd
from transformers import AutoTokenizer


def select_excel_file(file_path):
    """Load the Excel file and return a DataFrame."""
    return pd.ExcelFile(file_path)


def count_tokens_in_excel(file_path):
    """Count total tokens in all sheets of the specified Excel file."""
    # Load the tokenizer for your specific model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Change this to your specific model

    # Load the Excel file
    xls = select_excel_file(file_path)

    # To hold the total token count
    total_tokens = 0

    # Iterate through each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)  # Load the sheet into a DataFrame
        # Convert the DataFrame to a string
        text_data = df.to_string(index=False)  # or df.to_csv(index=False) for CSV representation

        # Count the tokens in the text data
        tokens = tokenizer.encode(text_data)  # Get the tokens
        token_count = len(tokens)  # Count the number of tokens
        total_tokens += token_count
        
        print(f"Tokens in sheet '{sheet_name}': {token_count}")

    print(f"Total tokens in the file: {total_tokens}")


# Example usage
file_path = "FSS- Regression_Test Case_V1.2.xlsx"  # Adjust this path if necessary
count_tokens_in_excel(file_path)


