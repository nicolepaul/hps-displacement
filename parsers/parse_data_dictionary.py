import pandas as pd


# Convert values list into dictionary
def parse_data_values(row):
    """Parses the 'Values' column in the data dictionary, which can be
    used to convert the integer keys to their corresponding str responses

    Args:
        row (pd.Series): Each row of a pd.DataFrame

    Returns:
        r_dict (pd.Series): Each row contains a dict to parse responses
    """

    # Initialize values
    r_dict = dict()

    # Attempt to parse list of values
    try:
        r_list = [r.strip() for r in row.split("\n")]
        r_dict = dict(
            (int(r.split(") ").pop(0)), r.split(") ").pop().strip())
            if ") " in r
            else (str(r.split("=").pop(0)).strip("' "), r.split("=").pop().strip("' "))
            for r in r_list
        )

    # Except errors; not all rows require a conversion dictionary
    except ValueError:
        pass

    # Return result
    return r_dict


def parse_data_dictionary(file_path, drop_bad=True):
    # Read data dictionary
    data_dict = pd.read_excel(file_path)

    # Create conversion dictionaries
    data_dict["Conversion"] = data_dict.Values.apply(parse_data_values)

    # Print that 'bad' values are being dropped
    if drop_bad:
        bad_vals = [-88, -99]
        conv_dicts = data_dict["Conversion"].tolist()
        for conv_dict in conv_dicts:
            for bad_val in bad_vals:
                try:
                    del conv_dict[bad_val]
                except KeyError:
                    pass
        data_dict["Conversion"] = conv_dicts

    # Return result
    return data_dict
