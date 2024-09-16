import os
import zipfile
import pandas as pd


def parse_puf_files(data_folder, drop_bad=True):
    """This function loads all available PUF CSV zip files

    Args:
        data_folder (str): Location where PUF CSV zipfiles are stored

    Returns:
        puf (pd.DataFrame): DataFrame combining all available PUF CSVs
    """

    # Find all available PUF zip files
    file_suffix = "_PUF_CSV.zip"
    puf_zip_files = [
        file for file in os.listdir(data_folder) if file.endswith(file_suffix)
    ]
    puf_zip_paths = [os.path.join(data_folder, file) for file in puf_zip_files]
    n_puf = len(puf_zip_files)

    # Initialize data
    pufs = [pd.DataFrame() for _ in range(n_puf)]
    for i in range(n_puf):
        with zipfile.ZipFile(puf_zip_paths[i], "r") as f:
            puf_name = [
                name
                for name in f.namelist()
                if name.endswith(".csv") and "repwgt" not in name
            ][0]
            pufs[i] = pd.read_csv(f.open(puf_name))

    # Combine data
    puf = pd.concat(pufs, axis=0)

    # Print that 'bad' values are being dropped
    if drop_bad:
        bad_vals = [-88, -99]
        puf.replace(bad_vals, float('nan'), inplace=True)
        print(f"Setting bad_vals ({bad_vals}) = NaN")

    # Return result
    return puf


def custom_puf_handling(puf, data_dict):

    # Identify inital columns
    in_cols = puf.columns.tolist()

    # Get hazard type
    puf, data_dict = get_hazard_type(puf, data_dict)

    # Bin continuous or discrete datasets
    puf, data_dict = convert_birth_year_to_age_bin(puf, data_dict)
    puf, data_dict = convert_hh_size_to_bin(puf, data_dict)
    puf, data_dict = convert_rent_to_bin(puf, data_dict)

    # Normalize household income
    puf, data_dict = normalize_income(puf, data_dict)

    # Rebin columns for tenure and living quarters, etc
    puf, data_dict = rebin_livqtr_column(puf, data_dict)
    puf, data_dict = rebin_tenure_column(puf, data_dict)
    puf, data_dict = rebin_race(puf, data_dict)

    # Get school enrollment variable
    puf, data_dict = rebin_school_enroll(puf, data_dict)

    # Create dummy columns for living quarters
    puf, data_dict = append_livqtr_columns(puf, data_dict)

    # Create columns for retun, protracted displacement
    puf, data_dict = append_returned_column(puf, data_dict)
    puf, data_dict = append_protracted_column(puf, data_dict)
    puf, data_dict = append_recovery_column(puf, data_dict)
    puf, data_dict = append_phase_column(puf, data_dict)
    puf, data_dict = append_phase_return_column(puf, data_dict)
    # Create columns for return time windows
    puf, data_dict = append_return_window_columns(puf, data_dict)

    # Determine new columns
    out_cols = puf.columns.tolist()
    new_cols = [col for col in out_cols if col not in in_cols]
    print(f"Added new columns: {new_cols}")

    return puf, data_dict


def get_hazard_type(df, data_dict):
    # Define HAZARD_TYPE
    # ND_TYPE{i}:
    # 1) Hurricane
    # 2) Flood
    # 3) Fire
    # 4) Tornado
    # 5) Other
    ntype = 5
    ref_cols = [f'ND_TYPE{i+1}' for i in range(ntype)]
    new_col = "HAZARD_TYPE"
    nd_conv = {
        1: 'Hurricane',
        2: 'Flood',
        3: 'Fire',
        4: 'Tornado',
        5: 'Other',
        6: 'Multiple'
    }
    # Gather hazard types
    idx = df.ND_DISPLACE == 1
    df[new_col] = float('nan')
    def get_disaster_type(row):    
        result = row[row == 1].index.tolist()
        if len(result) == 0:
            out = float('nan')
        elif len(result) == 1:
            out = int(result[0][-1])
        else:
            out = 6
        return out
    values = df[idx][ref_cols].apply(get_disaster_type, axis=1)
    df.loc[idx, new_col] = values.values
    # Update data dictionary
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = "Hazard type"
        new_row.at[new_col, 'Conversion'] = nd_conv
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def normalize_income(df, data_dict):
    # Normalize INCOME by THHLD_NUMPER
    # 1) Less than $25,000  
    # 2) \$25,000 - $34,999  
    # 3) \$35,000 - $49,999   
    # 4) \$50,000 - $74,999   
    # 5) \$75,000 - $99,999   
    # 6) \$100,000 - $149,999   
    # 7) \$150,000 - $199,999
    # 8) \$200,000 and above
    new_col = "INCOME_PER"
    numerator, denominator = "INCOME", "THHLD_NUMPER"
    # Establish representative values
    income_mid = {
        1: 25000,
        2: 35000,
        3: 50000,
        4: 75000,
        5: 100000,
        6: 150000,
        7: 200000,
        8: 300000,
    }
    # Select bins for new column
    rebin = [0, 10000, 20000, 30000, 50000, 100000, 150000, 1e16]
    n_bin = len(rebin)-1
    # Calculate values
    df[new_col] = df[numerator].replace(income_mid) / df[denominator]
    # Initialize conversion dictionary
    rebin_conv = dict.fromkeys(range(1, n_bin+1))
    for i in range(n_bin):
        # Get lower and upper bound
        lower, upper = rebin[i], rebin[i+1]-1
        # Arrange conversion dictionary
        if i == 0:
            rebin_conv[i+1] = f"Less than \${(upper+1):,.0f}"
        elif i == n_bin-1:
            rebin_conv[i+1] = f"\${lower:,.0f} and above"
        else:
            rebin_conv[i+1] = f"\${lower:,.0f} - \${upper:,.0f}"
        # Determine bins
        idx = (df[new_col] >= lower) & (df[new_col] < upper)
        df.loc[idx, new_col] = i+1
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Ordinal'
        new_row.loc[new_col, 'Name'] = 'Income per household member'
        new_row.at[new_col, 'Conversion'] = rebin_conv
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def rebin_race(df, data_dict):
    # Rebin RRACE
    # RACIAL_MINORITY = 1) No
    # 1) White
    # RACIAL_MINORITY = 2) Yes
    # 2) Black
    # 3) Asian
    # 4) Other/Mixed
    ref_col = "RRACE"
    new_col = "RMINORITY"
    rebin_map = {1: 1, 2: 2, 3: 2, 4: 2}
    rebin_conversion = {1: 'No', 2: 'Yes'}
    rebin_name = "Racial minority"
    df[new_col] = df[ref_col].replace(rebin_map)
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = rebin_name
        new_row.at[new_col, 'Conversion'] = rebin_conversion
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def rebin_school_enroll(df, data_dict):
    # Rebin SCHOOLENROLL
    # 1) None
    # 2) Public school
    # 3) Private school
    # 4) Public and private
    # Based on TENROLLPUB, TENROLLPRV
    new_col = "SCHOOLENROLL"
    idx_pub = df['TENROLLPUB'] > 0
    idx_prv = df['TENROLLPRV'] > 0
    df[new_col] = float('nan')
    df.loc[(~idx_pub)&(~idx_prv), new_col] = 1
    df.loc[(idx_pub)&(~idx_prv), new_col] = 2
    df.loc[(~idx_pub)&(idx_prv), new_col] = 3
    df.loc[(idx_pub)&(idx_prv), new_col] = 4
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = "School enrollment"
        new_row.at[new_col, 'Conversion'] = {1: 'None',
                                              2: 'Public school',
                                              3: 'Private school',
                                              4: 'Public and private'}
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def rebin_tenure_column(df, data_dict):
    # Rebin TENURE
    # TENURE_STATUS = 1) Owned
    # 1) Owned, free and clear
    # 2) Owned, with loan/mortgage
    # TENURE_STATUS = 2) Rented
    # 3) Rented
    # TENURE_STATUS = 3) Occupied without payment
    # 4) Occupied without payment
    ref_col = "TENURE"
    new_col = "TENURE_STATUS"
    rebin_map = {1: 1, 2: 1, 3: 2, 4: 3}
    rebin_conversion = {1: 'Owned', 2: 'Rented', 3: 'Occupied without payment'}
    rebin_name = "Tenure status"
    df[new_col] = df[ref_col].replace(rebin_map)
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = rebin_name
        new_row.at[new_col, 'Conversion'] = rebin_conversion
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def rebin_livqtr_column(df, data_dict):
    # Rebin LIVQTRRV
    # LIVQTR_REBIN = 1) Single-family
    # 2) Detached single-family
    # 3) Attached single-family
    # LIVQTR_REBIN = 2) Multi-family
    # 4) Apartment building, 2 units
    # 5) Apartment building, 3-4 units
    # 6) Apartment building, 5+ units
    # LIVQTR_REBIN = 3) Mobile home
    # 1) A mobile home
    # LIVQTR_REBIN = 4) Other
    # 7) Boat, RV, van, etc
    ref_col = "LIVQTRRV"
    new_col = "DWELLTYPE"
    rebin_map = {1: 3, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 4}
    rebin_conversion = {1: 'Single-family', 2: 'Multi-family', 3: 'Mobile home', 4: 'Other (boat, RV, van, etc.)'}
    rebin_name = "Dwelling type"
    df[new_col] = df[ref_col].replace(rebin_map)
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = rebin_name
        new_row.at[new_col, 'Conversion'] = rebin_conversion
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def append_livqtr_columns(df, data_dict):
    # Transform LIVQTRRV into dummy variables
    # LIVQTR_MOBILE = 1) Yes, 0) No
    # 1) A mobile home
    # LIVQTR_OTHER = 1) Yes, 0) No
    # 7) Boat, RV, van, etc
    # LIVQTR_SINGLE = 1) Yes, 0) No
    # 2) Detached single-family
    # 3) Attached single-family
    # LIVQTR_MULTI = 1) Yes, 0) No
    # 4) Apartment building, 2 units
    # 5) Apartment building, 3-4 units
    # 6) Apartment building, 5+ units
    ref_col = "LIVQTRRV"
    dummy_map = {
        "LIVQTR_OTHER":  {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1},
        "LIVQTR_MOBILE":  {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
        "LIVQTR_SINGLE": {1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
        "LIVQTR_MULTI":  {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0},
    }
    dummy_name = {
        "LIVQTR_OTHER":  "Dwelling type: Boat, RV, van, etc.",
        "LIVQTR_MOBILE":  "Dwelling type: Mobile home",
        "LIVQTR_SINGLE": "Dwelling type: Single-family",
        "LIVQTR_MULTI":  "Dwelling type: Multi-family",
    }
    for dummy_col in dummy_map:
        df[dummy_col] = df[ref_col].replace(dummy_map[dummy_col])
        if dummy_col not in data_dict.index:
            new_row = pd.DataFrame(index=[dummy_col], columns=data_dict.columns)
            new_row.loc[dummy_col, 'Type'] = 'Nominal'
            new_row.loc[dummy_col, 'Name'] = dummy_name[dummy_col]
            new_row.at[dummy_col, 'Conversion'] = {0: 'No', 1: 'Yes'}
            data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def append_return_window_columns(df, data_dict):
    # ND_HOWLONG
    # 1) Less than a week
    # 2) Less than a month
    # 3) One to six months
    # 4) More than six months
    # 5) Never returned
    nd_howlong = range(1,6)
    nd_dummy = range(1,4) # 1, 2, 3
    for i in nd_dummy:
        new_col = f"RETURN_{i}"
        map_dict = {j: 1 if j <= i else 0 for j in nd_howlong}
        df[new_col] = df["ND_HOWLONG"].replace(map_dict)
        if new_col not in data_dict.index:
            window = data_dict.loc['ND_HOWLONG', 'Conversion'][i]
            new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
            new_row.loc[new_col, 'Type'] = 'Nominal'
            new_row.loc[new_col, 'Name'] = f'Returned within {window.lower()}'
            new_row.at[new_col, 'Conversion'] = {0: 'Did not', 1: f'Returned within {window.lower()}'}
            data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def append_returned_column(df, data_dict):
    # Create new variable for return from ND_HOWLONG
    # 1) Less than a week
    # 2) Less than a month
    # 3) One to six months
    # 4) More than six months
    # 5) Never returned
    # RETURN --> 1: Did not return; 0: Returned
    new_col = "RETURNED"
    map_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
    df[new_col] = df["ND_HOWLONG"].replace(map_dict)
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = 'Returned'
        new_row.at[new_col, 'Conversion'] = {0: 'Returned', 1: 'Did not return'}
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def append_protracted_column(df, data_dict):
    # Create new variable for return from ND_HOWLONG
    # 1) Less than a week
    # 2) Less than a month
    # 3) One to six months
    # 4) More than six months
    # 5) Never returned
    # PROTRACTED --> 1: Protracted displacement; 0: Returned within 6 months
    new_col = "PROTRACTED"
    map_dict = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    df[new_col] = df["ND_HOWLONG"].replace(map_dict)
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = 'Protracted displacement'
        new_row.at[new_col, 'Conversion'] = {0: 'Not protracted', 1: 'Protracted'}
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def append_recovery_column(df, data_dict):
    # Create new variable for return from ND_HOWLONG
    # 1) Less than a week
    # 2) Less than a month
    # 3) One to six months
    # 4) More than six months
    # 5) Never returned
    # RECOVERY --> 1: Displacement > 1 month; 0: Displacement < 1 month or no return
    new_col = "RECOVERY"
    map_dict = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}
    df[new_col] = df["ND_HOWLONG"].replace(map_dict)
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = 'Recovery phase displacement'
        new_row.at[new_col, 'Conversion'] = {0: 'Recovery phase', 1: 'Emergency phase or no return'}
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def append_phase_column(df, data_dict):
    # Create new variable for emergency displacement from ND_HOWLONG
    # 1) Less than a week
    # 2) Less than a month
    # 3) One to six months
    # 4) More than six months
    # 5) Never returned
    # PROTRACTED --> 0: Emergency phase displacement; 1: Recovery phase or no return
    new_col = "PHASE"
    map_dict = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    df[new_col] = df["ND_HOWLONG"].replace(map_dict)
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = 'Displacement phase'
        new_row.at[new_col, 'Conversion'] = {0: 'Emergency phase', 1: 'Recovery phase or no return'}
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def append_phase_return_column(df, data_dict):
    # Create new variable for emergency displacement from ND_HOWLONG
    # 1) Less than a week
    # 2) Less than a month
    # 3) One to six months
    # 4) More than six months
    # 5) Never returned
    # PROTRACTED --> 0: Emergency phase displacement; 1: Recovery phase displacement, 2: Never returned
    new_col = "PHASE_RETURN"
    map_dict = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    df[new_col] = df["ND_HOWLONG"].replace(map_dict)
    if new_col not in data_dict.index:
        new_row = pd.DataFrame(index=[new_col], columns=data_dict.columns)
        new_row.loc[new_col, 'Type'] = 'Nominal'
        new_row.loc[new_col, 'Name'] = 'Displacement phase'
        new_row.at[new_col, 'Conversion'] = {0: 'Emergency phase', 1: 'Recovery phase', 2: 'Not returned'}
        data_dict = pd.concat([data_dict, new_row], axis=0)
    return df, data_dict


def convert_birth_year_to_age_bin(df, data_dict):
    # Survey baseline year
    survey_year = 2022
    # Determine bins
    age_bins = [0, 24, 34, 44, 54, 64, 74, 1000]
    # Determine index of bins
    n_bins = len(age_bins) - 1
    age_idx = range(n_bins)
    # Create friendly strings for bins
    age_strs = [f'{age_bins[i]+1} - {age_bins[i+1]}' for i in range(n_bins)]
    age_strs[0] = f'{age_bins[1]} or less'
    age_strs[-1] = f'{age_bins[-2]+1}+'
    conversion = {age_idx[i]: age_strs[i] for i in range(n_bins)}
    # Add new column for age bins
    age_col = 'AGE_BIN'
    birthyear_col = 'TBIRTH_YEAR'
    df[age_col] = survey_year - df[birthyear_col].astype(int)
    for i in range(n_bins):
        idx = (df[age_col] > age_bins[i]) & (df[age_col] <= age_bins[i+1])
        df.loc[idx, age_col] = age_idx[i]
    # Adjust data dictionary
    if age_col not in data_dict.index:
        new_row = pd.DataFrame(index=[age_col], columns=data_dict.columns)
        new_row.loc[age_col, 'Type'] = 'Ordinal'
        new_row.loc[age_col, 'Name'] = 'Age'
        new_row.at[age_col, 'Conversion'] = conversion
        data_dict = pd.concat([data_dict, new_row], axis=0)
    else:
        data_dict.at[age_col, 'Conversion'] = conversion
    # Return result
    return df, data_dict


def convert_hh_size_to_bin(df, data_dict):
    # Determine bins
    hh_bins = [0, 1, 2, 4, 7, 1000]
    # Determine index of bins
    n_bins = len(hh_bins) - 1
    hh_idx = range(n_bins)
    # Create friendly strings for bins
    hh_strs = [f'{hh_bins[i]+1} - {hh_bins[i+1]}' for i in range(n_bins)]
    hh_strs[0] = '1'
    hh_strs[1] = '2'
    hh_strs[-1] = f'{hh_bins[-2]+1}+'
    conversion = {hh_idx[i]: hh_strs[i] for i in range(n_bins)}
    # Add new column for age bins
    hh_bin_col = 'HH_BIN'
    hh_col = 'THHLD_NUMPER'
    df[hh_bin_col] = float('nan')
    for i in range(n_bins):
        idx = (df[hh_col] > hh_bins[i]) & (df[hh_col] <= hh_bins[i+1])
        df.loc[idx, hh_bin_col] = hh_idx[i]
    # Adjust data dictionary
    if hh_bin_col not in data_dict.index:
        new_row = pd.DataFrame(index=[hh_bin_col], columns=data_dict.columns)
        new_row.loc[hh_bin_col, 'Type'] = 'Ordinal'
        new_row.loc[hh_bin_col, 'Name'] = 'Household size'
        new_row.at[hh_bin_col, 'Conversion'] = conversion
        data_dict = pd.concat([data_dict, new_row], axis=0)
    else:
        data_dict.at[hh_bin_col]['Conversion'] = conversion
    # Return result
    return df, data_dict


def convert_rent_to_bin(df, data_dict):
    # Determine bins
    rent_bins = [0, 400, 800, 1200, 2000, 10000]
    # Determine index of bins
    n_bins = len(rent_bins) - 1
    rent_idx = range(n_bins)
    # Create friendly strings for bins
    rent_strs = [f'\${rent_bins[i]:,.0f} - \${rent_bins[i+1]:,.0f}' for i in range(n_bins)]
    rent_strs[0] = f'Less than \${rent_bins[2]:,.0f}'
    rent_strs[-1] = f'\${rent_bins[-2]:,.0f} or more'
    conversion = {rent_idx[i]: rent_strs[i] for i in range(n_bins)}
    # Add new column for age bins
    rent_bin_col = 'RENT_BIN'
    rent_col = 'TRENTAMT'
    df[rent_bin_col] = float('nan')
    for i in range(n_bins):
        idx = (df[rent_col] > rent_bins[i]) & (df[rent_col] <= rent_bins[i+1])
        df.loc[idx, rent_bin_col] = rent_idx[i]
    # Adjust data dictionary
    if rent_bin_col not in data_dict.index:
        new_row = pd.DataFrame(index=[rent_bin_col], columns=data_dict.columns)
        new_row.loc[rent_bin_col, 'Type'] = 'Ordinal'
        new_row.loc[rent_bin_col, 'Name'] = 'Rent (per month)'
        new_row.at[rent_bin_col, 'Conversion'] = conversion
        data_dict = pd.concat([data_dict, new_row], axis=0)
    else:
        data_dict.loc[rent_bin_col]['Conversion'] = conversion
    # Return result
    return df, data_dict