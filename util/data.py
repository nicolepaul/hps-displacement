import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression


def create_correlation_matx(data, corr_tol=0.7, absolute=True, method="spearman"):
    # TODO: Consider weights?

    # Determine (absolute value of) correlations
    corr_matx = data.corr(method=method)
    if absolute:
        corr_matx = corr_matx.abs()

    # Get upper and lower triangle
    upper = corr_matx.where(np.triu(np.ones(corr_matx.shape), k=1).astype(bool))
    lower = corr_matx.where(np.tril(np.ones(corr_matx.shape)).astype(bool))

    # Find features with correlation greater than a given threshold
    to_drop = [column for column in upper.columns if any(upper[column].abs() > corr_tol)]
    drop_str = '\n'.join([f'    {drop}' for drop in to_drop])
    print(f"These {len(to_drop):,.0f} columns exceed the tolerance of {corr_tol:.0%}:\n{drop_str}")

    # Return result
    return corr_matx, upper, lower, to_drop



def create_crosstab(
    data, data_dict, main_factor, curr_factor, weights="HWEIGHT", samples=False
):
    # Extract relevant variable name maps
    varname_map = {
        main_factor: data_dict.loc[main_factor, "Name"],
        curr_factor: data_dict.loc[curr_factor, "Name"],
    }
    # Extract relevant value maps
    main_map = data_dict.loc[main_factor, "Conversion"]
    curr_map = data_dict.loc[curr_factor, "Conversion"]

    # Remove unknown/unreported values
    rmv_values = [-88, -99]
    rmv_idx = (~data[main_factor].isin(rmv_values)) & (
        ~data[curr_factor].isin(rmv_values)
    )
    df = data[rmv_idx].copy()

    # Arrange crosstab
    df[main_factor] = df[main_factor].astype("category").map(main_map)
    df.rename(columns=varname_map, inplace=True)
    crosst = pd.crosstab(
        index=df[varname_map[curr_factor]],
        columns=df[varname_map[main_factor]],
        values=df[weights],
        aggfunc="sum",
        normalize="index",
    )

    # Add sample size if requested; otherwise use basic name map
    if samples:
        sample_sizes = df.groupby(varname_map[curr_factor])["SCRAM"].agg("count")
        for key in curr_map:
            if key not in rmv_values and "\n(n=" not in curr_map[key]:
                N = float(sample_sizes.loc[key])
                curr_map[key] = f"{curr_map[key]}\n(n={N:,.0f})"
        crosst.index = crosst.index.astype("category").map(curr_map)
    else:
        # Rename index
        crosst.rename(index=curr_map, inplace=True)

    # Return result
    return crosst


def expand_data_by_custom_weights(df, weights):
    # Expand rows by weight column

    # Expand dataframe
    df_in = df.copy().reset_index()
    df_expanded = df_in.reindex(df_in.index.repeat(weights))

    # Report back
    n_original, n_expanded = df_in.shape[0], df_expanded.shape[0]
    # print(f"Expanded {n_original:,.0f} rows into {n_expanded:,.0f} using custom weights")

    # Return result
    return df_expanded


def expand_train_test_split_by_weights(X, y, w):

    X_w = expand_data_by_custom_weights(X, w).drop(columns='index')
    y_w = expand_data_by_custom_weights(y, w).drop(columns='index')

    return X_w, y_w


def get_regression_summary(model, X_train, y_train, class_names):

    # Get regression coefficients, log odds, p values
    coef = np.transpose(model.coef_)
    odds = np.exp(coef)
    freg = f_regression(X_train, y_train)
    p = freg[1] # TODO: Check
    summary = None
    if len(class_names) > 2:
        print('multinomial')
        summaries = [pd.DataFrame(zip(coef[:,key-1], odds[:,key-1]),
                                index=X_train.columns,
                                columns=['coef', 'odds']).add_prefix(f"{class_names[key]}_") for key in class_names]
        summary = pd.concat(summaries, axis=1)
        summary.loc['intercept', [f"{class_names[key]}_coef" for key in class_names]] = model.intercept_
        c = sorted(summary.columns[1:], key=lambda x: str(x.split('_')[-1]))
        summary = summary.reindex(summary.columns[:1].tolist() + c, axis=1)
        
    else:
        summary = pd.DataFrame(zip(np.squeeze(coef), np.squeeze(odds), np.squeeze(p)), index=X_train.columns, columns=['coef', 'odds', 'p'])
        summary.loc['intercept', 'coef'] = model.intercept_

    return summary