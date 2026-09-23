feature_audit = pd.DataFrame({
    "feature": X.columns,
    "dtype": X.dtypes.astype(str).values,
    "missing": X.isna().sum().values,
    "n_unique": X.nunique().values,
    "min": X.min().values,
    "max": X.max().values,
})

print(feature_audit)
