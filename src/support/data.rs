use polars::prelude::*;

fn load_train_test_split(
    file_path: &str,
    target_column: &str,
    train_ratio: f64,
) -> PolarsResult<(DataFrame, DataFrame, Column, Column)> {
    let data = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?;

    let labels = data
        .column(target_column)?
        .cast(&DataType::Float64)?;
    let feature_columns = data
        .get_columns()
        .iter()
        .filter(|column| column.name().as_str() != target_column)
        .map(|column| column.cast(&DataType::Float64))
        .collect::<PolarsResult<Vec<_>>>()?;
    let features = DataFrame::new(feature_columns)?;

    let train_size = (features.height() as f64 * train_ratio) as usize;
    let test_size = features.height() - train_size;
    let x_train = features.slice(0, train_size);
    let x_test = features.slice(train_size as i64, test_size);
    let y_train = labels.slice(0, train_size);
    let y_test = labels.slice(train_size as i64, test_size);

    Ok((x_train, x_test, y_train, y_test))
}

pub fn diabetes_load() -> PolarsResult<(DataFrame, DataFrame, Column, Column)> {
    load_train_test_split("src/Data/diabetes.csv", "Outcome", 0.8)
}