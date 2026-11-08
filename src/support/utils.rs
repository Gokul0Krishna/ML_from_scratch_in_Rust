use polars::prelude::*;

fn load_train_test_split(
    file_path: &str, 
    target_column: &str, 
    train_ratio: f64
) -> PolarsResult<(DataFrame, DataFrame, Column, Column)> {
    
    // 1. Load the CSV file eagerly
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?;

    // 2. Separate Y (target) from X (features)
    let y = df.column(target_column)?.clone();
    let x = df.drop(target_column)?;

    // 3. Calculate train and test sizes based on the ratio (e.g., 0.8 for 80%)
    let total_rows = x.height();
    let train_size = (total_rows as f64 * train_ratio) as usize;
    let test_size = total_rows - train_size;

    // 4. Split X and Y into Train and Test sets using row slicing
    let x_train = x.slice(0, train_size);
    let x_test = x.slice(train_size as i64, test_size);

    let y_train = y.slice(0, train_size);
    let y_test = y.slice(train_size as i64, test_size);

    Ok((x_train, x_test, y_train, y_test))
}


pub fn iris_load() -> PolarsResult<(DataFrame,DataFrame,Column,Column)> {
    load_train_test_split("src/Data/Iris.csv", "Species", 0.8)
}

pub fn diabetes_load() -> PolarsResult<(DataFrame, DataFrame, Column, Column)> {
    load_train_test_split("src/Data/diabetes.csv", "Outcome", 0.8)
}