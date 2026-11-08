use crate::support::utils::diabetes_load;

pub fn Knn(k: i32) -> polars::prelude::PolarsResult<()> {
    let (x_train, x_test, y_train, y_test) = diabetes_load()?;
    print!("{:?} {:?} {:?} {:?}", x_train, x_test, y_train, y_test);
    Ok(())
}