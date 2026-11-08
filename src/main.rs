mod ml;
mod support;

use ml::knn::knn;
use polars::prelude::PolarsResult;

fn main() -> PolarsResult<()> {
    let accuracy = knn(2)?;
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    Ok(())
}

