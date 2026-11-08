mod ml;
mod support;

use ml::knn::Knn;

fn main() -> polars::prelude::PolarsResult<()> {
    Knn(2)
}

