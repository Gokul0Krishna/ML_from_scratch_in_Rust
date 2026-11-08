use crate::support::data::diabetes_load;
use polars::prelude::*;
use std::collections::HashMap;

pub fn knn(k: usize) -> PolarsResult<f64> {
    let (x_train, x_test, y_train, y_test) = diabetes_load()?;
    let x_tr_array = x_train.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    let x_te_array = x_test.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
    let y_tr_array = y_train.f64()?.to_ndarray()?;
    let y_te_array = y_test.f64()?.to_ndarray()?;
    
    let mut correct = 0;
    let total_test = x_te_array.nrows();

    for i in 0..total_test {
        let test_row = x_te_array.row(i);
        let mut distances: Vec<(f64, i32)> = Vec::with_capacity(x_tr_array.nrows());
        
        for j in 0..x_tr_array.nrows() {
            let train_row = x_tr_array.row(j);
            let dist_sq: f64 = test_row.iter()
                .zip(train_row.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum();
            
            let label = y_tr_array[j].round() as i32;
            distances.push((dist_sq, label));
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut vote_counts: HashMap<i32, usize> = HashMap::new();
        for n in 0..k.min(distances.len()) {
            let label = distances[n].1;
            *vote_counts.entry(label).or_insert(0) += 1;
        }

        let predicted_label = vote_counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap_or(0);

        let true_label = y_te_array[i].round() as i32;
        if predicted_label == true_label {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / total_test as f64;
    Ok(accuracy) // Return the f64 accuracy instead of Ok(())
}