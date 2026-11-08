use plotters::prelude::*;
use ndarray::{Array2, Array1};

pub fn plot_dataset_clusters(
    x_data: &Array2<f64>,
    y_data: &Array1<f64>,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;

    for i in 0..x_data.nrows() {
        let x = x_data[[i, 0]];
        let y = x_data[[i, 1]];
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
    }

    let x_pad = (max_x - min_x) * 0.1;
    let y_pad = (max_y - min_y) * 0.1;

    // Use f64 ranges to match array and cartesian coordinate expectations
    let mut chart = ChartBuilder::on(&root)
        .caption("Dataset Clusters by Class", ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (min_x - x_pad)..(max_x + x_pad), 
            (min_y - y_pad)..(max_y + y_pad)
        )?;

    chart.configure_mesh()
        .x_desc("Feature X1")
        .y_desc("Feature X2")
        .draw()?;

    let mut class_0: Vec<(f64, f64)> = Vec::new();
    let mut class_1: Vec<(f64, f64)> = Vec::new();

    for i in 0..x_data.nrows() {
        let x = x_data[[i, 0]];
        let y = x_data[[i, 1]];
        let label = y_data[i].round() as i32;

        if label == 0 {
            class_0.push((x, y));
        } else {
            class_1.push((x, y));
        }
    }

    chart.draw_series(
        class_0.iter().map(|coords| Circle::new(*coords, 5, RED.filled()))
    )?
    .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()))
    .label("Class 0"); // Fixed method name

    chart.draw_series(
        class_1.iter().map(|coords| Circle::new(*coords, 5, BLUE.filled()))
    )?
    .legend(|(x, y)| Circle::new((x, y), 5, BLUE.filled()))
    .label("Class 1"); // Fixed method name

    // Fixed method name (plural)
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    println!("Cluster plot successfully saved as {}!", file_name);

    Ok(())
}