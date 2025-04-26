use ndarray::{Array1, Array2};
use plotly::Plot;
use std::iter::zip;

pub fn plot_ds1(trace: &Array2<f64>) {
    let mut plot = Plot::new();

    for row in trace.rows().into_iter().enumerate() {
        let (i, trace) = row;

        let x_indices: Vec<usize> = (0..trace.len()).collect();

        use plotly::common::Mode;

        let power_trace = plotly::Scatter::new(x_indices.clone(), trace.to_vec())
            .name(format!("Power {i}"))
            .mode(Mode::Lines);

        plot.add_trace(power_trace);
    }

    plot.show();
}

pub fn plot_ds2(trace: &Array2<f64>, clock: &Array2<f64>) {
    let mut plot = Plot::new();

    for row in zip(trace.rows(), clock.rows()).enumerate() {
        let (i, (trace, clock)) = row;

        let x_indices: Vec<usize> = (0..trace.len()).collect();

        use plotly::common::Mode;

        let power_trace = plotly::Scatter::new(x_indices.clone(), trace.to_vec())
            .name(format!("Power {i}"))
            .mode(Mode::Lines);

        let clock_trace = plotly::Scatter::new(x_indices, clock.to_vec())
            .name(format!("Clock {i}"))
            .mode(Mode::Lines);

        plot.add_trace(power_trace);
        plot.add_trace(clock_trace);
    }

    plot.show();
}

pub fn plot_clocks(clock_a: &Array1<f64>, clock_b: &Array1<f64>) {
    let mut plot = Plot::new();

    let x_indices: Vec<usize> = (0..clock_a.len()).collect();

    use plotly::common::Mode;

    let power_trace = plotly::Scatter::new(x_indices.clone(), clock_a.to_vec())
        .name("Clock A")
        .mode(Mode::Lines);

    let clock_trace = plotly::Scatter::new(x_indices, clock_b.to_vec())
        .name("Clock B")
        .mode(Mode::Lines);

    plot.add_trace(power_trace);
    plot.add_trace(clock_trace);
    plot.show()
}

pub fn plot_resampled_trace(
    ref_clock: &Array1<f64>,
    original: &Array1<f64>,
    resampled: &Array1<f64>,
) {
    let mut plot = Plot::new();

    let x_indices: Vec<usize> = (0..original.len()).collect();

    use plotly::common::Mode;

    let clock_trace = plotly::Scatter::new(x_indices.clone(), ref_clock.to_vec())
        .name("Reference clock")
        .mode(Mode::Lines);

    let power_trace = plotly::Scatter::new(x_indices.clone(), original.to_vec())
        .name("Original")
        .mode(Mode::Lines);

    let resampled_trace = plotly::Scatter::new(x_indices, resampled.to_vec())
        .name("Resampled")
        .mode(Mode::Lines);

    plot.add_trace(clock_trace);
    plot.add_trace(power_trace);
    plot.add_trace(resampled_trace);
    plot.show()
}
