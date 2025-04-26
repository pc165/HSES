#![allow(dead_code, unused)]
use ndarray::{s, Array, Array1, Array2, Array3, ArrayBase, Axis, Ix1, OwnedRepr};
use ndarray_stats::QuantileExt;
use plotly::Plot;
use std::fs::File;
use std::io::{read_to_string, Write};
use std::iter::zip;
use std::path::Path;
use std::{env, io};

const SBOX: [i32; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

fn load_traces(dataset: &str) -> Array3<f64> {
    let traces = (0..16)
        .map(|x| {
            let path = format!("{dataset}/trace{x}.txt");
            let path = Path::new(&path);
            let file = File::open(path).unwrap();
            let trace = read_to_string(file)
                .unwrap()
                .split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .map(|x| if x > 0.8 { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();
            return trace;
        })
        .collect::<Vec<_>>();

    Array::from_shape_vec((16, 150, traces[0].len() / 150), traces.concat()).unwrap()
}

fn load_clocks(dataset: &str) -> Array3<f64> {
    let clocks = (0..16)
        .map(|x| {
            let path = format!("{dataset}/clock{x}.txt");
            let path = Path::new(&path);
            let file = File::open(path).unwrap();
            let clock = read_to_string(file)
                .unwrap()
                .split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .map(|x| if x > 0.5 { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();
            return clock;
        })
        .collect::<Vec<_>>();

    Array::from_shape_vec((16, 150, clocks[0].len() / 150), clocks.concat()).unwrap()
}

fn load_clear_text(path: &str) -> Array2<i32> {
    let path = Path::new(path);
    let file = File::open(path).unwrap();
    let cleartext = read_to_string(file)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse::<i32>())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    Array::from_shape_vec((cleartext.len() / 16, 16), cleartext).unwrap()
}

fn calculate_hamming_weights(clear_text: &Array2<i32>) -> Array3<f64> {
    let hamming_weights = (0..16)
        .flat_map(|byte_index| {
            (0..256)
                .flat_map(|key_guess| {
                    let hw = clear_text
                        .column(byte_index)
                        .map(|byte| SBOX[(byte ^ key_guess) as usize])
                        .map(|sbox_output| sbox_output.count_ones() as f64);
                    hw
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // byte_index, key_guess, hw
    Array3::from_shape_vec((16, 256, 150), hamming_weights).unwrap()
}

fn pearson_correlation(x: &ndarray::ArrayView1<f64>, y: &ndarray::ArrayView1<f64>) -> f64 {
    let mean_x = x.mean().unwrap();
    let mean_y = y.mean().unwrap();

    let diff_x = x - mean_x;
    let diff_y = y - mean_y;

    let numerator = diff_x.dot(&diff_y);

    let denom_x = diff_x.dot(&diff_x).sqrt();
    let denom_y = diff_y.dot(&diff_y).sqrt();

    (numerator / (denom_x * denom_y)).abs()
}

fn plot_ds1(trace: &Array2<f64>) {
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

fn plot_ds2(trace: &Array2<f64>, clock: &Array2<f64>) {
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

fn attack_ds1(dataset: &str) {
    // 150 x 16
    let clear_text = load_clear_text(&format!("{dataset}/cleartext.txt"));
    // 16 x 256 x 150
    let hamming_weights = calculate_hamming_weights(&clear_text);
    // 16 x 150 x 50000
    let traces = load_traces(dataset);
    // let trace = traces.slice(s![0, 0..2, 0..300]).to_owned();
    // plot1(&trace);

    let key = (0..16)
        .map(|byte_index| {
            // 150 x 50000
            let byte_power = traces.index_axis(Axis(0), byte_index);

            // 256 x 150
            let byte_hw = hamming_weights.index_axis(Axis(0), byte_index);

            print!("Calculating byte {byte_index}... ");
            io::stdout().flush().unwrap();

            // 256 x 1
            let byte_correlation = byte_hw.map_axis(Axis(1), |hw| {
                // 50000
                let corr_for_each_sample = byte_power.map_axis(Axis(0), |sample| {
                    // 150 x 150
                    pearson_correlation(&sample, &hw)
                });

                corr_for_each_sample.max().unwrap().to_owned()
            });

            let guess = byte_correlation.argmax().unwrap();

            println!("{}", guess);
            guess as i32
        })
        .collect::<Vec<_>>();

    dbg!(&key);
    dbg!(key.iter().sum::<i32>());
    assert_eq!(key.iter().sum::<i32>(), 1712);
}

fn plot_clocks(clock_a: &Array1<f64>, clock_b: &Array1<f64>) {
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

fn find_edges(clock: &Array1<f64>) -> Array1<f64> {
    Array1::from_vec(
        (1..clock.len())
            .map(|i| {
                if clock[i] > 0.5 && clock[i - 1] < 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>(),
    )
}

fn arg_of_ones(array: &Array1<f64>) -> Array1<isize> {
    array
        .iter()
        .enumerate()
        .filter_map(|(idx, &val)| if val == 1.0 { Some(idx as isize) } else { None })
        .collect()
}

fn calculate_sync_offsets(
    clock_a: &Array1<f64>,
    clock_b: &Array1<f64>,
) -> ArrayBase<OwnedRepr<isize>, Ix1> {
    let edges_a = find_edges(clock_a);
    let edges_b = find_edges(clock_b);
    // plot_clocks(&clock_a, &clock_b);
    // plot_clocks(&edges_a, &edges_b);

    let edge_indices_a = arg_of_ones(&edges_a);
    let edge_indices_b = arg_of_ones(&edges_b);

    let offsets = edge_indices_a - edge_indices_b;

    offsets
}

fn attack_ds2(dataset: &str) {
    // 150 x 16
    let clear_text = load_clear_text(&format!("{dataset}/cleartext.txt"));
    // 16 x 256 x 150
    let hamming_weights = calculate_hamming_weights(&clear_text);
    // 16 x 150 x 50000
    let traces = load_traces(dataset);
    // 16 x 150 x 50000
    let clocks = load_clocks(dataset);

    let clock = clocks.slice(s![0, 0..2, 0..5000]).to_owned();
    // let trace = traces.slice(s![0, 0..2, 0..5000]).to_owned();
    // plot_ds2(&trace, &clock);

    let clock0 = clocks.slice(s![0, 0, ..]).to_owned();
    let clock1 = clocks.slice(s![0, 1, ..]).to_owned();
    let offsets = calculate_sync_offsets(&clock0, &clock1);

    println!("{:#?}", offsets);

    // let key = (0..16).map(|byte_index| 1).collect::<Vec<_>>();
    //
    // dbg!(&key);
    // dbg!(key.iter().sum::<i32>());
    // assert_eq!(key.iter().sum::<i32>(), 1434);
}

fn main() {
    let args = env::args().collect::<Vec<_>>();

    if args.len() != 2 {
        panic!("Expected exactly one argument");
    }

    if args[1].ends_with("1") {
        attack_ds1(args[1].as_str());
    }

    if args[1].ends_with("2") {
        attack_ds2(args[1].as_str());
    }
}
