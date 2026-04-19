//! Throughput benchmark. Run with `cargo run --release --example bench`.
//!
//! Loads the embedded default model and runs N samples through `process_sample`,
//! reporting samples/sec and the real-time budget used at 44.1 kHz.

use std::time::Instant;
use tcn_clap::model::TcnModel;

const DEFAULT_MODEL_JSON: &str = include_str!("../assets/tcn.json");

fn main() {
    let mut model = TcnModel::load_from_json_str(DEFAULT_MODEL_JSON)
        .expect("load embedded model");
    model.update_conditioning(0.0, 0.5);

    let n: usize = 200_000;
    let input: Vec<f32> = (0..n)
        .map(|t| 0.3 * (t as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
        .collect();
    let mut output = vec![0.0f32; n];

    // Warm up the JIT / caches.
    for i in 0..1000 {
        output[i] = model.process_sample(input[i]);
    }

    let t0 = Instant::now();
    for i in 0..n {
        output[i] = model.process_sample(input[i]);
    }
    let elapsed = t0.elapsed();

    let samples_per_sec = n as f64 / elapsed.as_secs_f64();
    let ns_per_sample = elapsed.as_nanos() as f64 / n as f64;
    let real_time_ratio = samples_per_sec / 44100.0;

    println!("processed {} samples in {:?}", n, elapsed);
    println!("  {:.2} MSamples/sec", samples_per_sec / 1e6);
    println!("  {:.1} ns/sample", ns_per_sample);
    println!("  {:.2}× real-time at 44.1 kHz ({}% of budget)",
             real_time_ratio, (100.0 / real_time_ratio) as u32);
    println!("  output peak = {:.4}",
             output.iter().copied().fold(0.0f32, |m, v| m.max(v.abs())));
}
