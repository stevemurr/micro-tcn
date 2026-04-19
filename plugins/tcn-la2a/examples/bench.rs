//! Throughput benchmark. Run with `cargo run --release --example bench`.

use std::time::Instant;
use tcn_plugin_core::TcnModel;

const DEFAULT_MODEL_JSON: &str = include_str!("../assets/tcn.json");
const BLOCK_SIZE: usize = 512;

fn bench_block(model: &mut TcnModel, input: &[f32], output: &mut [f32]) {
    let n = input.len();
    output.copy_from_slice(input);
    let mut offset = 0;
    while offset + BLOCK_SIZE <= n {
        model.process_block_inplace(&mut output[offset..offset + BLOCK_SIZE]);
        offset += BLOCK_SIZE;
    }
    if offset < n {
        model.process_block_inplace(&mut output[offset..n]);
    }
}

fn main() {
    let mut model = TcnModel::load_from_json_str(DEFAULT_MODEL_JSON)
        .expect("load embedded model");
    // LA2A conditioning: [limit (0/1), peak_reduction (0..1)].
    model.update_conditioning(&[0.0, 0.5]);

    let n: usize = 200_000;
    let input: Vec<f32> = (0..n)
        .map(|t| 0.3 * (t as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
        .collect();
    let mut output = vec![0.0f32; n];

    // Warm up caches.
    bench_block(&mut model, &input[..BLOCK_SIZE * 4], &mut output[..BLOCK_SIZE * 4]);
    model.reset();

    let t0 = Instant::now();
    bench_block(&mut model, &input, &mut output);
    let elapsed = t0.elapsed();

    let samples_per_sec = n as f64 / elapsed.as_secs_f64();
    let ns_per_sample = elapsed.as_nanos() as f64 / n as f64;
    let real_time_ratio = samples_per_sec / 44100.0;

    println!("block_size = {BLOCK_SIZE}");
    println!("processed {} samples in {:?}", n, elapsed);
    println!("  {:.2} MSamples/sec", samples_per_sec / 1e6);
    println!("  {:.1} ns/sample  ({:.1} µs/block)", ns_per_sample, ns_per_sample * BLOCK_SIZE as f64 / 1000.0);
    println!("  {:.1}× real-time at 44.1 kHz  ({}% of budget)",
             real_time_ratio, (100.0 / real_time_ratio) as u32);
    println!("  output peak = {:.4}",
             output.iter().copied().fold(0.0f32, |m, v| m.max(v.abs())));
}
