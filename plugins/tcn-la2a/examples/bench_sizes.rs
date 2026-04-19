use std::time::Instant;
use tcn_plugin_core::TcnModel;

const DEFAULT_MODEL_JSON: &str = include_str!("../assets/tcn.json");

fn run_blocks(model: &mut TcnModel, buf: &mut [f32], block_size: usize) {
    let mut offset = 0;
    while offset + block_size <= buf.len() {
        model.process_block_inplace(&mut buf[offset..offset + block_size]);
        offset += block_size;
    }
}

fn bench_at(model: &mut TcnModel, block_size: usize) {
    let n: usize = block_size * 1000;
    let input: Vec<f32> = (0..n)
        .map(|t| 0.3 * (t as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
        .collect();
    let mut output = vec![0.0f32; n];

    model.allocate_block_buffers(block_size);
    model.update_conditioning(&[0.0, 0.5]);

    // warm up with correctly-sized blocks
    output.copy_from_slice(&input);
    run_blocks(model, &mut output[..block_size * 8], block_size);
    model.reset();

    output.copy_from_slice(&input);
    let t0 = Instant::now();
    run_blocks(model, &mut output, block_size);
    let elapsed = t0.elapsed();

    let real_time_ratio = (n as f64 / 44100.0) / elapsed.as_secs_f64();
    let cpu_pct = 100.0 / real_time_ratio;
    println!("  N={:4}  {:6.1}× real-time  ({:.1}% CPU/plugin, {:.1}% for 2 plugins)",
        block_size, real_time_ratio, cpu_pct, cpu_pct * 2.0);
}

fn main() {
    let mut model = TcnModel::load_from_json_str(DEFAULT_MODEL_JSON).unwrap();
    println!("Block size sweep (Apple Accelerate SGEMM):");
    for &n in &[32usize, 64, 128, 256, 512] {
        bench_at(&mut model, n);
    }
}
