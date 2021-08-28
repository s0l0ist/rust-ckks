extern crate nalgebra as na;

use ckks::ckks::encoder::Encoder;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use na::DMatrix;
use num_complex::Complex64;
use rand::{distributions::Standard, Rng};

fn gen_rand_complex_vec(num_elements: usize) -> Vec<Complex64> {
    let mut rng = rand::thread_rng();
    let v: Vec<f64> = (&mut rng)
        .sample_iter(Standard)
        .take(num_elements)
        .collect();
    v.iter().map(|&x| Complex64::new(x, x)).collect()
}

fn pi(g: &mut BenchmarkGroup<criterion::measurement::WallTime>) {
    let scale = 64.0;
    let num_elements = 1024;
    let cv = gen_rand_complex_vec(num_elements);

    let plain = DMatrix::from_vec(cv.len(), 1, cv);
    let encoder = Encoder::new(num_elements, scale);

    g.bench_with_input(BenchmarkId::new("pi", num_elements), &plain, |b, p| {
        b.iter(|| encoder.pi(&p));
    });
}

fn pi_inverse(g: &mut BenchmarkGroup<criterion::measurement::WallTime>) {
    let scale = 64.0;
    let num_elements = 1024;
    let cv = gen_rand_complex_vec(num_elements);

    let plain = DMatrix::from_vec(cv.len(), 1, cv);
    let encoder = Encoder::new(num_elements, scale);

    g.bench_with_input(
        BenchmarkId::new("pi_inverse", plain.len()),
        &plain,
        |b, p| {
            b.iter(|| encoder.pi_inverse(&p));
        },
    );
}

fn bench_ckks(c: &mut Criterion) {
    let mut group = c.benchmark_group("CKKS");
    pi(&mut group);
    pi_inverse(&mut group);

    group.finish();
}

criterion_group!(benches, bench_ckks);
criterion_main!(benches);
