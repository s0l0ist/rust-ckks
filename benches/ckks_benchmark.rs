extern crate nalgebra as na;

use ckks::ckks::encoder::Encoder;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use na::DMatrix;
use num_complex::Complex64;
use rand::{distributions::Standard, Rng};

const SCALE: f64 = 64.0;
const NUM_ELEMENTS: usize = 1024;

// Generates a vector of a given size with random complex (real, im) values
fn gen_rand_complex_vec(num_elements: usize) -> Vec<Complex64> {
    let mut rng = rand::thread_rng();
    let v: Vec<(f64, f64)> = (&mut rng)
        .sample_iter(Standard)
        .take(num_elements)
        .collect();
    v.iter()
        .map(|&x| num_complex::Complex64::new(x.0, x.1))
        .collect()
}

fn bench_pi(c: &mut Criterion) {
    let cv = gen_rand_complex_vec(NUM_ELEMENTS);
    let plain = DMatrix::from_vec(cv.len(), 1, cv);
    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

    let mut group = c.benchmark_group("CKKS");
    group.bench_with_input(BenchmarkId::new("pi", NUM_ELEMENTS), &plain, |b, p| {
        b.iter(|| encoder.pi(&p));
    });
    group.bench_with_input(
        BenchmarkId::new("pi_inverse", plain.len()),
        &plain,
        |b, p| {
            b.iter(|| encoder.pi_inverse(&p));
        },
    );

    group.finish();
}

fn bench_create_sigma_r_basis(c: &mut Criterion) {
    let xi = (2.0 * std::f64::consts::PI * Complex64::new(0.0, 1.0) / (NUM_ELEMENTS as f64)).exp();
    let n = NUM_ELEMENTS / 2;

    let mut group = c.benchmark_group("CKKS");
    group.bench_with_input(
        BenchmarkId::new("create_sigma_r_basis", NUM_ELEMENTS),
        &xi,
        |b, &x| {
            b.iter(|| Encoder::create_sigma_r_basis(x, n));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("vandermonde", NUM_ELEMENTS),
        &xi,
        |b, &x| {
            b.iter(|| Encoder::vandermonde(x, n));
        },
    );

    group.finish();
}

fn bench_compute_basis_coordinates(c: &mut Criterion) {
    let cv = gen_rand_complex_vec(NUM_ELEMENTS / 2);
    let mat = DMatrix::from_vec(cv.len(), 1, cv);

    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

    let mut group = c.benchmark_group("CKKS");
    group.bench_with_input(
        BenchmarkId::new("compute_basis_coordinates", NUM_ELEMENTS),
        &mat,
        |b, x| {
            b.iter(|| black_box(encoder.compute_basis_coordinates(x)));
        },
    );

    group.finish();
}

fn bench_coordinate_wise_random_rounding(c: &mut Criterion) {
    let cv = gen_rand_complex_vec(NUM_ELEMENTS);
    let rv: Vec<f64> = cv.iter().map(|&x| x.re).collect();
    let mat = DMatrix::from_vec(1, cv.len(), rv);

    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

    let mut group = c.benchmark_group("CKKS");
    group.bench_with_input(
        BenchmarkId::new("coordinate_wise_random_rounding", NUM_ELEMENTS),
        &mat,
        |b, x| {
            b.iter(|| black_box(encoder.coordinate_wise_random_rounding(x)));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("round_coordinates", NUM_ELEMENTS),
        &mat,
        |b, x| {
            b.iter(|| black_box(encoder.round_coordinates(x)));
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_pi,
    bench_create_sigma_r_basis,
    bench_compute_basis_coordinates,
    bench_coordinate_wise_random_rounding
);
criterion_main!(benches);
