use ckks::ckks::encoder::Encoder;
use core::hint::black_box;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DMatrix;
use num_complex::Complex64;
use rand::Rng;
use rand::distr::StandardUniform;

const SCALE: f64 = 64.0;
const NUM_ELEMENTS: usize = 1024;

// Generates a vector of a given size with random complex (real, im) values
fn gen_rand_complex_vec(num_elements: usize) -> Vec<Complex64> {
    let real = rand::rng()
        .sample_iter(StandardUniform)
        .take(num_elements)
        .collect::<Vec<f64>>();

    let img = rand::rng()
        .sample_iter(StandardUniform)
        .take(num_elements)
        .collect::<Vec<f64>>();

    real.iter()
        .zip(img.iter())
        .map(|(&r, &i)| Complex64::new(r, i))
        .collect::<Vec<Complex64>>()
}

fn bench_pi(c: &mut Criterion) {
    let cv = gen_rand_complex_vec(NUM_ELEMENTS);
    let plain = DMatrix::from_vec(cv.len(), 1, cv);
    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

    let mut group = c.benchmark_group("CKKS");
    group.bench_with_input(BenchmarkId::new("pi", NUM_ELEMENTS), &plain, |b, p| {
        b.iter(|| encoder.pi(p));
    });
    group.bench_with_input(
        BenchmarkId::new("pi_inverse", plain.len()),
        &plain,
        |b, p| {
            b.iter(|| black_box(encoder.pi_inverse(p)));
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
            b.iter(|| black_box(Encoder::create_sigma_r_basis(x, n)));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("vandermonde", NUM_ELEMENTS),
        &xi,
        |b, &x| {
            b.iter(|| black_box(Encoder::vandermonde(x, n)));
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

fn bench_rounding(c: &mut Criterion) {
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
fn bench_sigma_r_discretization(c: &mut Criterion) {
    let cv = gen_rand_complex_vec(NUM_ELEMENTS / 2);
    let mat = DMatrix::from_vec(cv.len(), 1, cv);

    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

    let mut group = c.benchmark_group("CKKS");

    group.bench_with_input(
        BenchmarkId::new("sigma_r_discretization", NUM_ELEMENTS),
        &mat,
        |b, x| {
            b.iter(|| black_box(encoder.sigma_r_discretization(x)));
        },
    );

    group.finish();
}

fn bench_sigma_inverse(c: &mut Criterion) {
    let cv = gen_rand_complex_vec(NUM_ELEMENTS / 2);
    let mat = DMatrix::from_vec(cv.len(), 1, cv);

    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

    let mut group = c.benchmark_group("CKKS");

    group.bench_with_input(
        BenchmarkId::new("sigma_inverse", NUM_ELEMENTS),
        &mat,
        |b, x| {
            b.iter(|| black_box(encoder.sigma_inverse(x)));
        },
    );

    group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let cv = gen_rand_complex_vec(NUM_ELEMENTS / 4);
    let mat = DMatrix::from_vec(cv.len(), 1, cv);

    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

    let mut group = c.benchmark_group("CKKS");

    group.bench_with_input(BenchmarkId::new("encode", NUM_ELEMENTS), &mat, |b, x| {
        b.iter(|| black_box(encoder.encode(x)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pi,
    bench_create_sigma_r_basis,
    bench_compute_basis_coordinates,
    bench_rounding,
    bench_sigma_r_discretization,
    bench_sigma_inverse,
    bench_encode
);
criterion_main!(benches);
