use ckks::ckks::encoder::Encoder;
use core::hint::black_box;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DMatrix;
use num_complex::Complex64;
use rand::Rng;
use rand::distr::StandardUniform;

const SCALE: f64 = 64.0;
const NUM_ELEMENTS: usize = 1024;

/// Generate a complex vector of the given length with uniform real/imag parts
fn gen_rand_complex_vec(n: usize) -> Vec<Complex64> {
    let real: Vec<f64> = rand::rng().sample_iter(StandardUniform).take(n).collect();
    let imag: Vec<f64> = rand::rng().sample_iter(StandardUniform).take(n).collect();
    real.into_iter()
        .zip(imag)
        .map(|(re, im)| Complex64::new(re, im))
        .collect()
}

fn bench_projection(c: &mut Criterion) {
    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
    let mat = DMatrix::from_vec(NUM_ELEMENTS, 1, gen_rand_complex_vec(NUM_ELEMENTS));

    let mut group = c.benchmark_group("CKKS/projection");
    group.bench_with_input(BenchmarkId::new("pi", NUM_ELEMENTS), &mat, |b, x| {
        b.iter(|| black_box(encoder.pi(x)));
    });
    group.bench_with_input(
        BenchmarkId::new("pi_inverse", NUM_ELEMENTS),
        &mat,
        |b, x| {
            b.iter(|| black_box(encoder.pi_inverse(x)));
        },
    );
    group.finish();
}

fn bench_basis_ops(c: &mut Criterion) {
    let xi = (2.0 * std::f64::consts::PI * Complex64::i() / (NUM_ELEMENTS as f64)).exp();
    let n = NUM_ELEMENTS / 2;

    let mut group = c.benchmark_group("CKKS/basis");
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

fn bench_encoding_pipeline(c: &mut Criterion) {
    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
    let half_mat = DMatrix::from_vec(NUM_ELEMENTS / 2, 1, gen_rand_complex_vec(NUM_ELEMENTS / 2));
    let real_row = DMatrix::from_vec(1, NUM_ELEMENTS / 2, half_mat.iter().map(|c| c.re).collect());

    let mut group = c.benchmark_group("CKKS/encoding_pipeline");

    group.bench_with_input(
        BenchmarkId::new("compute_basis_coordinates", NUM_ELEMENTS),
        &half_mat,
        |b, m| {
            b.iter(|| black_box(encoder.compute_basis_coordinates(m)));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("round_coordinates", NUM_ELEMENTS),
        &real_row,
        |b, m| {
            b.iter(|| black_box(encoder.round_coordinates(m)));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("coordinate_wise_random_rounding", NUM_ELEMENTS),
        &real_row,
        |b, m| {
            b.iter(|| black_box(encoder.coordinate_wise_random_rounding(m)));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("sigma_r_discretization", NUM_ELEMENTS),
        &half_mat,
        |b, m| {
            b.iter(|| black_box(encoder.sigma_r_discretization(m)));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("sigma_inverse", NUM_ELEMENTS),
        &half_mat,
        |b, m| {
            b.iter(|| black_box(encoder.sigma_inverse(m)));
        },
    );

    group.finish();
}

fn bench_full_encode(c: &mut Criterion) {
    let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
    let quarter_mat =
        DMatrix::from_vec(NUM_ELEMENTS / 4, 1, gen_rand_complex_vec(NUM_ELEMENTS / 4));

    let mut group = c.benchmark_group("CKKS/encode");
    group.bench_with_input(
        BenchmarkId::new("encode", NUM_ELEMENTS),
        &quarter_mat,
        |b, m| {
            b.iter(|| black_box(encoder.encode(m)));
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_projection,
    bench_basis_ops,
    bench_encoding_pipeline,
    bench_full_encode
);
criterion_main!(benches);
