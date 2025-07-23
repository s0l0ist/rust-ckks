use crate::ckks::random::UniformRandomGenerator;
use nalgebra::DMatrix;
use num_complex::Complex64;
use rustnomial::{Evaluable, Polynomial};
use std::cell::RefCell;

// Basic CKKS encoder to encode complex vectors into polynomials.
pub struct Encoder {
    pub xi: Complex64,
    pub m: usize,
    pub n: usize,
    pub sigma_r_basis: DMatrix<Complex64>,
    pub sigma_r_basis_norms: Vec<f64>,
    pub vandermonde_t: DMatrix<Complex64>,
    pub scale: f64,
    pub rng: RefCell<UniformRandomGenerator>,
}

impl Encoder {
    pub fn new(m: usize, scale: f64) -> Self {
        let xi = (2.0 * std::f64::consts::PI * Complex64::new(0.0, 1.0) / (m as f64)).exp();
        let n = m / 2;
        let sigma_r_basis = Encoder::create_sigma_r_basis(xi, n);
        let sigma_r_basis_norms = sigma_r_basis
            .row_iter()
            .map(|b| b.iter().map(|x| x.norm_sqr()).sum())
            .collect();
        let vandermonde_t = Encoder::vandermonde(xi, n).transpose();
        let rng = RefCell::new(UniformRandomGenerator::new());

        Self {
            xi,
            m,
            n,
            sigma_r_basis,
            sigma_r_basis_norms,
            vandermonde_t,
            scale,
            rng,
        }
    }

    pub fn create_sigma_r_basis(xi: Complex64, n: usize) -> DMatrix<Complex64> {
        Encoder::vandermonde(xi, n)
    }

    pub fn vandermonde(xi: Complex64, n: usize) -> DMatrix<Complex64> {
        let mut matrix = Vec::with_capacity(n * n);
        for i in 0..n {
            let root = xi.powu((2 * i as u32) + 1);
            let mut power = Complex64::new(1.0, 0.0);
            for _ in 0..n {
                matrix.push(power);
                power *= root;
            }
        }
        DMatrix::from_vec(n, n, matrix)
    }

    pub fn encode(&self, z: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let pi_z = self.pi_inverse(z);
        let scaled_pi_z = pi_z.scale(self.scale);
        let rounded_scale_pi_z = self.sigma_r_discretization(&scaled_pi_z);
        self.sigma_inverse(&rounded_scale_pi_z)
    }

    pub fn pi_inverse(&self, z: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let conj = z.conjugate();
        let conj_rev = conj.iter().rev().cloned();
        let combined = z.iter().cloned().chain(conj_rev);
        DMatrix::from_iterator(z.nrows() * 2, z.ncols(), combined)
    }
    pub fn sigma_r_discretization(&self, z: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let coordinates = self.compute_basis_coordinates(z);
        let rounded_coordinates = self.coordinate_wise_random_rounding(&coordinates);
        self.sigma_r_basis.tr_mul(&rounded_coordinates.transpose())
    }

    pub fn compute_basis_coordinates(&self, z: &DMatrix<Complex64>) -> DMatrix<f64> {
        let z_t = z.transpose();
        let coords = self
            .sigma_r_basis
            .row_iter()
            .zip(&self.sigma_r_basis_norms)
            .map(|(b, &norm_sq)| {
                let dot = z_t.dotc(&b);
                dot.re / norm_sq
            });
        DMatrix::from_iterator(1, self.sigma_r_basis.nrows(), coords)
    }

    pub fn coordinate_wise_random_rounding(
        &self,
        coordinates: &DMatrix<f64>,
    ) -> DMatrix<Complex64> {
        let fractional = self.round_coordinates(coordinates);
        let mut rng_ref = self.rng.borrow_mut();

        let samples = fractional.iter().map(|&x| {
            let choices = [0.0, -1.0];
            let weights = [1.0 - x, x];
            rng_ref.weighted_choice(&choices, &weights)
        });

        let rounded = coordinates
            .iter()
            .zip(samples)
            .map(|(&x, y)| Complex64::new(x + y, 0.0));

        DMatrix::from_iterator(1, coordinates.len(), rounded)
    }

    pub fn round_coordinates(&self, coordinates: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::from_iterator(
            1,
            coordinates.len(),
            coordinates.iter().map(|c| c - c.floor()),
        )
    }

    pub fn sigma_inverse(&self, b: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        self.vandermonde_t.clone().lu().solve(b).unwrap()
    }

    pub fn decode(&self, p: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let rescaled_p = p.unscale(self.scale);
        let z = self.sigma(&rescaled_p);
        self.pi(&z)
    }

    pub fn pi(&self, z: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        z.rows(0, z.nrows() / 2).into_owned()
    }

    pub fn sigma(&self, p: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let poly = self.to_polynomial(p);
        let values = (0..self.n).map(|i| {
            let root = self.xi.powu((2 * i as u32) + 1);
            poly.eval(root)
        });
        DMatrix::from_iterator(self.n, 1, values)
    }

    pub fn to_polynomial(&self, x_coeffs: &DMatrix<Complex64>) -> Polynomial<Complex64> {
        let poly_vec: Vec<Complex64> = x_coeffs.iter().rev().copied().collect();
        Polynomial::new(poly_vec)
    }

    pub fn from_polynomial(&self, poly: &Polynomial<Complex64>) -> DMatrix<Complex64> {
        let mut coeffs = poly.terms.clone();
        coeffs.reverse();
        DMatrix::from_row_slice(self.n, 1, &coeffs)
    }
}

#[cfg(test)]
mod complex {
    use super::*;

    const NUM_ELEMENTS: usize = 8;
    const NUM_ROWS: usize = NUM_ELEMENTS / 2;
    const NUM_COLS: usize = 1;
    const SCALE: f64 = 64.0;

    #[test]
    fn xi() {
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        assert_eq!(
            encoder.xi,
            Complex64::new(0.7071067811865476, 0.7071067811865475)
        );
    }

    #[test]
    fn pi() {
        let plain = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, -5.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let expected = DMatrix::from_vec(
            NUM_ROWS / 2,
            NUM_COLS,
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0)],
        );
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let pi = encoder.pi(&plain);
        assert_eq!(pi, expected);
    }
    #[test]
    fn pi_inverse() {
        let plain = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, -5.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let expected = DMatrix::from_vec(
            NUM_ROWS * 2,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, -5.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(4.0, -0.0),
                Complex64::new(3.0, 5.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(1.0, -0.0),
            ],
        );
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let conjugate = encoder.pi_inverse(&plain);
        assert_eq!(conjugate, expected);
    }
    #[test]
    fn vandermonde() {
        let vandermonde = Encoder::vandermonde(
            Complex64::new(0.7071067811865476, 0.7071067811865475),
            NUM_ELEMENTS / 2, // n is (m / 2)
        );

        let vandermonde_expected = DMatrix::from_row_slice(
            NUM_ROWS,
            NUM_ROWS, // Num of cols is the same as rows
            &vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.7071067811865476, 0.7071067811865475),
                Complex64::new(-0.7071067811865474, 0.7071067811865477),
                Complex64::new(-0.7071067811865479, -0.7071067811865471),
                Complex64::new(0.707106781186547, -0.707106781186548),
                Complex64::new(0.0000000000000002220446049250313, 1.0),
                Complex64::new(-0.0000000000000004440892098500626, -1.0),
                Complex64::new(0.0000000000000011102230246251565, 1.0),
                Complex64::new(-0.0000000000000013877787807814457, -1.0),
                Complex64::new(-0.7071067811865474, 0.7071067811865477),
                Complex64::new(0.707106781186548, 0.707106781186547),
                Complex64::new(0.7071067811865464, -0.7071067811865487),
                Complex64::new(-0.707106781186549, -0.707106781186546),
            ],
        );
        assert_eq!(vandermonde, vandermonde_expected);
    }

    #[test]
    fn create_sigma_r_basis() {
        let sigma_r = Encoder::create_sigma_r_basis(
            Complex64::new(0.7071067811865476, 0.7071067811865475),
            NUM_ELEMENTS / 2, // n is (m / 2)
        );

        // The expected value is the just a transposition of the vandermonde
        let sigma_r_expected = DMatrix::from_vec(
            NUM_ROWS,
            NUM_ROWS, // Num of cols is the same as rows
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.7071067811865476, 0.7071067811865475),
                Complex64::new(0.0000000000000002220446049250313, 1.0),
                Complex64::new(-0.7071067811865474, 0.7071067811865477),
                Complex64::new(1.0, 0.0),
                Complex64::new(-0.7071067811865474, 0.7071067811865477),
                Complex64::new(-0.0000000000000004440892098500626, -1.0),
                Complex64::new(0.707106781186548, 0.707106781186547),
                Complex64::new(1.0, 0.0),
                Complex64::new(-0.7071067811865479, -0.7071067811865471),
                Complex64::new(0.0000000000000011102230246251565, 1.0),
                Complex64::new(0.7071067811865464, -0.7071067811865487),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.707106781186547, -0.707106781186548),
                Complex64::new(-0.0000000000000013877787807814457, -1.0),
                Complex64::new(-0.707106781186549, -0.707106781186546),
            ],
        );

        assert_eq!(sigma_r, sigma_r_expected);
    }

    #[test]
    fn compute_basis_coordinates() {
        let vect = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(192.0, 256.0),
                Complex64::new(128.0, -64.0),
                Complex64::new(128.0, 64.0),
                Complex64::new(192.0, -256.0),
            ],
        );
        let basis_expected: DMatrix<f64> = DMatrix::from_vec(
            NUM_COLS, // Notice the dimensions have changed!
            NUM_ROWS,
            vec![
                160.0,
                90.5096679918781,
                159.99999999999997,
                45.25483399593886,
            ],
        );
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let basis = encoder.compute_basis_coordinates(&vect);
        assert_eq!(basis, basis_expected);
    }

    #[test]
    fn round_coordinates() {
        let coordinates: DMatrix<f64> = DMatrix::from_vec(
            NUM_COLS,
            NUM_ROWS,
            vec![
                160.0,
                90.5096679918781,
                159.99999999999997,
                45.25483399593886,
            ],
        );
        let rounded_expected: DMatrix<f64> = DMatrix::from_vec(
            NUM_COLS,
            NUM_ROWS,
            vec![
                0.0,
                0.5096679918781035,
                0.9999999999999716,
                0.2548339959388599,
            ],
        );
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let rounded = encoder.round_coordinates(&coordinates);
        assert_eq!(rounded, rounded_expected);
    }

    #[test]
    fn coordinate_wise_random_rounding() {
        let coordinates: DMatrix<f64> = DMatrix::from_vec(
            NUM_COLS,
            NUM_ROWS,
            vec![
                160.0,
                90.5096679918781,
                159.99999999999997,
                45.25483399593886,
            ],
        );
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let rounded_coordinates = encoder.coordinate_wise_random_rounding(&coordinates);
        assert_eq!(rounded_coordinates.len(), coordinates.len());
    }

    #[test]
    fn sigma_r_discretization() {
        let vect = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(192.0, 256.0),
                Complex64::new(128.0, -64.0),
                Complex64::new(128.0, 64.0),
                Complex64::new(192.0, -256.0),
            ],
        );

        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let sig_r_disc = encoder.sigma_r_discretization(&vect);
        assert_eq!(sig_r_disc.len(), vect.len());
    }

    #[test]
    fn encode() {
        let plain = DMatrix::from_vec(
            NUM_ROWS / 2,
            NUM_COLS,
            vec![Complex64::new(3.0, 4.0), Complex64::new(2.0, -1.0)],
        );

        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let encoded = encoder.encode(&plain);
        assert_eq!(encoded.len(), 4);
    }

    #[test]
    fn decode() {
        let encoded = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(160.0, 0.00000000000002842170943040401),
                Complex64::new(91.00000000000001, 0.000000000000007105427357601003),
                Complex64::new(160.0, -0.0000000000000026645352591003745),
                Complex64::new(45.0, 0.0),
            ],
        );

        let decode_expected = DMatrix::from_vec(
            NUM_ROWS / 2,
            NUM_COLS,
            vec![
                Complex64::new(3.0082329989778316, 4.002601910021413),
                Complex64::new(1.991767001022168, -0.9973980899785859),
            ],
        );
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let decoded = encoder.decode(&encoded);
        assert_eq!(decoded, decode_expected);
    }

    #[test]
    fn encode_then_decode() {
        let plain = DMatrix::from_vec(
            NUM_ROWS / 2,
            NUM_COLS,
            vec![Complex64::new(3.0, 4.0), Complex64::new(2.0, -1.0)],
        );

        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let encoded = encoder.encode(&plain);
        let decoded = encoder.decode(&encoded);
        assert_eq!(decoded.len(), 2);
    }

    #[test]
    fn to_from_polynomial() {
        let plain = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let poly = encoder.to_polynomial(&plain);
        let plain_expected = encoder.from_polynomial(&poly);
        assert_eq!(plain, plain_expected);
    }

    #[test]
    fn sigma_inverse() {
        // a matrix with dimensions 1 cols × 4 rows.
        let plain = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );

        let sigma_inv_expected = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(2.5, 0.0000000000000005551115123125783),
                Complex64::new(-0.00000000000000022204460492503136, 0.7071067811865479),
                Complex64::new(-0.000000000000000513478148889135, 0.5000000000000002),
                Complex64::new(-0.0000000000000008326672684688678, 0.7071067811865474),
            ],
        );

        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let sigma_inv = encoder.sigma_inverse(&plain);
        assert_eq!(sigma_inv, sigma_inv_expected);
    }

    #[test]
    fn sigma() {
        let original_matrix = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let encoded_matrix = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(2.5, 0.0000000000000005551115123125783),
                Complex64::new(-0.00000000000000022204460492503136, 0.7071067811865479),
                Complex64::new(-0.000000000000000513478148889135, 0.5000000000000002),
                Complex64::new(-0.0000000000000008326672684688678, 0.7071067811865474),
            ],
        );

        let sigma_expected = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, -0.00000000000000011102230246251565),
                Complex64::new(2.0, 0.00000000000000016653345369377348),
                Complex64::new(3.0, -0.00000000000000013877787807814457),
                Complex64::new(4.0, 0.00000000000000011102230246251565),
            ],
        );

        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let sigma = encoder.sigma(&encoded_matrix);
        assert_eq!(sigma, sigma_expected);
        let diff = sigma - original_matrix;
        let normalized = diff.dot(&diff).sqrt();
        assert_eq!(
            normalized,
            Complex64::new(0.0, 0.00000000000000026766507790745728)
        )
    }
}
