use crate::ckks::random::UniformRandomGenerator;
use crate::na::ComplexField;
use na::DMatrix;
use num_complex::Complex64;
use rustnomial::{Evaluable, Polynomial};
use std::cell::RefCell;
use std::convert::TryInto;

// Basic CKKS encoder to encode complex vectors into polynomials.
pub struct Encoder {
    pub xi: Complex64,
    pub m: usize,
    pub n: usize,
    pub sigma_r_basis: DMatrix<Complex64>,
    pub scale: f64,
    pub rng: RefCell<UniformRandomGenerator>,
}

impl Encoder {
    // Initialization of the encoder for M, a power of 2.
    //
    // xi, which is an m-th root of unity will, be used as a basis for our computations
    pub fn new(m: usize, scale: f64) -> Self {
        // xi = e^(2 * pi * i / m)
        let xi = (2.0 * std::f64::consts::PI * Complex64::new(0.0, 1.0) / (m as f64)).exp();
        let n = m / 2;
        let sigma_r_basis = Encoder::create_sigma_r_basis(xi, n);

        // also hold a ref to a mutable RNG
        let rng = RefCell::new(UniformRandomGenerator::new());

        Self {
            xi,
            m,
            n,
            sigma_r_basis,
            scale,
            rng,
        }
    }

    // Projects a vector of H into C^{N/2}.
    pub fn pi(&self, z: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let mut z_slice: Vec<Complex64> = Vec::with_capacity(self.n);
        for coeff in z.iter() {
            z_slice.push(coeff.clone());
        }
        let n = self.m / 4;
        let dmatrix = DMatrix::from_row_slice(n, 1, &z_slice[..n]);
        dmatrix
    }

    // Expands a vector of C^{N/2} by expanding it with its complex conjugate.
    pub fn pi_inverse(&self, z: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let mut z_concat: Vec<Complex64> = Vec::with_capacity(self.n * 2);

        // TODO: Simplify the concatenation
        // let z_conjugate = z.conjugate();
        let mut z_conjugate: Vec<Complex64> = Vec::with_capacity(self.n);
        for coeff in z.iter() {
            z_concat.push(coeff.clone());
            z_conjugate.push(coeff.conjugate());
        }
        z_concat.append(&mut z_conjugate);
        let dmatrix = DMatrix::from_row_slice(self.n * 2, 1, &z_concat);
        dmatrix
    }

    // Creates the basis (sigma(1), sigma(X), ..., sigma(X** N-1)).
    pub fn create_sigma_r_basis(xi: Complex64, n: usize) -> DMatrix<Complex64> {
        Encoder::vandermonde(xi, n).transpose()
    }

    // Computes the coordinates of a vector with respect to the orthogonal lattice basis.
    pub fn compute_basis_coordinates(&self, z: &DMatrix<Complex64>) -> DMatrix<f64> {
        // output = np.array([np.real(np.vdot(z, b) / np.vdot(b,b)) for b in self.sigma_R_basis])
        let mut output: Vec<f64> = vec![];

        for b in self.sigma_r_basis.row_iter() {
            let ans = z.conjugate().dot(&b) / b.conjugate().dot(&b);
            let real = ans.real();
            output.push(real);
        }

        let dmatrix = DMatrix::from_row_slice(1, output.len(), &output);
        dmatrix
    }

    // Gives the integral rest.
    pub fn round_coordinates(&self, coordinates: &DMatrix<f64>) -> DMatrix<f64> {
        let mut output: Vec<f64> = vec![];
        for coeff in coordinates.iter() {
            let temp = coeff - coeff.floor();
            output.push(temp)
        }
        let dmatrix = DMatrix::from_row_slice(1, output.len(), &output);
        dmatrix
    }

    // Rounds coordinates randonmly.
    pub fn coordinate_wise_random_rounding(&self, coordinates: &DMatrix<f64>) -> DMatrix<i64> {
        let mut output: Vec<f64> = vec![];
        let r = self.round_coordinates(coordinates);
        let mut rng_ref = self.rng.borrow_mut();
        for &c in r.iter() {
            let choices = vec![c, c - 1.0];
            let wieghts = vec![1.0 - c, c];
            let sample = rng_ref.weighted_choice(&choices, &wieghts);
            output.push(sample);
        }

        let dmatrix = DMatrix::from_row_slice(1, output.len(), &output);
        let rounded_coordinates = coordinates - dmatrix;
        let new_rounded_coordinates = rounded_coordinates.map(|x| x as i64);
        new_rounded_coordinates
    }

    // Computes the Vandermonde matrix from a m-th root of unity.
    pub fn vandermonde(xi: Complex64, n: usize) -> DMatrix<Complex64> {
        // We will generate a flat Vector containing all elements for
        // a matrix
        let mut matrix: Vec<Complex64> = Vec::with_capacity(n);
        for i in 0..n {
            let i: u32 = i.try_into().expect("Couldn't convert usize to u32");
            // For each row we select a different root
            let root: Complex64 = xi.powu((2 * i) + 1);

            // Then we store its powers
            for j in 0..n {
                let j: u32 = j.try_into().expect("Couldn't convert usize to u32");
                let ans = root.powu(j);

                // Push into a flat 1D vector that will be transformed into a matrix later
                matrix.push(ans);
            }
        }
        // Create dynamic matrix from our native matrix (row-major order)
        let dmatrix = DMatrix::from_row_slice(n, n, &matrix);
        dmatrix
    }

    // TODO: Encodes a vector by expanding it first to H,
    // scale it, project it on the lattice of sigma(R), and performs
    // sigma inverse.
    // pub fn encode(&self, z: &DMatrix<Complex64>) -> Polynomial<Complex64> {
    //     let pi_z = self.pi_inverse(z);
    //     let scaled_pi_z = self.scale * pi_z;
    //     let rounded_scale_pi_z = self.sigma_r_discretization(scaled_pi_z);
    //     let p = self.sigma_inverse(rounded_scale_pi_z);

    //     // We round it afterwards due to numerical imprecision
    //     let coeff = p.coeff.real() as i64; // auto rounds
    //     let poly = self.to_polynomial(coeff);
    //     poly
    // }

    // // TODO: Decodes a polynomial by removing the scale,
    // // evaluating on the roots, and project it on C^(N/2)
    // pub fn decode(&self, p: &Polynomial<Complex64>) -> DMatrix<Complex64> {
    //     let rescaled_p = p / self.scale;
    //     let z = self.sigma(rescaled_p);
    //     let pi_z = self.pi(&z);
    //     pi_z
    // }

    // sigma-inverse is a vector, b, in a polynomial using an m-th root of unity
    pub fn sigma_inverse(&self, b: &DMatrix<Complex64>) -> Polynomial<Complex64> {
        // First we create the Vandermonde matrix
        let a = Encoder::vandermonde(self.xi, self.n);
        // Then we solve the system and return the resultant matrix
        let decomp = a.lu();
        let x_coeffs = decomp.solve(b).expect("Linear resolution failed.");
        Encoder::to_polynomial(&self, &x_coeffs)
    }

    // sigma a polynomial by applying it to the M-th roots of unity.
    pub fn sigma(&self, poly: &Polynomial<Complex64>) -> DMatrix<Complex64> {
        // We simply apply the polynomial on the roots
        let mut matrix: Vec<Complex64> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let i: u32 = i.try_into().expect("Couldn't convert usize to u32");
            let root: Complex64 = self.xi.powu((2 * i) + 1);
            // Evaluate polynomial with the given root
            let result = poly.eval(root);
            matrix.push(result)
        }

        // We will always return n cols x 1 row matrix;
        let dmatrix = DMatrix::from_row_slice(self.n, 1, &matrix);
        dmatrix
    }

    // Converts a matrix into a polynomial
    pub fn to_polynomial(&self, x_coeffs: &DMatrix<Complex64>) -> Polynomial<Complex64> {
        // TODO: Figure out a way to collect these elements idomatically
        // calling .collect() refuses to work.
        let mut poly_vec: Vec<Complex64> = Vec::with_capacity(self.n);
        for coeff in x_coeffs.iter() {
            poly_vec.push(coeff.clone());
        }
        // Reverse the vec because the 'polynomials' library constructs
        // a polynomial with the highest power to lowest power ex: 3x^3 + 2x^2 + x + 8
        // and we need it in the form of 8 + x + 2x^2 + 3x^3
        poly_vec.reverse();
        let poly = Polynomial::new(poly_vec);
        poly
    }

    // Converts a polynomial into a matrix
    pub fn from_polynomial(&self, poly: &Polynomial<Complex64>) -> DMatrix<Complex64> {
        let mut matrix: Vec<Complex64> = Vec::with_capacity(self.n);
        for coeff in poly.terms.iter() {
            matrix.push(coeff.clone());
        }

        // Reverse the vec because the 'polynomials' library constructs
        // a polynomial with the highest power to lowest power ex: 3x^3 + 2x^2 + x + 8
        // and we need it in the form of 8 + x + 2x^2 + 3x^3
        matrix.reverse();

        let dmatrix = DMatrix::from_row_slice(self.n, 1, &matrix);
        dmatrix
    }
}

#[cfg(test)]
mod complex {
    use super::*;

    const NUM_ELEMENTS: usize = 8;
    const NUM_ROWS: usize = NUM_ELEMENTS / 2;
    const NUM_COLS: usize = 1;
    const SCALE: f64 = 20.0;
    #[test]
    fn test_xi() {
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
                Complex64::new(1.0, -0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(3.0, 5.0),
                Complex64::new(4.0, -0.0),
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

        let vandermonde_expected = DMatrix::from_vec(
            NUM_ROWS,
            NUM_ROWS, // Num of cols is the same as rows
            vec![
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
            NUM_COLS,
            NUM_ROWS,
            vec![
                Complex64::new(192.0, 256.0),
                Complex64::new(128.0, -64.0),
                Complex64::new(128.0, 64.0),
                Complex64::new(192.0, -256.0),
            ],
        );
        let basis_expected: DMatrix<f64> = DMatrix::from_vec(
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
        let rounded_coordinates_expected: DMatrix<i64> =
            DMatrix::from_vec(NUM_COLS, NUM_ROWS, vec![160, 90, 160, 45]);
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);
        let rounded_coordinates = encoder.coordinate_wise_random_rounding(&coordinates);
        assert_eq!(
            rounded_coordinates.len(),
            rounded_coordinates_expected.len()
        );
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
    fn encode() {
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

        let encoded_expected = DMatrix::from_vec(
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
        let encoded_matrix = encoder.from_polynomial(&sigma_inv);
        assert_eq!(encoded_matrix, encoded_expected);
    }

    #[test]
    fn decode() {
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

        let encoded_expected = DMatrix::from_vec(
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
        let encoded_poly = encoder.to_polynomial(&encoded_matrix);
        let sigma = encoder.sigma(&encoded_poly);
        assert_eq!(sigma, encoded_expected);
        let diff = sigma.clone() - original_matrix.clone();
        let normalized = diff.dot(&diff).sqrt();
        assert_eq!(
            normalized,
            Complex64::new(0.0, 0.00000000000000026766507790745728)
        )
    }

    #[test]
    fn add_two() {
        let m1 = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let m2 = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(-4.0, 0.0),
            ],
        );

        let add_expected = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(2.0, -0.00000000000000012449156634005058),
                Complex64::new(
                    -0.0000000000000004440892098500626,
                    0.00000000000000033306690738754696,
                ),
                Complex64::new(6.0, -0.00000000000000033306690738754696),
                Complex64::new(0.0000000000000008881784197001252, 0.0),
            ],
        );
        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

        let p1 = encoder.sigma_inverse(&m1);
        let p2 = encoder.sigma_inverse(&m2);
        let p_add = p1 + p2;

        let add_decoded = encoder.sigma(&p_add);
        assert_eq!(add_decoded, add_expected);
    }

    #[test]
    fn multiply() {
        let m1 = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let m2 = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(-4.0, 0.0),
            ],
        );
        let expected = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0000000000000013, 0.0),
                Complex64::new(-4.0, -0.0000000000000004440892098500626),
                Complex64::new(8.999999999999996, -0.0000000000000002220446049250313),
                Complex64::new(-16.000000000000004, -0.00000000000001176836406102666),
            ],
        );

        let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

        let p1 = encoder.sigma_inverse(&m1);
        let p2 = encoder.sigma_inverse(&m2);
        let plain_modulus = Polynomial::from(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]);

        let p_product = p1 * p2;
        let (_, p_mod) = p_product.div_mod(&plain_modulus);
        let sigma = encoder.sigma(&p_mod);
        assert_eq!(sigma, expected);
    }

    // #[test]
    // fn multiply_integer_polynomials() {
    //     let m1 = DMatrix::from_vec(
    //         NUM_ROWS,
    //         NUM_COLS,
    //         vec![
    //             Complex64::new(1.0, 0.0),
    //             Complex64::new(2.0, 0.0),
    //             Complex64::new(3.0, 0.0),
    //             Complex64::new(4.0, 0.0),
    //         ],
    //     );
    //     let coordinates = DMatrix::from_vec(
    //         NUM_ROWS,
    //         NUM_COLS,
    //         vec![
    //             Complex64::new(1.0, 0.0),
    //             Complex64::new(1.0, 0.0),
    //             Complex64::new(1.0, 0.0),
    //             Complex64::new(1.0, 0.0),
    //         ],
    //     );

    //     let encoder = Encoder::new(NUM_ELEMENTS, SCALE);

    //     let sigma = encoder.sigma_r_basis.transpose();
    //     let product = sigma * coordinates;
    //     println!("encoder.encode(&product) {:#?}", encoder.encode(&product));
    //     println!("product {}", product);
    //     assert_eq!(true, false);
    // }
}
