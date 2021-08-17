use na::DMatrix;
use num_complex::Complex64;
use rustnomial::{Evaluable, Polynomial};
use std::convert::TryInto;

// Basic CKKS encoder to encode complex vectors into polynomials.
pub struct Encoder {
    pub xi: Complex64,
    pub m: usize,
}

impl Encoder {
    // Initialization of the encoder for M, a power of 2.
    //
    // xi, which is an m-th root of unity will, be used as a basis for our computations
    fn new(m: usize) -> Self {
        // xi = e^(2 * pi * i / m)
        let xi = (2.0 * std::f64::consts::PI * Complex64::new(0.0, 1.0) / (m as f64)).exp();
        Self { xi, m }
    }

    // Computes the Vandermonde matrix from a m-th root of unity.
    pub fn vandermonde(xi: Complex64, m: usize) -> DMatrix<Complex64> {
        // Floor division
        let n = m / 2;

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
    // Encodes a vector, b, in a polynomial using an m-th root of unity (sigma-inverse)
    pub fn encode(&self, b: &DMatrix<Complex64>) -> Polynomial<Complex64> {
        // First we create the Vandermonde matrix
        let a = Encoder::vandermonde(self.xi, self.m);
        // Then we solve the system and return the resultant matrix
        let decomp = a.lu();
        let x_coeffs = decomp.solve(b).expect("Linear resolution failed.");
        Encoder::to_polynomial(&self, &x_coeffs)
    }
    // Decodes a polynomial by applying it to the M-th roots of unity. (sigma)
    pub fn decode(&self, poly: &Polynomial<Complex64>) -> DMatrix<Complex64> {
        let n = self.m / 2;

        // We simply apply the polynomial on the roots
        let mut matrix: Vec<Complex64> = Vec::with_capacity(n);
        for i in 0..n {
            let i: u32 = i.try_into().expect("Couldn't convert usize to u32");
            let root: Complex64 = self.xi.powu((2 * i) + 1);
            // Evaluate polynomial with the given root
            let result = poly.eval(root);
            matrix.push(result)
        }

        // We will always return n cols x 1 row matrix;
        let dmatrix = DMatrix::from_row_slice(n, 1, &matrix);
        dmatrix
    }

    // Converts a matrix into a polynomial
    fn to_polynomial(&self, x_coeffs: &DMatrix<Complex64>) -> Polynomial<Complex64> {
        let n = self.m / 2;

        // TODO: Figure out a way to collect these elements idomatically
        // calling .collect() refuses to work.
        let mut poly_vec: Vec<Complex64> = Vec::with_capacity(n);
        for coeff in x_coeffs.iter() {
            poly_vec.push(*coeff);
        }
        // Reverse the vec because Polynomial library constructs
        // a polynomial with the highest power to lowest power ex: 3x^3 + 2x^2 + x + 8
        // and we need it in the form of 8 + x + 2x^2 + 3x^3
        poly_vec.reverse();
        let poly = Polynomial::new(poly_vec);
        poly
    }
    // Converts a polynomial into a matrix
    fn from_polynomial(&self, poly: &Polynomial<Complex64>) -> DMatrix<Complex64> {
        let n = self.m / 2;
        let mut matrix: Vec<Complex64> = Vec::with_capacity(n);
        for coeff in poly.terms.iter() {
            matrix.push(*coeff);
        }

        println!("from_poly matrix {:#?}", matrix);

        // Reverse the vec because Polynomial library constructs
        // a polynomial with the highest power to lowest power ex: 3x^3 + 2x^2 + x + 8
        // and we need it in the form of 8 + x + 2x^2 + 3x^3
        matrix.reverse();

        let dmatrix = DMatrix::from_row_slice(n, 1, &matrix);
        dmatrix
    }
}

#[cfg(test)]
mod complex {
    use super::*;
    #[test]
    pub fn test_xi() {
        let encoder = Encoder::new(8);
        assert_eq!(
            encoder.xi,
            Complex64::new(0.7071067811865476, 0.7071067811865475)
        );
    }

    #[test]
    pub fn test_sigma_inverse() {
        const NUM_ELEMENTS: usize = 8;
        const NUM_ROWS: usize = 4;
        const NUM_COLS: usize = 1;

        // a matrix with dimensions 1 cols × 4 rows.
        let b = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );

        let expected = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(2.5, 0.0000000000000005551115123125783),
                Complex64::new(-0.00000000000000022204460492503136, 0.7071067811865479),
                Complex64::new(-0.000000000000000513478148889135, 0.5000000000000002),
                Complex64::new(-0.0000000000000008326672684688678, 0.7071067811865474),
            ],
        );

        let encoder = Encoder::new(NUM_ELEMENTS);
        let p = encoder.encode(&b);
        let p_matrix = encoder.from_polynomial(&p);
        assert_eq!(p_matrix, expected);
    }

    #[test]
    pub fn test_sigma() {
        const NUM_ELEMENTS: usize = 8;
        const NUM_ROWS: usize = 4;
        const NUM_COLS: usize = 1;

        let original = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        );
        let sigma_inverse = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(2.5, 0.0000000000000005551115123125783),
                Complex64::new(-0.00000000000000022204460492503136, 0.7071067811865479),
                Complex64::new(-0.000000000000000513478148889135, 0.5000000000000002),
                Complex64::new(-0.0000000000000008326672684688678, 0.7071067811865474),
            ],
        );

        let b_expected = DMatrix::from_vec(
            NUM_ROWS,
            NUM_COLS,
            vec![
                Complex64::new(1.0, -0.00000000000000011102230246251565),
                Complex64::new(2.0, 0.00000000000000016653345369377348),
                Complex64::new(3.0, -0.00000000000000013877787807814457),
                Complex64::new(4.0, 0.00000000000000011102230246251565),
            ],
        );

        let encoder = Encoder::new(NUM_ELEMENTS);
        let sig_poly = encoder.to_polynomial(&sigma_inverse);
        let b_reconstructed = encoder.decode(&sig_poly);
        println!("b_reconstructed {}", b_reconstructed);
        assert_eq!(b_reconstructed, b_expected);
        let diff = b_reconstructed.clone() - original.clone();
        let normalized = diff.dot(&diff).sqrt();
        assert_eq!(
            normalized,
            Complex64::new(0.0, 0.00000000000000026766507790745728)
        )
    }

    #[test]
    fn add_two() {
        const NUM_ELEMENTS: usize = 8;
        const NUM_ROWS: usize = 4;
        const NUM_COLS: usize = 1;

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
        let encoder = Encoder::new(NUM_ELEMENTS);

        let p1 = encoder.encode(&m1);
        let p2 = encoder.encode(&m2);
        let p_add = p1 + p2;

        let add_decoded = encoder.decode(&p_add);
        assert_eq!(add_decoded, add_expected);
    }

    #[test]
    fn multiply() {
        const NUM_ELEMENTS: usize = 8;
        const NUM_ROWS: usize = 4;
        const NUM_COLS: usize = 1;

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

        let encoder = Encoder::new(NUM_ELEMENTS);

        let p1 = encoder.encode(&m1);
        let p2 = encoder.encode(&m2);
        let plain_modulus = Polynomial::from(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]);

        let p_product = p1 * p2;
        let (_, p_mod) = p_product.div_mod(&plain_modulus);
        let decoded = encoder.decode(&p_mod);
        assert_eq!(decoded, expected);
    }
}
