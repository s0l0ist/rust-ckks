import pprint
import numpy as np
from numpy.polynomial import Polynomial

pp = pprint.PrettyPrinter(indent=4)


class CKKSEncoder:
    """Basic CKKS encoder to encode complex vectors into polynomials."""

    def __init__(self, M: int):
        """Initialization of the encoder for M a power of 2. 

        xi, which is an M-th root of unity will, be used as a basis for our computations.
        """
        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M

    @staticmethod
    def vandermonde(xi: np.complex128, M: int) -> np.array:
        """Computes the Vandermonde matrix from a m-th root of unity."""

        N = M // 2
        matrix = []
        # We will generate each row of the matrix
        for i in range(N):
            # For each row we select a different root
            root = xi ** (2 * i + 1)
            row = []

            # Then we store its powers
            for j in range(N):
                row.append(root ** j)
            matrix.append(row)
        return matrix

    def sigma_inverse(self, b: np.array) -> Polynomial:
        """Encodes the vector b in a polynomial using an M-th root of unity."""

        # First we create the Vandermonde matrix
        A = CKKSEncoder.vandermonde(self.xi, self.M)

        # Then we solve the system
        coeffs = np.linalg.solve(A, b)

        # Finally we output the polynomial
        p = Polynomial(coeffs)
        return p

    def sigma(self, p: Polynomial) -> np.array:
        """Decodes a polynomial by applying it to the M-th roots of unity."""

        outputs = []
        N = self.M // 2

        # We simply apply the polynomial on the roots
        for i in range(N):
            root = self.xi ** (2 * i + 1)
            output = p(root)
            print("root ", root)
            outputs.append(output)
        return np.array(outputs)


def main():
    M = 8
    encoder = CKKSEncoder(M)
    b = np.array([1, 2, 3, 4])
    p = encoder.sigma_inverse(b)
    print("p = ", p)
    d = encoder.sigma(p)
    print("d = ", d)


if __name__ == "__main__":
    # execute only if run as a script
    main()
