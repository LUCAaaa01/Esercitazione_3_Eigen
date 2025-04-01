#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Funzione per risolvere il sistema e stampare la soluzione e l'errore relativo
void Ris_System(const Matrix2d &A, const Vector2d &b, const Vector2d &x_exact) {
    cout << "A:" << endl << A << endl;
    cout << "b:" << endl << b << endl;

    Vector2d sol1, sol2;
    // Metodo PA=LU
    FullPivLU<Matrix2d> lu(A);
    sol1 = lu.solve(b);
    cout << "La soluzione della fattorizzazione PA=LU è: " << sol1.transpose() << endl;

    // Metodo QR
    FullPivHouseholderQR<Matrix2d> qr(A);
    sol2 = qr.solve(b);
    cout << "La soluzione della fattorizzazione QR è: " << sol2.transpose() << endl;

    // Calcolo errore relativo
    Vector2d err1 = ((sol1 - x_exact).cwiseAbs().array() / x_exact.cwiseAbs().array()).matrix();
    cout << "L'errore relativo della fattorizzazione PA=LU è: " << err1.transpose() << endl;
    Vector2d err2 = ((sol2 - x_exact).cwiseAbs().array() / x_exact.cwiseAbs().array()).matrix();
    cout << "L'errore relativo della fattorizzazione QR è: " << err2.transpose() << endl;
}

int main()
{
    Vector2d x_exact;
    x_exact << -1.0e+0, -1.0e+0;

    cout << "Punto 1:" << endl;
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    Ris_System(A1, b1, x_exact);

    cout << "\nPunto 2:" << endl;
    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    Ris_System(A2, b2, x_exact);

    cout << "\nPunto 3:" << endl;
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    Ris_System(A3, b3, x_exact);

    return 0;
}
