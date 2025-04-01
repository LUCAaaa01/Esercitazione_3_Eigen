#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    // Soluzione attesa: x = [-1, -1]
    Vector2d x;
    x << -1.0e+0, -1.0e+0;

    //----------------- Punto 1 -----------------
    cout << "Punto 1:" << endl;
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    cout << "A:" << endl << A1 << endl;

    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    cout << "b:" << endl << b1 << endl;

    Vector2d sol1, sol2;
    // Metodo PA=LU
    FullPivLU<Matrix2d> lu1(A1);
    sol1 = lu1.solve(b1);
    cout << "La soluzione della fattorizzazione PA=LU è: " << sol1.transpose() << endl;

    // Metodo QR
    FullPivHouseholderQR<Matrix2d> qr1(A1);
    sol2 = qr1.solve(b1);
    cout << "La soluzione della fattorizzazione QR è: " << sol2.transpose() << endl;

    // Calcolo errore relativo
    Vector2d err1, err2;
    err1 = ((sol1 - x).cwiseAbs().array() / sol1.cwiseAbs().array()).matrix();
    cout << "L'errore relativo della fattorizzazione PA=LU è: " << err1.transpose() << endl;
    err2 = ((sol2 - x).cwiseAbs().array() / sol2.cwiseAbs().array()).matrix();
    cout << "L'errore relativo della fattorizzazione QR è: " << err2.transpose() << endl;

    //----------------- Punto 2 -----------------
    cout << "\nPunto 2:" << endl;
    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    cout << "A:" << endl << A2 << endl;

    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    cout << "b:" << endl << b2 << endl;

    // Metodo PA=LU
    FullPivLU<Matrix2d> lu2(A2);
    sol1 = lu2.solve(b2);
    cout << "La soluzione della fattorizzazione PA=LU è: " << sol1.transpose() << endl;

    // Metodo QR
    FullPivHouseholderQR<Matrix2d> qr2(A2);
    sol2 = qr2.solve(b2);
    cout << "La soluzione della fattorizzazione QR è: " << sol2.transpose() << endl;

    // Calcolo errore relativo
    err1 = ((sol1 - x).cwiseAbs().array() / sol1.cwiseAbs().array()).matrix();
    cout << "L'errore relativo della fattorizzazione PA=LU è: " << err1.transpose() << endl;
    err2 = ((sol2 - x).cwiseAbs().array() / sol2.cwiseAbs().array()).matrix();
    cout << "L'errore relativo della fattorizzazione QR è: " << err2.transpose() << endl;

    //----------------- Punto 3 -----------------
    cout << "\nPunto 3:" << endl;
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    cout << "A:" << endl << A3 << endl;

    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    cout << "b:" << endl << b3 << endl;

    // Metodo PA=LU
    FullPivLU<Matrix2d> lu3(A3);
    sol1 = lu3.solve(b3);
    cout << "La soluzione della fattorizzazione PA=LU è: " << sol1.transpose() << endl;

    // Metodo QR
    FullPivHouseholderQR<Matrix2d> qr3(A3);
    sol2 = qr3.solve(b3);
    cout << "La soluzione della fattorizzazione QR è: " << sol2.transpose() << endl;

    // Calcolo errore relativo
    err1 = ((sol1 - x).cwiseAbs().array() / sol1.cwiseAbs().array()).matrix();
    cout << "L'errore relativo della fattorizzazione PA=LU è: " << err1.transpose() << endl;
    err2 = ((sol2 - x).cwiseAbs().array() / sol2.cwiseAbs().array()).matrix();
    cout << "L'errore relativo della fattorizzazione QR è: " << err2.transpose() << endl;

    return 0;
}
