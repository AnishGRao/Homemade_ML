#include "containers.h"

double TANH(double val) {
    return tanh(val);
}

//matrix multiplicaation
RowVector MMULT(RowVector A, Matrix B) {
    RowVector ret(A.data.size());
    ret.set_zero();
    //iter over cols
    for (int i = 0; i < A.data.size(); i++) {
        int k = 0;
        //iter over rows
        for (auto row : (B.data)) {
            ret.data[i] += row[i] * A.data[k++];
        }
    }
    return ret;
}

//matrix subtraction
RowVector MSUB(RowVector A, RowVector B) {
    RowVector ret(A.data.size());
    for (int i = 0; i < A.data.size(); i++) {
        ret.data[i] = A.data[i] - B.data[i];
    }
    return ret;
}

//scalar-matrix multiplication
void SMUL(Matrix * A, double D) {
    for (int i = 0; i < A->data.size(); i++)
        for (int j = 0; j < A->data[0].size(); i++)
            A->data[i][j] *= D;
}

std::variant<Matrix, double> DPROD(Matrix & A, Matrix & B) {
    //regular dprod stuff
    auto ret = Matrix(A.data.size(), B.data[0].size());
    ret.setZero();
    //go through A's rows
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < B.data[0].size(); j++)
            for (int k = 0; k < A.data[0].size(); ++k)
                ret.data[i][j] += A.data[i][k] * B.data[k][j];
    return ret.data.size() == 1 && ret.data[0].size() == 1 ? ret.data[0][0] : ret;
}

std::variant<Matrix, double> DPROD(Matrix * A, Matrix & B) {
    //regular dprod stuff
    auto ret = Matrix(A->data.size(), B.data[0].size());
    ret.setZero();
    //go through A's rows
    for (int i = 0; i < A->data.size(); i++)
        for (int j = 0; j < B.data[0].size(); j++)
            for (int k = 0; k < A->data[0].size(); ++k)
                ret.data[i][j] += A->data[i][k] * B.data[k][j];
    return ret.data.size() == 1 && ret.data[0].size() == 1 ? ret.data[0][0] : ret;
}

Matrix MSUM(Matrix A, Matrix B) {
    auto ret = Matrix(A.data.size(), A.data[0].size());
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < A.data[0].size(); j++)
            ret.data[i][j] = A.data[i][j] + B.data[i][j];
    return ret;
}

Matrix MSUM(Matrix A, Matrix * B) {
    auto ret = Matrix(A.data.size(), A.data[0].size());
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < A.data[0].size(); j++)
            ret.data[i][j] = A.data[i][j] + B->data[i][j];
    return ret;
}

Matrix MAPPLY(Matrix A, double (*func)(double)) {
    auto ret = A;
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < A.data[0].size(); j++)
            ret.data[i][j] = (*func)(ret.data[i][j]);
    return ret;
}

//dot product of two row vectors
double DPROD(RowVector & A, RowVector & B) {
    double ret = 0;
    for (auto a = A.data.begin(), b = B.data.begin(); a != A.data.end() && b != B.data.end(); a++, b++)
        ret += (*a) * (*b);
    return ret;
}