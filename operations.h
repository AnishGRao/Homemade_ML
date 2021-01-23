#include "containers.h"
double TANH(double val) {
    return tanh(val);
}

double EXP(double val) {
    return exp(val);
}

double SQRT(double val) {
    union {
        int i;
        float x;
    } u;
    u.x = val;
    u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
    return u.x;
}

double INVERT(double val) {
    return 1 / val;
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
        for (int j = 0; j < A->data[0].size(); j++)
            A->data[i][j] *= D;
}

Matrix SMUL(Matrix A, double D) {
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < A.data[0].size(); j++)
            A.data[i][j] *= D;
    return A;
}

double SCALARDPROD(Matrix A, Matrix B) {
    auto ret = Matrix(A.data.size(), B.data[0].size());
    ret.setZero();
    //go through A's rows
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < B.data[0].size(); j++)
            for (int k = 0; k < A.data[0].size(); ++k)
                ret.data[i][j] += A.data[i][k] * B.data[k][j];
    return ret.data[0][0];
}

Matrix  MDPROD(Matrix A, Matrix B) {
    auto ret = Matrix(A.data.size(), B.data[0].size());
    ret.setZero();
    //go through A's rows
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < B.data[0].size(); j++)
            for (int k = 0; k < A.data[0].size(); ++k)
                ret.data[i][j] += A.data[i][k] * B.data[k][j];
    return ret;
}
//Finds the sum of all elements in a matrix of any dimensionality
double MTOTSUM(Matrix A) {
    double ret = 0;
    for (auto row : A.data)
        for (auto ele : row)
            ret += ele;
    return ret;
}

Matrix MADD(Matrix A, double d) {
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < A.data[0].size(); j++)
            A.data[i][j] += d;
    return A;
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

Matrix MSOP(Matrix A, double num, int op) {
    if (op == 1) {
        for (int i = 0; i < A.data.size(); i++)
            for (int j = 0; j < A.data[0].size(); j++)
                A.data[i][j] += num;
    }
    return A;
}

//Shape must be same, and brodcasting already applied for dissimilar matrices.
Matrix NAIVEMUL(Matrix A, Matrix B) {
    auto ret = Matrix(A.data.size(), A.data[0].size());
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < A.data[0].size(); j++)
            ret.data[i][j] = A.data[i][j] * B.data[i][j];
    return ret;

}

void MSROWOP(Matrix ** A, int row, int operation, int scalar) {
    switch (operation) {
        //addition
    case 1:
        for (int i = 0; i < (*A)->data[0].size(); i++) {
            (*A)->data[row][i] += scalar;
            break;
        }
        break;
    default: break;
    }
}


void CLIP(Matrix & A, int min, int max) {
    for (int i = 0; i < A.data.size(); i++)
        for (int j = 0; j < A.data[0].size(); j++)
            A.data[i][j] = A.data[i][j] > max ? max : A.data[i][j] < min ? min : A.data[i][j];
}

std::vector<double> FLATTEN(std::vector<double> const & v) {
    return v;
}

std::vector<double> FLATTEN(Matrix A) {
    std::vector<double> ret;
    for (auto const & r : A.data) {
        auto s = FLATTEN(r);
        ret.reserve(ret.size() + s.size());
        ret.insert(ret.end(), s.cbegin(), s.cend());
    }
    return ret;
}

//dot product of two row vectors
double DPROD(RowVector & A, RowVector & B) {
    double ret = 0;
    for (auto a = A.data.begin(), b = B.data.begin(); a != A.data.end() && b != B.data.end(); a++, b++)
        ret += (*a) * (*b);
    return ret;
}