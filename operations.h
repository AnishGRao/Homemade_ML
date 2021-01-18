#include "containers.h"
RowVector MMULT(RowVector * A, RowVector * B) {
    ;
}
RowVector MMULT(RowVector * A, Matrix * B) {
    RowVector ret(A->data.size());
    ret.set_zero();
    //iter over cols
    for (int i = 0; i < A->data.size(); i++) {
        int k = 0;
        //iter over rows
        for (auto row : (B->data)) {
            ret.data[i] += row[i] * A->data[k++];
        }
    }
    return ret;
}

RowVector MMULT(Matrix * A, RowVector * B) {
    ;
}

RowVector MMULT(Matrix * A, Matrix * B) {
    ;
}


double DPROD(RowVector * A, RowVector * B) {
    double ret = 0;
    for (auto a = A->data.begin(), b = B->data.begin(); a != A->data.end() && b != B->data.end(); a++, b++)
        ret += (*a) * (*b);
    return ret;
}