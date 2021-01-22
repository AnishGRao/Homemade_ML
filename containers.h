#include <bits/stdc++.h>
#include <random>
#include <math.h>
#include <cmath>
#include <variant>
#define ANISHDEBUG

//always going to be updated
//right now looking over pytorch elements in order to have strong linalg basis

class RowVector {
public:
    std::vector<double> data;
    RowVector() {
        ;
    }
    RowVector(int size) {
        data.resize(size);
    };
    RowVector(const RowVector & other) {
        this->data = other.data;
    }
    RowVector & operator= (const RowVector & other) {
        if (this->data != other.data) {
            this->data = other.data;
        }
        return *this;
    }

    //simple resize
    void resize(int size = 1) {
        this->data.resize(size);
    }


    //optional resize, and setting all values to a U distribution [-1,1]
    void set_random(int size = -1) {
        std::random_device mch;
        std::default_random_engine generator(mch());
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        if (size != -1)
            this->resize(size);
        for (int i = 0; i < this->data.size(); i++) {
            this->data[i] = distribution(generator);
        }
    }

    void set_block(int sRow, int sCol, int bRow, int bCol, RowVector toSet) {
        if (bRow != 1) {
            std::cerr << "OUT-OF-BOUNDS BLOCK ACCESS: ROWVECTOR HAS 1 ROW, YOU ATTEMPTED TO ACCESS ROW " << bRow << "\n";
            exit(1);
        }
        for (int i = sCol; i < bCol; i++) {
            this->data[i] = toSet.data[i];
        }
    }

    void set_zero() {
        for (int i = 0; i < this->data.size(); i++)
            this->data[i] = 0;
    }
};

class ColVector {
public:
    std::vector<double> data;
};

class Matrix {
public:
    std::vector<std::vector<double>>data;
    struct ele {
        int row = -1;
        int col = -1;
    };
    Matrix(int rows = 1, int cols = 1) {
        data.resize(rows, std::vector<double>(cols, 0));
    }
    Matrix(const Matrix & other) {
        this->data = other.data;
    }
    Matrix(const Matrix * other) {
        this->data = other->data;
    }
    Matrix & operator= (const Matrix & other) {
        if (this->data != other.data) {
            data = other.data;
        }
        return *this;
    }
    Matrix & operator= (const Matrix * other) {
        if (this->data != other->data) {
            data = other->data;
        }
        return *this;
    }

    //template <typename row, typename col>
    void setRandom() {
        //auto cRow = pick<row>(arg0, arg1);
        //auto cCol = pick<col>(arg0, arg1);
        //if (cCol == -1 && cRow == -1) {
        std::random_device mch;
        std::default_random_engine generator(mch());
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        for (int i = 0; i < this->data.size(); i++)
            for (int j = 0; j < this->data[0].size(); j++) {
                this->data[i][j] = distribution(generator);
            }
        //}
        //TODO add in the functionality for setzero for indiv. rows and cols.
    }

    void setZero(ele element) {
        if (element.row != -1) {
            for (int i = 0; i < this->data[element.row].size(); i++) {
                this->data[element.row][i] = 0;
            }
        }
        else {
            for (int i = 0; i < this->data.size(); i++) {
                this->data[i][element.col] = 0;
            }
        }
    }
    void setZero() {
        for (int i = 0; i < data.size(); i++)
            for (int j = 0; j < data[0].size(); j++)
                data[i][j] = 0;
    }
    Matrix transpose() {
        Matrix ret(this->data[0].size(), this->data.size());
        for (int i = 0; i < this->data.size(); ++i)
            for (int j = 0; j < this->data[0].size(); ++j)
                ret.data[j][i] = this->data[i][j];
        return ret;
    }



};

std::ostream & operator<< (std::ostream & out, const RowVector & r) {
    out << "\n\n[ ";
    for (auto i : r.data)
        out << i << " ";
    out << "]\n\n";
    return out;
}
