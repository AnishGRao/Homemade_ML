#include "../Data_Loaders.h"

class RNN {
    std::vector<char> data, chars;
    std::unordered_map<char, int> char_to_num;
    std::unordered_map<int, char> num_to_char;
    int data_size, vocab_size, hidden_size, seq_length;
    float lr;
    Matrix * weight_x_hidden;
    Matrix * weight_y_hidden;
    Matrix * weight_hidden_time;
    Matrix * bias_hidden;
    Matrix * bias_y;
    RNN() {
        ;
    }

    void set_consts() {
        //sizes
        data_size = data.size();
        vocab_size = chars.size();

        //hyperparameters
        hidden_size = 100;
        seq_length = 25;
        lr = .1;
    }
    void create_maps() {
        for (int i = 0; i < vocab_size; i++)
            char_to_num[chars[i]] = i, num_to_char[i] = chars[i];
    }
    void model_parameters() {
        weight_x_hidden = new Matrix(hidden_size, vocab_size);
        weight_x_hidden->setRandom();
        SMUL(weight_x_hidden, .01);

        weight_hidden_time = new Matrix(hidden_size, hidden_size);
        weight_hidden_time->setRandom();
        SMUL(weight_hidden_time, .01);

        weight_y_hidden = new Matrix(vocab_size, hidden_size);
        weight_y_hidden->setRandom();
        SMUL(weight_y_hidden, .01);

        bias_hidden = new Matrix(hidden_size, 1);
        bias_hidden->setZero({ 0,-1 });

        bias_y = new Matrix(vocab_size, 1);
        bias_y->setZero({ 0,-1 });
    }

    void loss(std::vector <int> & inputs, std::vector<int> & targets, Matrix & hprev) {
        std::vector<Matrix> xs = {}, hs = {}, ys = {}, ps = {};
        hs.push_back(hprev);

        double loss = 0;
        for (int t = 0; t < inputs.size(); t++) {
            xs.push_back(Matrix(vocab_size, 1));
            xs.back().setZero();
            xs.back().data[inputs[t]][0] = 1;
            //hidden state
            hs.push_back(
                MAPPLY(
                    MSUM(
                        MSUM(
                            std::get<Matrix>(
                                DPROD(weight_x_hidden, xs[t])
                                ),
                            std::get<Matrix>(
                                DPROD(weight_hidden_time, (t == 0 ? hs.back() : hs[t - 1]))
                                )
                        )
                        , bias_hidden),
                    &TANH
                )
            );
            ys[t] = std::get<Matrix>(DPROD(weight_y_hidden, hs[t]));
            //ps[t] =

        }

    }
    void init_RNN() {
        set_consts();
        create_maps();
        model_parameters();
    }
};


void initialize_RNN() {

}