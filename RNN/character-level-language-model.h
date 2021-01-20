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
        std::unordered_map<int, Matrix> xs = {}, hs = {}, ys = {}, ps = {};
        hs[-1] = (hprev);
        double loss = 0, temp_val;
        for (int t = 0; t < inputs.size(); t++) {
            xs[t] = Matrix(vocab_size, 1);
            xs[t].setZero();
            xs[t].data[inputs[t]][0] = 1;
            //hidden state
            hs[t] = (
                MAPPLY(
                    MSUM(
                        MSUM(
                            std::get<Matrix>(DPROD(weight_x_hidden, xs[t])),
                            std::get<Matrix>(DPROD(weight_hidden_time, hs[t - 1]))
                        )
                        , bias_hidden),
                    &TANH
                )
            );
            ys[t] = std::get<Matrix>(DPROD(weight_y_hidden, hs[t]));
            ps[t] = SMUL(MAPPLY(ys[t], &EXP), 1 / (MTOTSUM(MAPPLY(ys[t], &EXP))));
            //TODO check if nat log
            loss += -log(ps[t].data[targets[t]][0]);
        }
        //wow more stuff, how fun.
        Matrix * dweightxh = new Matrix(weight_x_hidden), * dweightyh = new Matrix(weight_y_hidden),
            * dweighthh = new Matrix(weight_hidden_time), * dbiash = new Matrix(bias_hidden),
            * dbiasy = new Matrix(bias_y), * dhnext = new Matrix(hs[0]);

        dweighthh->setZero(), dweightxh->setZero(), dweightyh->setZero();
        dbiash->setZero(), dbiasy->setZero();
        dhnext->setZero();

        for (int t = inputs.size() - 1; t >= 0; t--) {
            auto temp_dy = new Matrix(ps[t]);
            //add -1 to all points in ps[t][targets[t]]'s copy
            MSROWOP(&temp_dy, targets[t], 1, -1);
            *dweightyh = MSUM(*dweightyh, std::get<Matrix>(DPROD(dweightyh, hs[t].transpose())));
            *dbiasy = MSUM(*dbiasy, *temp_dy);
            auto temp_dh = new Matrix(MSUM(std::get<Matrix>(DPROD(dweightyh->transpose(), *temp_dy)), dhnext));
            auto temp_dhraw = new Matrix(

            );
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