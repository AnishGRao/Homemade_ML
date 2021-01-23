#include "../Data_Loaders.h"

class RNN {
    std::vector<char> data, chars;
    std::unordered_map<char, int> char_to_num;
    std::unordered_map<int, char> num_to_char;
    int data_size, vocab_size, hidden_size, seq_length;
    double lr;
    Matrix * weight_x_hidden;
    Matrix * weight_y_hidden;
    Matrix * weight_hidden_time;
    Matrix * bias_hidden;
    Matrix * bias_y;
    std::random_device rd;
    std::mt19937 e2;


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
        std::mt19937 e3(this->rd());
        e2 = e3;
        this->weight_x_hidden = new Matrix(hidden_size, vocab_size);
        this->weight_x_hidden->setRandom();
        SMUL(weight_x_hidden, .01);

        this->weight_hidden_time = new Matrix(hidden_size, hidden_size);
        this->weight_hidden_time->setRandom();
        SMUL(weight_hidden_time, .01);

        this->weight_y_hidden = new Matrix(vocab_size, hidden_size);
        this->weight_y_hidden->setRandom();
        SMUL(weight_y_hidden, .01);

        this->bias_hidden = new Matrix(hidden_size, 1);
        this->bias_hidden->setZero();
        this->bias_y = new Matrix(vocab_size, 1);
        this->bias_y->setZero();
    }
    double loss(std::vector <int> & inputs, std::vector<int> & targets, Matrix & hprev,
        Matrix ** dweighthh, Matrix ** dweightxh, Matrix ** dweightyh, Matrix ** dbiasy, Matrix ** dbiash) {
        std::unordered_map<int, Matrix *> xs = {}, hs = {}, ys = {}, ps = {};
        hs[-1] = new Matrix(hprev);
        double loss = 0, temp_val;
        for (int t = 0; t < inputs.size(); t++) {

            xs[t] = new Matrix(vocab_size, 1);
            xs[t]->setZero();
            xs[t]->data[inputs[t]][0] = 1;
            hs[t] = new Matrix(MAPPLY(MSUM(MSUM(MDPROD(weight_x_hidden, xs[t]), MDPROD(weight_hidden_time, hs[t - 1])), bias_hidden), &TANH));
            ys[t] = new Matrix(MDPROD(weight_y_hidden, hs[t]));
            ps[t] = new Matrix(SMUL(MAPPLY(ys[t], &EXP), 1 / (MTOTSUM(MAPPLY(ys[t], &EXP)))));
            loss += -log(ps[t]->data[targets[t]][0]);
        }
        Matrix * dhnext = new Matrix(hs[0]);
        dhnext->setZero();
        for (int t = inputs.size() - 1; t >= 0; t--) {
            auto temp_dy = new Matrix(ps[t]);
            MSROWOP(&temp_dy, targets[t], 1, -1);
            **dweightyh = MSUM(**dweightyh, MDPROD(temp_dy, hs[t]->transpose()));
            **dbiasy = MSUM(**dbiasy, *temp_dy);
            auto temp_dh = new Matrix(MSUM(MDPROD(weight_y_hidden->transpose(), *temp_dy), dhnext));
            auto temp_dhraw = new Matrix(NAIVEMUL(MSOP(NAIVEMUL(*hs[t], *hs[t]), -1, 1), *temp_dh));
            **dbiash = MSUM(*temp_dhraw, *dbiash);
            **dweightxh = MSUM(**dweightxh, MDPROD(temp_dhraw, xs[t]->transpose()));
            **dweighthh = MSUM(**dweighthh, MDPROD(temp_dhraw, hs[t - 1]->transpose()));
            *dhnext = MDPROD(weight_hidden_time->transpose(), temp_dhraw);
        }
        CLIP(**dweighthh, -5, 5);
        CLIP(**dweightxh, -5, 5);
        CLIP(**dweightyh, -5, 5);
        CLIP(**dbiash, -5, 5);
        CLIP(**dbiasy, -5, 5);
        return loss;
    }
    void sample(Matrix ** h, int seed, int n, std::vector<int> & sample_ix) {
        auto x = new Matrix(vocab_size, 1);
        x->setZero();
        x->data[seed][0] = 1;
        for (int t = 0; t < n; t++) {
            **h = MAPPLY(MSUM(MDPROD(weight_x_hidden, x), MDPROD(weight_hidden_time, *h)), &TANH);
            auto y = MSUM(MDPROD(weight_y_hidden, *h), bias_y);
            auto f = FLATTEN(SMUL(MAPPLY(y, &EXP), 1.0 / MTOTSUM(MAPPLY(y, &EXP))));
            std::discrete_distribution<> d(f.begin(), f.end());
            auto k = d(e2);
            x->setZero();
            x->data[k][0] = 1;
            sample_ix.push_back(k);
        }
    }
public:
    RNN(std::vector<char> & data, std::vector<char> & chars) {
        this->data = data;
        this->chars = chars;
    }
    void run_rnn() {
        set_consts();
        create_maps();
        model_parameters();
        int n = 0, p = 0;

        auto mwxh = new Matrix(weight_x_hidden), mwhh = new Matrix(weight_hidden_time), mwhy = new Matrix(weight_y_hidden),
            mbh = new Matrix(bias_hidden), mby = new Matrix(bias_y), dweightxh = new Matrix(weight_x_hidden),
            dweightyh = new Matrix(weight_y_hidden), dweighthh = new Matrix(weight_hidden_time), dbiash = new Matrix(bias_hidden),
            dbiasy = new Matrix(bias_y);

        mwxh->setZero(), mwhy->setZero(), mwhh->setZero(), mbh->setZero(), mby->setZero();
        auto smooth_loss = -log(1.0 / vocab_size) * seq_length;
        auto hprev = new Matrix();
        std::string printer;
        std::vector<int> inputs, targets, sample_ix;

        while (true)
        {
            if (p + seq_length + 1 >= data.size() || n == 0)
            {
                hprev = new Matrix(hidden_size, 1);
                p = 0;
            }
            for (auto ch = data.begin() + p; ch != data.begin() + p + seq_length; ch++)
            {
                inputs.push_back(char_to_num[*ch]);
            }
            for (auto ch = data.begin() + p + 1; ch != data.begin() + p + seq_length + 1; ch++)
            {
                targets.push_back(char_to_num[*ch]);
            }
            if (n % 100 == 0)
            {
                sample(&hprev, inputs[0], 200, sample_ix);
                for (auto ix : sample_ix)
                    printer += num_to_char[ix];
                std::cout << "----\n " << printer << "\n----\n";
                //do stuff TODO
            }
            dweighthh->setZero(), dweightxh->setZero(), dweightyh->setZero();
            dbiash->setZero(), dbiasy->setZero();
            auto _loss = loss(inputs, targets, *hprev, &dweighthh, &dweightxh, &dweightyh, &dbiasy, &dbiash);
            smooth_loss = smooth_loss * .999 + _loss * .001;
            if (n % 100) {
                std::cout << "Iteration " << n << ", Loss: " << smooth_loss << "\n";
            }

            //adagrad updates

            *mwxh = MSUM(*mwxh, NAIVEMUL(dweightxh, dweightxh));
            *weight_x_hidden = MSUM(*weight_x_hidden, NAIVEMUL(SMUL(*dweightxh, -lr), MAPPLY(MAPPLY(MADD(mwxh, 1e-8), &SQRT), &INVERT)));

            *mwhh = MSUM(*mwhh, NAIVEMUL(dweighthh, dweighthh));
            *weight_hidden_time = MSUM(*weight_hidden_time, NAIVEMUL(SMUL(*dweighthh, -lr), MAPPLY(MAPPLY(MADD(mwhh, 1e-8), &SQRT), &INVERT)));

            *mwhy = MSUM(*mwhy, NAIVEMUL(dweightyh, dweightyh));
            *weight_y_hidden = MSUM(*weight_y_hidden, NAIVEMUL(SMUL(*dweightyh, -lr), MAPPLY(MAPPLY(MADD(mwhy, 1e-8), &SQRT), &INVERT)));

            *mbh = MSUM(*mbh, NAIVEMUL(dbiash, dbiash));
            *bias_hidden = MSUM(*bias_hidden, NAIVEMUL(SMUL(*dbiash, -lr), MAPPLY(MAPPLY(MADD(mbh, 1e-8), &SQRT), &INVERT)));

            *mby = MSUM(*mby, NAIVEMUL(dbiasy, dbiasy));
            *bias_y = MSUM(*bias_y, NAIVEMUL(SMUL(*dbiasy, -lr), MAPPLY(MAPPLY(MADD(mby, 1e-8), &SQRT), &INVERT)));

            p += seq_length;
            n += 1;
        }



    }
};