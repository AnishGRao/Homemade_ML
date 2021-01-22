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

    void loss(std::vector <int> & inputs, std::vector<int> & targets, Matrix & hprev) {
        std::unordered_map<int, Matrix *> xs = {}, hs = {}, ys = {}, ps = {};
        hs[-1] = new Matrix(hprev);
        double loss = 0, temp_val;
        for (int t = 0; t < inputs.size(); t++) {

            xs[t] = new Matrix(vocab_size, 1);
            xs[t]->setZero();
            xs[t]->data[inputs[t]][0] = 1;
            //std::cout << hs[-1].data.size() << "\t" << hs[-1].data[0].size() << "\n";
            //std::cout << weight_hidden_time.data.size() << "\t" << weight_hidden_time.data[0].size() << "\n";
            //std::cout << weight_y_hidden.data.size() << "\t" << weight_y_hidden.data[0].size() << "\n";
            //std::cout << xs[t].data.size() << "\t" << xs[t].data[0].size() << "\n";
            //std::cout << weight_x_hidden.data.size() << "\t" << weight_x_hidden.data[0].size() << "\n";
            //hidden state
            //auto A = MDPROD(weight_x_hidden, xs[t]);
            //auto B = MDPROD(weight_hidden_time, hs[t - 1]);
            //auto C = MSUM(A, B);
            //auto D = MSUM(C, bias_hidden);
            //auto E = MAPPLY(D, &TANH);
            hs[t] = new Matrix(MAPPLY(MSUM(MSUM(MDPROD(weight_x_hidden, xs[t]), MDPROD(weight_hidden_time, hs[t - 1])), bias_hidden), &TANH));
            //std::cout << weight_y_hidden->data.size() << "\t" << weight_y_hidden->data[0].size() << "\n";
            //std::cout << hs[t]->data.size() << "\t" << hs[t]->data[0].size() << "\n";

            ys[t] = new Matrix(MDPROD(weight_y_hidden, hs[t]));
            //std::cout << ys[t]->data.size() << "\t" << ys[t]->data[0].size() << "\n";

            ps[t] = new Matrix(SMUL(MAPPLY(ys[t], &EXP), 1 / (MTOTSUM(MAPPLY(ys[t], &EXP)))));

            //std::cout << ps[t]->data.size() << "\t" << ps[t]->data[0].size() << "\n";
            //TODO check if nat log
            //exit(0);
            loss += -log(ps[t]->data[targets[t]][0]);
        }
        Matrix * dweightxh = new Matrix(weight_x_hidden), * dweightyh = new Matrix(weight_y_hidden),
            * dweighthh = new Matrix(weight_hidden_time), * dbiash = new Matrix(bias_hidden),
            * dbiasy = new Matrix(bias_y), * dhnext = new Matrix(hs[0]);
        dweighthh->setZero(), dweightxh->setZero(), dweightyh->setZero();
        dbiash->setZero(), dbiasy->setZero();
        dhnext->setZero();
        for (int t = inputs.size() - 1; t >= 0; t--) {
            auto temp_dy = new Matrix(ps[t]);
            //add -1 to all points in ps[t][targets[t]]'s copy
            //std::cout << temp_dy->data.size() << "\t" << temp_dy->data[0].size() << "\n";

            MSROWOP(&temp_dy, targets[t], 1, -1);
            //std::cout << temp_dy->data.size() << "\t" << temp_dy->data[0].size() << "\n\n\n";

            //auto L = hs[t]->transpose();
            //int ck = 0;
            //std::cout << hs[t]->data.size() << "\t" << hs[t]->data[0].size() << "\n";
            //for (auto i : hs[t]->data) {
            //    for (auto j : i) {
            //        std::cout << j << "\t";
            //    }
            //    std::cout << std::endl;
            //}
            //std::cout << std::endl << "\n\n\n";
            //std::cout << hs[t]->transpose().data.size() << "\t" << hs[t]->transpose().data[0].size() << "\n";
            //ck = 0;
            //for (auto i : L.data) {
            //    for (auto j : i) {
            //        std::cout << j << "\t";
            //    }
            //    std::cout << std::endl;
            //}
            //exit(0);
            //std::cout << temp_dy->data.size() << "\t" << temp_dy->data[0].size() << "\n";
            //std::cout << hs[t]->transpose().data.size() << "\t" << hs[t]->transpose().data[0].size() << "\n";
            * dweightyh = MSUM(*dweightyh, MDPROD(temp_dy, hs[t]->transpose()));
            *dbiasy = MSUM(*dbiasy, *temp_dy);
            //auto A = weight_y_hidden->transpose();
            //std::cout << A.data.size() << "\t" << A.data[0].size() << "\n";
            //std::cout << temp_dy->data.size() << "\t" << temp_dy->data[0].size() << "\n";
            //auto B = MDPROD(A, *temp_dy);
            //auto C = MSUM(B, dhnext);
            //std::cout << weight_y_hidden->data.size() << "\t" << weight_y_hidden->data[0].size() << "\n";
            //std::cout << temp_dy->data.size() << "\t" << temp_dy->data[0].size() << "\n";

            auto temp_dh = new Matrix(MSUM(MDPROD(weight_y_hidden->transpose(), *temp_dy), dhnext));
            //std::cout << hs[t]->data.size() << "\t" << hs[t]->data[0].size() << "\n";
            auto temp_dhraw = new Matrix(NAIVEMUL(MSOP(NAIVEMUL(*hs[t], *hs[t]), -1, 1), *temp_dh));
            *dbiash = MSUM(*temp_dhraw, *dbiash);
            *dweightxh = MSUM(*dweightxh, MDPROD(temp_dhraw, xs[t]->transpose()));
            *dweighthh = MSUM(*dweighthh, MDPROD(temp_dhraw, hs[t - 1]->transpose()));
            *dhnext = MDPROD(weight_hidden_time->transpose(), temp_dhraw);
        }


    }
    void init_RNN() {


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
            mbh = new Matrix(bias_hidden), mby = new Matrix(bias_y);
        mwxh->setZero(), mwhy->setZero(), mwhh->setZero(), mbh->setZero(), mby->setZero();
        auto smooth_loss = -log(1.0 / vocab_size) * seq_length;
        auto hprev = new Matrix();
        std::vector<int> inputs, targets;
        while (1)
        {
            if (p + seq_length + 1 >= data.size() || n == 0)
            {
                hprev = new Matrix(hidden_size, 1);
                hprev->setZero();
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
                ;//do stuff TODO
            }
            loss(inputs, targets, *hprev);
        }



    }
};