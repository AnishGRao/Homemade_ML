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
        //get size of data, and total vocabulary (data is all chars, and chars is all unique chars)
        data_size = data.size();
        vocab_size = chars.size();

        //hyperparameters--try changing them and see what happens!
        hidden_size = 100;
        seq_length = 25;
        lr = .1;
    }
    void create_maps() {
        //create two maps--one that maps from a character to a number, and another that does the opposite
        //since ML inputs and outputs numbers, this is how we translate chars to text, and vice versa
        for (int i = 0; i < vocab_size; i++)
            char_to_num[chars[i]] = i, num_to_char[i] = chars[i];
    }
    void model_parameters() {

        //initialize the random eng as a mersenne twister
        std::mt19937 e3(this->rd());
        e2 = e3;

        //the following simply create matrices with the given sizes, and set the values inside to a standard normal distribution
        //the random numbers are then multiplied by .01. this is to keep them small-ish.
        this->weight_x_hidden = new Matrix(hidden_size, vocab_size);
        this->weight_x_hidden->setRandom(.01);

        this->weight_hidden_time = new Matrix(hidden_size, hidden_size);
        this->weight_hidden_time->setRandom(.01);

        this->weight_y_hidden = new Matrix(vocab_size, hidden_size);
        this->weight_y_hidden->setRandom(.01);

        this->bias_hidden = new Matrix(hidden_size, 1);

        this->bias_y = new Matrix(vocab_size, 1);
    }

    //the reason for the disgustingly long call is simple
    //i really really really dont want to deal with returning 5 Matrices every call--so make them by reference!
    //also, this calculates loss.
    double loss(std::vector <int> & inputs, std::vector<int> & targets, Matrix & hprev,
        Matrix ** dweighthh, Matrix ** dweightxh, Matrix ** dweightyh, Matrix ** dbiasy, Matrix ** dbiash) {

        //multitudinous unordered maps to map integers to matrices, and hold weights accordingly.
        std::unordered_map<int, Matrix *> xs = {}, hs = {}, ys = {}, ps = {};

        //begin the iteration by setting hidden's "first" weight to the previous weight. It is negative 1, because we start normally 
        //from  0, and cant check previous from there.
        hs[-1] = new Matrix(hprev);

        //regular inits.
        double loss = 0, temp_val;

        //iterate over the size of the inputs (a list of ints, just like targets.)
        for (int t = 0; t < inputs.size(); t++) {
            //encoding a 1-of-k representation
            xs[t] = new Matrix(vocab_size, 1);
            xs[t]->data[inputs[t]][0] = 1;

            //hidden state.
            hs[t] = new Matrix(MAPPLY(MSUM(MSUM(MDPROD(weight_x_hidden, xs[t]), MDPROD(weight_hidden_time, hs[t - 1])), bias_hidden), &TANH));
            //log likelihood
            ys[t] = new Matrix(MDPROD(weight_y_hidden, hs[t]));
            //actual probabilities for chars
            ps[t] = new Matrix(SMUL(MAPPLY(ys[t], &EXP), 1 / (MTOTSUM(MAPPLY(ys[t], &EXP)))));
            //softmax.
            loss += -log(ps[t]->data[targets[t]][0]);
        }

        Matrix * dhnext = new Matrix(hs[0]);
        dhnext->setZero();
        for (int t = inputs.size() - 1; t >= 0; t--) {
            //backpropagate into y, basically analytic grad descent
            auto temp_dy = new Matrix(ps[t]);
            MSROWOP(&temp_dy, targets[t], 1, -1);

            **dweightyh = MSUM(**dweightyh, MDPROD(temp_dy, hs[t]->transpose()));

            **dbiasy = MSUM(**dbiasy, *temp_dy);

            //backpropagate into hidden
            auto temp_dh = new Matrix(MSUM(MDPROD(weight_y_hidden->transpose(), *temp_dy), dhnext));

            //backpropagate through tanh 
            auto temp_dhraw = new Matrix(NAIVEMUL(MSOP(SMUL(NAIVEMUL(*hs[t], *hs[t]), -1), 1, 1), *temp_dh));

            **dbiash = MSUM(*temp_dhraw, *dbiash);

            **dweightxh = MSUM(**dweightxh, MDPROD(temp_dhraw, xs[t]->transpose()));

            **dweighthh = MSUM(**dweighthh, MDPROD(temp_dhraw, hs[t - 1]->transpose()));

            *dhnext = MDPROD(weight_hidden_time->transpose(), temp_dhraw);
        }

        //clipping to kill explosions--try changing these--see how messy it gets.
        CLIP(**dweighthh, -5, 5);
        CLIP(**dweightxh, -5, 5);
        CLIP(**dweightyh, -5, 5);
        CLIP(**dbiash, -5, 5);
        CLIP(**dbiasy, -5, 5);
        return loss;
    }

    //simply sample from h, which is the memory state, and get it using a seed and a seed letter.
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

    //this step took too long when it was single-threaded, so I multithreaded it.
    //no shared arrays, and simply forwarding through the NN, and getting the gradient.
    void adagrad(Matrix ** m, Matrix * d, char c) {
        **m = MSUM(**m, NAIVEMUL(*d, *d));
        switch (c)
        {
        case 0:
            *weight_x_hidden = MSUM(*weight_x_hidden, NAIVEMUL(SMUL(*d, -lr), MAPPLY(MAPPLY(MADD(**m, 1e-8), &SQRT), &INVERT)));
            break;
        case 1:
            *weight_hidden_time = MSUM(*weight_hidden_time, NAIVEMUL(SMUL(*d, -lr), MAPPLY(MAPPLY(MADD(**m, 1e-8), &SQRT), &INVERT)));
            break;
        case 2:
            *weight_y_hidden = MSUM(*weight_y_hidden, NAIVEMUL(SMUL(*d, -lr), MAPPLY(MAPPLY(MADD(**m, 1e-8), &SQRT), &INVERT)));
            break;
        case 3:
            *bias_hidden = MSUM(*bias_hidden, NAIVEMUL(SMUL(*d, -lr), MAPPLY(MAPPLY(MADD(**m, 1e-8), &SQRT), &INVERT)));
            break;
        case 4:
            *bias_y = MSUM(*bias_y, NAIVEMUL(SMUL(*d, -lr), MAPPLY(MAPPLY(MADD(**m, 1e-8), &SQRT), &INVERT)));
            break;
        default: break;
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

        //just init all of the matrices we will be using--theres a lot.
        auto mwxh = new Matrix(weight_x_hidden), mwhh = new Matrix(weight_hidden_time), mwhy = new Matrix(weight_y_hidden),
            mbh = new Matrix(bias_hidden), mby = new Matrix(bias_y), dweightxh = new Matrix(weight_x_hidden),
            dweightyh = new Matrix(weight_y_hidden), dweighthh = new Matrix(weight_hidden_time), dbiash = new Matrix(bias_hidden),
            dbiasy = new Matrix(bias_y);

        mwxh->setZero(), mwhy->setZero(), mwhh->setZero(), mbh->setZero(), mby->setZero();

        //loss at iter0.
        auto smooth_loss = -log(1.0 / vocab_size) * seq_length;

        auto hprev = new Matrix();
        std::string printer;
        std::vector<int> inputs, targets, sample_ix;

        while (true)
        {

            //prepare hprev for the sampling.
            if (p + seq_length + 1 >= data.size() || n == 0)
            {
                delete hprev;
                hprev = new Matrix(hidden_size, 1);
                p = 0;
            }

            //get inputs and targets from data, notice we are doing it at "se_length" length
            //this is the size of input data. We keep sampling from this over and over, and this is how we test it.
            for (auto ch = data.begin() + p; ch != data.begin() + p + seq_length; ch++)
            {
                inputs.emplace_back(char_to_num[*ch]);
            }
            for (auto ch = data.begin() + p + 1; ch != data.begin() + p + seq_length + 1; ch++)
            {
                targets.emplace_back(char_to_num[*ch]);
            }

            //whenever we have done 100 iterations, sample from the net, and show output.
            //you can delete this code, and nothing but the output will change--it is simply for visuals.
            if (n % 100 == 0)
            {
                sample(&hprev, inputs[0], 200, sample_ix);
                for (auto ix : sample_ix)
                    printer += num_to_char[ix];
                std::cout << "----\n " << printer << "\n----\n";
                printer.clear();
                //do stuff TODO
            }
            //we need to set these to 0, as originally they were local to the loss function, but since they are used later here, 
            //we need to reset these every run.
            dweighthh->setZero(), dweightxh->setZero(), dweightyh->setZero();
            dbiash->setZero(), dbiasy->setZero();

            //forward through net, and grab gradient.
            auto _loss = loss(inputs, targets, *hprev, &dweighthh, &dweightxh, &dweightyh, &dbiasy, &dbiash);
            smooth_loss = smooth_loss * .999 + _loss * .001;
            if (n % 100 == 0)
            {
                std::cout << "Iteration " << n << ", Loss: " << smooth_loss << "\n";
            }

            //adagrad updates--commented out is the single threaded implementation of it.
            //its probably one of the simplest multithreading applications you'll see--
            //call a func, and join--thats it.
            /*
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
            */
            std::thread _hx(&RNN::adagrad, this, &mwxh, dweightxh, 0);
            std::thread _ht(&RNN::adagrad, this, &mwhh, dweighthh, 1);
            std::thread _hy(&RNN::adagrad, this, &mwhy, dweightyh, 2);
            std::thread _bh(&RNN::adagrad, this, &mbh, dbiash, 3);
            std::thread _by(&RNN::adagrad, this, &mby, dbiasy, 4);

            int join_counter = 5;
            while (join_counter != 0) {
                if (_hx.joinable())
                    _hx.join(), join_counter--;
                if (_ht.joinable())
                    _ht.join(), join_counter--;
                if (_hy.joinable())
                    _hy.join(), join_counter--;
                if (_bh.joinable())
                    _bh.join(), join_counter--;
                if (_by.joinable())
                    _by.join(), join_counter--;
            }
            p += seq_length;
            n += 1;
            //clear these, so that we dont just keep adding to them.
            inputs.clear();
            targets.clear();
            sample_ix.clear();
        }



    }
};