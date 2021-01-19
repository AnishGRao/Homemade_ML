#include "poker.h"
#include "operations.h"

//will be constantly updated

class NeuralNetwork {
public:

    double activationFunction(double num) {
        return (double) tanhf(num);
    }

    double activationFunctionDerivative(double num) {
        return 1 - tanhf(num) * tanhf(num);
    }
    //todo:
    //make topology ints for speed
    NeuralNetwork(std::vector<uint> topology, float lr = (float) .05) {
        this->topology = topology;
        this->lr = lr;
        for (auto i = 0; i < topology.size(); i++) {
            //todo replace with:
            neuronalLayers.push_back(new RowVector(topology[i] + (i != (topology.size() + 1))));

            //neurons
            //if (i == topology.size() - 1)
            //    neuronalLayers.push_back(new RowVector(topology[i]));
            //else
            //    neuronalLayers.push_back(new RowVector(topology[i] + 1));

            //initializations
            cachingLayers.push_back(new RowVector(neuronalLayers.size()));
            deltas.push_back(new RowVector(neuronalLayers.size()));


            //ones, basically :/
            if (i != topology.size() - 1) {
                neuronalLayers.back()->data[topology[i]] = 1.0;
                cachingLayers.back()->data[topology[i]] = 1.0;
            }

            if (i > 0) {
                if (i != topology.size() - 1) {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                    //todo maybe fix https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ac476e5852129ba32beaa1a8a3d7ee0db
                    weights.back()->setRandom();
                    //just set the rowvec to 0;
                    weights.back()->setZero({ .col = topology[i] });
                    weights.back()->data[topology[i - 1]][topology[i]] = 1.0;
                }
                else {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                    weights.back()->setRandom();
                }
            }
        }
    };

    void propagateForward(RowVector & in) {
        neuronalLayers.front()->set_block(0, 0, 1, neuronalLayers.front()->data.size() - 1, in);

        for (int i = 1; i < this->topology.size(); i++)
            (*neuronalLayers[i]) = MMULT(*neuronalLayers[i - 1], *weights[i - 1]);

        //need to apply activation function to all elements of the current layer
        //auto the activation function
        auto activ_func = std::bind(&NeuralNetwork::activationFunction, this, std::placeholders::_1);
        for (int i = 1; i < this->topology.size(); i++)
            std::for_each(cachingLayers[i]->data.begin(), cachingLayers[i]->data.begin() + topology[i], activ_func),
            std::transform(cachingLayers[i]->data.begin(), cachingLayers[i]->data.begin() + topology[i], neuronalLayers[i]->data.begin(),
                [](double & n) { return n;});

    }

    void propagateBackward(RowVector & out) {
        calcErrors(out);
        updateWeights();
    }

    void calcErrors(RowVector & out) {
        (*deltas.back()) = MSUB(out, (*neuronalLayers.back()));

        for (int i = topology.size() - 2; i > 0; i--) {
            (*deltas[i]) = MMULT(*deltas[i + 1], (weights[i]->transpose()));
        }
    }
    void updateWeights() {
        bool top;
        for (int i = 0; i < topology.size() - 1; i++) {
            top = i != (topology.size() - 2);
            for (int j = 0; j < weights[i]->data[0].size() - top; j++)
                for (int k = 0; k < weights[i]->data.size(); k++)
                    weights[i]->data[k][j] += lr * deltas[i + 1]->data[j] * activationFunctionDerivative(cachingLayers[i + 1]->data[j]) * neuronalLayers[i]->data(k);

        }
    }
    void train(std::vector<RowVector *> input_data, std::vector<RowVector *> output_data) {
        for (int i = 0; i < input_data.size(); i++) {
            std::cout << "Input: "
        }
    }
    std::vector<RowVector *> neuronalLayers;
    std::vector<RowVector *> cachingLayers;
    std::vector<RowVector *> deltas;
    std::vector<Matrix *> weights;
    std::vector<uint> topology;
    float lr;
private:
};