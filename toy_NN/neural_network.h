#include "../Data_Loaders.h"


class NeuralNetwork {
public:
    std::vector<RowVector> neuronalLayers;
    std::vector<RowVector> cachingLayers;
    std::vector<RowVector> deltas;
    std::vector<Matrix> weights;
    std::vector<int> topology;
    float lr;
    double activationFunction(double num) {
        return (double) num > 0 ? num : 0;
    }

    double activationFunctionDerivative(double num) {
        return (num <= 0) ? 0 : 1;
    }

    NeuralNetwork(const NeuralNetwork & other) {
        this->neuronalLayers = other.neuronalLayers;
        this->cachingLayers = other.cachingLayers;
        this->deltas = other.deltas;
        this->weights = other.weights;
        this->topology = other.topology;
        this->lr = other.lr;
    }

    NeuralNetwork(std::vector<int> topology, float lr = (float) .01) {
        this->topology = topology;
        this->lr = lr;
        deltas.resize(topology.size());
        for (auto i = 0; i < topology.size(); i++) {
            //todo replace with:
            //neuronalLayers.push_back(new RowVector(topology[i] + (i != (topology.size() + 1))));

            //neurons
            if (i == topology.size() - 1)
                neuronalLayers.push_back(RowVector(topology[i]));
            else
                neuronalLayers.push_back(RowVector(topology[i] + 1));

            //initializations
            auto tempVar = RowVector(neuronalLayers.size());
            cachingLayers.push_back(tempVar);
            deltas[i] = tempVar;


            //ones, basically :/
            if (i != topology.size() - 1) {
                neuronalLayers.back().data[topology[i]] = 1.0;
                cachingLayers.back().data[topology[i]] = 1.0;
            }

            if (i > 0) {
                if (i != topology.size() - 1) {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                    //todo maybe fix https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ac476e5852129ba32beaa1a8a3d7ee0db
                    weights.back().setRandom();
                    //just set the rowvec to 0;
                    weights.back().setZero({ -1, topology[i] });
                    weights.back().data[topology[i - 1]][topology[i]] = 1.0;
                }
                else {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                    weights.back().setRandom();
                }
            }
        }
    };

    void propagateForward(RowVector & in) {
        cachingLayers.front().set_block(0, 0, 1, neuronalLayers.front().data.size() - 1, in);

        for (int i = 1; i < this->topology.size(); i++)
            (cachingLayers[i]) = MMULT(cachingLayers[i - 1], weights[i - 1]);

        //need to apply activation function to all elements of the current layer
        //auto the activation function
        auto activ_func = std::bind(&NeuralNetwork::activationFunction, this, std::placeholders::_1);
        for (int i = 1; i < this->topology.size(); i++)
            std::for_each(cachingLayers[i].data.begin(), cachingLayers[i].data.begin() + topology[i], activ_func),
            std::transform(cachingLayers[i].data.begin(), cachingLayers[i].data.begin() + topology[i], neuronalLayers[i].data.begin(),
                [](double & n) { return n;});

    }

    void propagateBackward(RowVector & out) {
        calcErrors(out);
        updateWeights();
    }

    void calcErrors(RowVector & out) {
        (deltas.back()) = MSUB(out, (neuronalLayers.back()));

        for (int i = topology.size() - 2; i > 0; i--) {
            (deltas[i]) = MMULT(deltas[i + 1], (weights[i].transpose()));
        }
    }
    void updateWeights() {
        bool top;
        for (int i = 0; i < topology.size() - 1; i++) {
            top = i != (topology.size() - 2);
            for (int j = 0; j < weights[i].data[0].size() - top; j++)
                for (int k = 0; k < weights[i].data.size(); k++)
                    weights[i].data[k][j] += neuronalLayers[i].data[k] * lr * deltas[i + 1].data[j] * activationFunctionDerivative(cachingLayers[i + 1].data[j]);

        }
    }
    void train(std::vector<RowVector *> input_data, std::vector<RowVector *> output_data) {
        for (int i = 0; i < input_data.size(); i++) {
            //std::cout << "Input: " << (*input_data[i]);
            propagateForward(*input_data[i]);
            //std::cout << "Correct: " << *output_data[i];
            //std::cout << "Produced: " << neuronalLayers.back();
            propagateBackward(*output_data[i]);
            //std::cout << "MSE at iteration " << i << ": " << std::sqrt(DPROD(deltas.back(), deltas.back()) / deltas.back().data.size()) << "\n\n";
            //MSE GRAPH
            for (int i = 0; i < ceil(std::sqrt(DPROD(deltas.back(), deltas.back()) / deltas.back().data.size())); i++) {
                std::cout << "X";
            }
            std::cout << std::endl;
        }
    }
};