#include "neural_network.h"

void generate_data(std::string filename) {
    std::ofstream file1(filename + "-in");
    std::ofstream file2(filename + "-out");
    for (int i = 0; i < 1000; i++) {
        double x = double(rand()) / double(RAND_MAX);
        double y = double(rand()) / double(RAND_MAX);
        file1 << x << "," << y << "\n";
        file2 << toy_function(x, y) << "\n";
    }
    file1.close();
    file2.close();
}


int main() {
    NeuralNetwork NN({ 2,3,1 });
    std::vector<RowVector> in_data, out_data;
    generate_data("TOY_TEST");
    ReadCsv("TOY_TEST-in", in_data);
    ReadCsv("TOY_TEST-out", out_data);
    NN.train(in_data, out_data);
    return 0;
}