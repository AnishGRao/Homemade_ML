#include "operations.h"

void ReadCsv(std::string filename, std::vector<RowVector> & data) {
    //if there are multi-passes
    data.clear();
    std::ifstream file(filename);
    std::string line, element;
    std::getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<double> parser;
    //please dont use comma+space--just delete all spaces.
    //go into vim and do:
    /*
        esc
        :%s/ //g
    */
    while (std::getline(ss, element, ',')) {
        parser.push_back(double(std::stod(&element[0])));
    }
    int nCols = parser.size();
    data.push_back(RowVector(nCols));
    for (int i = 0; i < nCols; i++)
        data.back().data[i] = parser[i];

    if (file.is_open()) {
        int k = 0;
        while (std::getline(file, line, '\n')) {
            std::stringstream ss(line);
            data.push_back(RowVector(nCols));
            int i = 0;
            std::cout << k++ << "\n";
            if (line == "0.835887,0.682699") {
                std::cout << "break" << "\n";
            }
            while (std::getline(ss, element, ',')) {
                data.back().data[i++] = double(std::stod(&element[0]));
            }
        }
    }
}

double toy_function(double x, double y) {
    return 2 * x + 10 + y;
}
