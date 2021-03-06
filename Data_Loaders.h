#include "operations.h"

void ReadFileChars(char * filename, std::vector<char> & data, std::vector<char> & chars) {
    data.clear();
    chars.clear();
    std::ifstream file(filename);
    char ch;
    while (file.get(ch)) {
        data.push_back(ch);
    }
    std::set<char> temp(data.begin(), data.end());
    chars = std::vector<char>(temp.begin(), temp.end());
}

void ReadCsv(std::string filename, std::vector<RowVector *> & data) {
    //if there are multi-passes
    data.clear();
    std::ifstream file(filename.c_str());
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
    data.push_back(new RowVector(nCols));

    for (int i = 0; i < nCols; i++)
        data.back()->data[i] = parser[i];

    if (file.is_open()) {
        int k = 0;
        int i;
        while (std::getline(file, line, '\n')) {
            std::stringstream ss(line);
            data.push_back(new RowVector(nCols));
            i = 0;
            while (std::getline(ss, element, ',')) {
                data.back()->data[i++] = double(std::stod(&element[0]));
            }
        }
    }
}

double toy_function(double x, double y) {
    return x * y + 10;
}
