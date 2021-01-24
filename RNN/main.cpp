#include "character-level-language-model.h"


int main(int argc, char ** argv) {
	if (argc == 1)
	{
		std::cerr << "no input file.";
		exit(1);
	}
	std::vector<char> data, chars;
	ReadFileChars(argv[1], data, chars);
	auto rnn = new RNN(data, chars);
	time_t start, end;
	time(&start);
	rnn->run_rnn();
	time(&end);
	std::cout << std::endl << (double) difftime(end, start) << std::endl;
	return 0;
}
