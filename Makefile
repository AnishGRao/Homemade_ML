all:
	g++ -O3 -ldl -pthread neural_network.cpp -o ML.o -lncurses;
debug:
	g++ -g -O3 -ldl -pthread neural_network.cpp -o ML.exe -lncurses;
clean:
	rm *.o *.exe
