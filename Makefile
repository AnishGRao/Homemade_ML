all:
	g++ -O3 -ldl -pthread Test_Network.cpp -o ML.o;
debug:
	g++ -g -ldl -pthread Test_Network.cpp -o ML.exe;
clean:
	rm -f -- *.o *.exe *.txt core Test_Network TOY_TEST*; clear;
