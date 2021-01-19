all:
	g++ -O3 -ldl -pthread Test_Network.cpp -o ML.o -lncurses;
debug:
	g++ -g -ldl -pthread Test_Network.cpp -o ML.exe -lncurses;
clean:
	rm -f -- *.o *.exe *.txt core; clear;
