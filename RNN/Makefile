all:
	g++ -ldl -lpthread -std=c++2a main.cpp character-level-language-model.h -o ml; 
	g++ -ldl -lpthread -g -std=c++2a main.cpp character-level-language-model.h -o ml_debug;
debug:
	g++ -ldl -lpthread -g -std=c++2a main.cpp character-level-language-model.h -o ml_debug;
clean:
	rm -f ml ml_debug; clear	
