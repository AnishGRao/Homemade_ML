#include <bits/stdc++.h>
#include <cstdlib>
#include <curses.h>
#include <unistd.h>
/*
THE MAIN PURPOSE OF THIS IS TO CREATE A FRAMEWORK FOR A POKER ML BOT.

I WANT TO DO THIS FROM SCRATCH WITHOUT ANY EXTERNAL LIBS.

I AM A MASOCHIST.

*/
std::vector<int> deck(52, 0);

//if into the security recordings you go... only pain will you find
//                  -Tupac
void deck_creator() {
    setlocale(LC_ALL, "en_US.UTF-8");
    std::ios_base::sync_with_stdio(0);
    int i = 0x1f0a1;
    for (int j = 0; j < 52; j++) {
        deck[j] = i, i += 0x00001, i += i == 0x1f0ac || i == 0x1f0bc || i == 0x1f0cc || i == 0x1f0dc ? 0x00001 : 0,
            i += (i == 0x1f0af) || (i == 0x1f0bf) || (i == 0x1f0cf) ? 2 : 0;
        if (i == 0x1f0df)
            break;
    }
}
