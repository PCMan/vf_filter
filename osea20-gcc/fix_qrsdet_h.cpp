#include <iostream>
#include "qrsdet.h.ori"

using namespace std;

#define PRE_BLANK	MS200

#define OUTPUT(f, name)   cout << "#define\t" #name "\t" << name << endl

int main(int argc, char** argv) {
    OUTPUT(f, SAMPLE_RATE);
    OUTPUT(f, MS_PER_SAMPLE);
    OUTPUT(f, MS10);
    OUTPUT(f, MS25);
    OUTPUT(f, MS30);
    OUTPUT(f, MS80);
    OUTPUT(f, MS95);
    OUTPUT(f, MS100);
    OUTPUT(f, MS125);
    OUTPUT(f, MS150);
    OUTPUT(f, MS160);
    OUTPUT(f, MS175);
    OUTPUT(f, MS195);
    OUTPUT(f, MS200);
    OUTPUT(f, MS220);
    OUTPUT(f, MS250);
    OUTPUT(f, MS300);
    OUTPUT(f, MS360);
    OUTPUT(f, MS450);
    OUTPUT(f, MS1000);
    OUTPUT(f, MS1500);
    OUTPUT(f, DERIV_LENGTH);
    OUTPUT(f, LPBUFFER_LGTH);
    OUTPUT(f, HPBUFFER_LGTH);
    OUTPUT(f, WINDOW_WIDTH);
    OUTPUT(f, FILTER_DELAY);
    OUTPUT(f, DER_DELAY);
    return 0;
}

