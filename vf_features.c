/*
 * vf_features.c
 * 
 * Copyright 2016 Hong Jen Yee (PCMan) <pcman.tw@gmail.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * 
 */

#define _GNU_SOURCE  /* for memmem() */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/*
static void print_bin_seq(const uint8_t* binary_sequence, int length) {
    int i;
    for(i = 0; i < length; ++i) {
        putchar(binary_sequence[i] ? '1' : '0');
    }
}
*/

/* binary_sequence is a sequence containing 0 and 1 only. */
double lempel_ziv_complexity(const uint8_t* binary_sequence, int length) {
    int cn = 1;
    double bn;
    const uint8_t* end_of_sequence = binary_sequence + length;
    const uint8_t* S = binary_sequence;
    int S_length = 1;
    const uint8_t* Q = binary_sequence + S_length;
    int Q_length = 1;
    while((Q + Q_length) <= end_of_sequence) {
        int SQpi_length = S_length + Q_length - 1;
        /* SQpi: SQ concatenation with the last char deleted => S + Q[:-1] */
        void* found = memmem(S, SQpi_length, Q, Q_length);
        if(found) {  /* current Q is a substring of S + Q[:-1] */
            ++Q_length;  /* add the last char to Q */
            
        } else {  /* current Q is not a substring of S + Q[:-1] */
            ++cn;  /* increase complexity count */
            S_length += Q_length;  /* append Q to S */
            Q += Q_length;  /* reset Q to the next unseen char */
            Q_length = 1;
        }
    }
    /* normalization => C(n) = c(n)/b(n), b(n) = n/log2 n */
    bn = (double)length / log2(length);
    return (double)cn / bn;
}


/* calculate the LZ complexity for a IMF derived from EMD */
double imf_lempel_ziv_complexity(const double* imf, int length) {
    double lz = 0.0;
    /* convert the float sequence from EMD/IMF to 12-bit precision integer sequences */
    int i, j;
    int bin_len = length * 12;
    uint8_t* binary_sequence = (uint8_t*)malloc(bin_len * sizeof(uint8_t));
    for(i = 0, j = 11; i < length; ++i, j += 12) {
        int c;
        /* FIXME: handle byte order problems */
        uint16_t value = (uint16_t)imf[i];
        /* convert to bits */
        for(c = 0; c < 12; ++c) {
            binary_sequence[j - c] = (uint8_t)(value >> c) & 1;
        }
    }
    lz = lempel_ziv_complexity(binary_sequence, bin_len);
    free(binary_sequence);
    return lz;
}

/*

int main(int argc, char** argv) {
    uint8_t test[] = {1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1};
    printf("LZ: %lf\n", lempel_ziv_complexity(test, sizeof(test) / sizeof(char)));
    return 0;
}

*/
