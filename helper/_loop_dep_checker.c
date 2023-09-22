#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef enum { Init, Read, Write } RWStatus;
#define logn 30
RWStatus poly[1 << logn];

bool setToRead(uint64_t i) {
    if (poly[i] == Write) return false;
    poly[i] = Read;
    return true;
}

bool setToWrite(uint64_t i) {
    if (poly[i] == Read || poly[i] == Write) return false;
    poly[i] = Write;
    return true;
}

int main() {
    memset(poly, 0, sizeof(poly));

    for(int offset=(1<<3);offset<(1<<logn);offset+=(1<<(3+1))) {
        for(int i=offset+(1<<3)-1-4;i>=offset+(1<<3)-6;--i) {assert(setToWrite(i)); assert(setToRead(4));}
        for(int i=offset+(1<<3)-1-6;i>=offset+(1<<3)-7;--i) {assert(setToWrite(i)); assert(setToRead(4));assert(setToRead(6));}
        for(int i=offset+(1<<3)-1-7;i>=offset+(1<<3)-8;--i) {assert(setToWrite(i)); assert(setToRead(4));assert(setToRead(6));assert(setToRead(7));}
    }

    for(int offset=(1<<3);offset<(1<<logn);offset+=(1<<(3+1))) {
        printf("offset = %d\n", offset);
        for(int i=offset-1-0;i>=offset-4;--i) {assert(setToWrite(i)); assert(setToRead(4));assert(setToRead(6));assert(setToRead(7));}
        for(int i=offset-1-4;i>=offset-6;--i) {assert(setToWrite(i)); assert(setToRead(6));assert(setToRead(7));}
        for(int i=offset-1-6;i>=offset-7;--i) {assert(setToWrite(i)); assert(setToRead(7));}
    }

    printf("No dependency found!\n");
    return 0;
}