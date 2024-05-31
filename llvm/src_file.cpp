#include "src.h"
#include <stdio.h>
#include <stdlib.h> 
#include <stddef.h>

struct Baz {
    int n;
    test_union s; 
    test_union *s_ptr;
    test_union **s_ptrptr;
};

struct RT {
  char A;
  struct RT *B[10][20];
  char C;
  struct RT *D;
};
struct ST {
  int X;
  double Y;
  struct RT *Z;
};

struct NT {
  int X;
  double Y;
  struct RT Z[];
};

struct RT *foo(struct ST *s) {
  int i = 0;
  i += 1;
  return s[1].Z[4].B[i][13];
}
extern "C" {

int test(int i) {
  printf("Hello, world! %d\n", i);
  return 0;
}

class A {
  int n;
  int m;
  char *test3;
};


int main() {
    class A testA = A();
    struct Baz b;
    struct ST s;
    struct NT n;
    struct a {
      int n;
    };
    struct a a1;
    struct Baz *b_ptr = (struct Baz *)malloc(sizeof(struct Baz));  
    srand(0x12345); 
    b.n = rand();
    a1.n = rand();
    foo(&s);
    // b_ptr->s_ptr->m->f.i = 10;
    printf("offsetof(struct Baz, s_ptr) = %lu\n", offsetof(struct Baz, s_ptr));
    printf("Hello, world!, %d, %d\n", b.n, a1.n);
    test(3);

    for(int i = 0; i < 3; i++) {
      printf("Hello, world! %d\n", i);
    }

    for(int i = 0; i < 3; i++) {
      printf("Hello, world2! %d\n", i);
    }
    return 0;
}
}