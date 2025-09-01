#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ORDER 4
#define N 32

struct node {
    int idx;
    struct node *next;
};

struct freelist {
    struct node *head;
};

static struct freelist free_area[MAX_ORDER + 1];
static int block_order[N];

static void freelist_add(struct freelist *fl, int idx) {
    struct node *n = malloc(sizeof(*n));
    n->idx = idx;
    n->next = fl->head;
    fl->head = n;
}

static void freelist_remove(struct freelist *fl, int idx) {
    struct node **p = &fl->head, *cur;
    while ((cur = *p)) {
        if (cur->idx == idx) {
            *p = cur->next;
            free(cur);
            return;
        }
        p = &cur->next;
    }
}

static void print_state(const char *msg) {
    int i;
    printf("-- %s ---\n", msg);

    printf("block_order: ");
    for (i = 0; i < N; i++) {
        printf("%2d ", block_order[i]);
    }
    printf("\n");

    for (i = 0; i <= MAX_ORDER; i++) {
        struct node *n = free_area[i].head;
        printf("free_area[%d]:", i);
        while (n) {
            printf(" %d", n->idx);
            n = n->next;
        }
        printf("\n");
    }
    printf("-------------\n");
}

static void buddy_init() {
    int i;
    for (i = 0; i < N; i++)
        block_order[i] = -1;
    for (i = 0; i <= MAX_ORDER; i++)
        free_area[i].head = NULL;

    for (i = 0; i < N; i += (1 << MAX_ORDER)) {
        freelist_add(&free_area[MAX_ORDER], i);
        block_order[i] = MAX_ORDER;
    }
}

/**
 * When splitting, two buddies are always produced.
 * left buddy at idx, right buddy at idx + (1 << order)
 * always start from the leftmost block of a free buddy
 */
static int alloc(int order) {
    int o;
    for (o = order; o <= MAX_ORDER; o++) {
        if (free_area[o].head) {
            int idx = free_area[o].head->idx;
            freelist_remove(&free_area[o], idx);
            while (o > order) {
                o--;
                int buddy = idx + (1 << o);
                printf("  alloc split buddy %d for block 0x%x of order 0x%x, mask 0x%x\n", buddy, idx, o, (1 << o));
                freelist_add(&free_area[o], buddy);
                block_order[buddy] = o;
            }
            block_order[idx] = -1;
            print_state("after alloc");
            return idx;
        }
    }
    return -1; // fail
}

/**
 * idx is always multiple of (1 << order)
 */
static void free_block(int idx, int order) {
    int curr_order = order;
    int block_start = idx;

    // this loop checking and merging buddy blocks
    while (curr_order < MAX_ORDER) {
        // check if neighbor block (under the same order) is freed
        int buddy = block_start ^ (1 << curr_order);
        printf("  free buddy %d for block 0x%x of order 0x%x, mask 0x%x\n", buddy, block_start, curr_order, (1 << curr_order));
        if (block_order[buddy] != curr_order)
            break;
        freelist_remove(&free_area[curr_order], buddy);
        if (buddy < block_start)
            block_start = buddy;
        curr_order++;
    }

    freelist_add(&free_area[curr_order], block_start);
    block_order[block_start] = curr_order;

    print_state("after free");
}

static int reserve(int idx, int order) {
    int o, block_start;

    // alignment check
    if (idx & ((1 << order) - 1)) {
        printf("reserve(%d,%d) alignment error\n", idx, order);
        return -1;
    }

    // find covering block
    for (o = order; o <= MAX_ORDER; o++) {
        block_start = idx & ~((1 << o) - 1);
        printf("  reserve block_start 0x%x for block 0x%x of order 0x%x, mask 0x%x\n", block_start, idx, o, ~((1 << o) - 1));
        if (block_order[block_start] == o) {
            freelist_remove(&free_area[o], block_start);
            /**
             * split down, unlike alloc, reserve at specific idx
             * We don’t know in advance whether the requested offset is
             * in the left half or the right half of the block. so we use XOR
             */
            while (o > order) {
                o--;
                int buddy = block_start ^ (1 << o);
                printf("  split buddy 0x%x for block 0x%x of order 0x%x, mask 0x%x\n", buddy, block_start, o, (1 << o));
                freelist_add(&free_area[o], buddy);
                block_order[buddy] = o;
            }
            break;
        }
    }

    // mark allocated
    block_order[idx] = -1;

    print_state("after reserve");
    return idx;
}

int main() {
    buddy_init();
    print_state("after init");

    int a = alloc(0);
    printf("  alloc a 0x%d\n\n", a);
    int b = alloc(0);
    printf("  alloc b 0x%d\n\n", b);
    free_block(a, 0);
    printf("  freed a 0x%d\n\n", a);
    free_block(b, 0);
    printf("  freed b 0x%d\n\n", b);

    int c = alloc(2);

    int r = reserve(8, 2); // force reserve at idx=8, order=2

    free_block(c, 2);
    free_block(r, 2);

    return 0;
}
