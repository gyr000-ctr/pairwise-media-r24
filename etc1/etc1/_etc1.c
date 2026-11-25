/*
 * ETC1 (Ericsson Texture Compression) block codec core.
 *
 * The Python-facing task models each 4x4 block as 16 scalar values in [0, 1]
 * (i.e. a grayscale block).  A block is still emitted as a fully standard
 * ETC1 64-bit word (two 2x4 / 4x2 subblocks, individual or differential
 * base colours, 8 intensity-modifier tables), so any conforming ETC1
 * decoder recovers the same gray image with R == G == B.
 *
 * This translation unit has no dependency on Python: the ctypes loader in
 * _etc1_native.py calls it directly.  It is compiled with -O3.
 *
 * Public entry points (all thread-safe once etc1_init() has completed):
 *
 *   void     etc1_init(void);
 *   uint64_t etc1_encode_u64(const uint8_t *q16);
 *   void     etc1_decode_rgb(const uint8_t *block, uint8_t *rgb48);
 *   int      etc1_diff_valid(const uint8_t *block);   // 0/1 diagnostics
 *
 * etc1_encode_u64() packs the resulting block as a big-endian 64-bit value
 * (byte 0 is the most significant byte, matching the on-disk ETC1 order).
 * It returns the minimum-error encoding over *both* flip modes, and within
 * each flip over both the individual and the differential base-colour modes.
 */

#include <stdint.h>
#include <string.h>

/* Intensity modifier tables (the 8 ETC1 "table codewords"). */
static const int kMod[8][4] = {
    {-8, -2, 2, 8},
    {-17, -5, 5, 17},
    {-29, -9, 9, 29},
    {-42, -13, 13, 42},
    {-60, -18, 18, 60},
    {-80, -24, 24, 80},
    {-106, -33, 33, 106},
    {-183, -47, 47, 183},
};

/*
 * Mapping between a modifier-table slot (index into kMod[t]) and the 2-bit
 * pattern stored in the selector planes.  The ETC1 bit patterns enumerate
 * the modifier slots in the order "2, 3, 1, 0".
 */
static const uint8_t kRawOfSlot[4] = {3, 2, 0, 1};
static const uint8_t kSlotOfRaw[4] = {2, 3, 1, 0};

/* Per (table, pixel value) cost of every 5-bit / 4-bit base colour.
 * cost5[t][q][c5] = squared error of the best modifier for target q,
 * sel5[t][q][c5]  = the stored 2-bit selector pattern producing it. */
static uint16_t g_cost5[8][256][32];
static uint8_t g_sel5[8][256][32];
static uint16_t g_cost4[8][256][16];
static uint8_t g_sel4[8][256][16];
static int g_exp5[32];
static int g_exp4[16];
static int g_ready = 0;

static int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

static void build_tables(void) {
    int t, q, c, k;
    for (c = 0; c < 32; c++) g_exp5[c] = (c << 3) | (c >> 2);
    for (c = 0; c < 16; c++) g_exp4[c] = (c << 4) | c;
    for (t = 0; t < 8; t++) {
        const int *m = kMod[t];
        for (q = 0; q < 256; q++) {
            for (c = 0; c < 32; c++) {
                int base = g_exp5[c];
                int best = 0x7fffffff, bk = 0;
                for (k = 0; k < 4; k++) {
                    int d = q - clamp255(base + m[k]);
                    int cost = d * d;
                    if (cost < best) { best = cost; bk = k; }
                }
                g_cost5[t][q][c] = (uint16_t)best;
                g_sel5[t][q][c] = kRawOfSlot[bk];
            }
            for (c = 0; c < 16; c++) {
                int base = g_exp4[c];
                int best = 0x7fffffff, bk = 0;
                for (k = 0; k < 4; k++) {
                    int d = q - clamp255(base + m[k]);
                    int cost = d * d;
                    if (cost < best) { best = cost; bk = k; }
                }
                g_cost4[t][q][c] = (uint16_t)best;
                g_sel4[t][q][c] = kRawOfSlot[bk];
            }
        }
    }
    g_ready = 1;
}

void etc1_init(void) {
    if (!g_ready) build_tables();
}

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

/* Summed per-pixel costs for one subblock (8 pixels). */
typedef struct {
    uint32_t c5[8][32];
    uint32_t c4[8][16];
} sub_sums;

static void compute_sums(const uint8_t *q, const int *idx, sub_sums *s) {
    int t, i, c;
    for (t = 0; t < 8; t++) {
        uint32_t a[32];
        uint32_t b[16];
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        for (i = 0; i < 8; i++) {
            const uint16_t *r5 = g_cost5[t][q[idx[i]]];
            const uint16_t *r4 = g_cost4[t][q[idx[i]]];
            for (c = 0; c < 32; c++) a[c] += r5[c];
            for (c = 0; c < 16; c++) b[c] += r4[c];
        }
        memcpy(s->c5[t], a, sizeof(a));
        memcpy(s->c4[t], b, sizeof(b));
    }
}

typedef struct {
    uint32_t err;
    int diff;      /* 0 = individual (4-bit) bases, 1 = differential bases */
    int base0;     /* c4 for individual mode, c5 for differential mode    */
    int base1;     /* c4 for individual mode, signed delta for diff mode  */
    int t0, t1;    /* intensity table index of subblock 0 / 1              */
} enc_result;

/*
 * Best encoding for one flip mode.
 *
 * Individual mode: the two subblocks are independent, so for each of them we
 * take the minimum over all (table, 4-bit base) pairs -- a fully exhaustive
 * search over the legal individual-mode space.
 *
 * Differential mode: the second base is base0 + delta with delta in [-4, 3]
 * and both bases must stay in [0, 31] (otherwise a decoder would have to
 * clamp, which would make the mode bits illegal).  Tables are independent,
 * so first collapse the table axis and then enumerate (base0, delta).
 */
static void best_for_flip(const uint8_t *q, const int *idx0, const int *idx1,
                          enc_result *out) {
    sub_sums s0, s1;
    int t, c, d;
    uint32_t best_abs, best_diff;
    int abs_t0 = 0, abs_c0 = 0, abs_t1 = 0, abs_c1 = 0;
    uint32_t a0[32], a1[32];
    int at0[32], at1[32];
    uint32_t diff_c0 = 0, diff_t0 = 0, diff_t1 = 0, diff_d = 0;

    compute_sums(q, idx0, &s0);
    compute_sums(q, idx1, &s1);

    /* ---- individual (4-bit) bases ---- */
    {
        uint32_t b0 = 0xffffffffu, b1 = 0xffffffffu;
        for (t = 0; t < 8; t++) {
            for (c = 0; c < 16; c++) {
                if (s0.c4[t][c] < b0) { b0 = s0.c4[t][c]; abs_t0 = t; abs_c0 = c; }
                if (s1.c4[t][c] < b1) { b1 = s1.c4[t][c]; abs_t1 = t; abs_c1 = c; }
            }
        }
        best_abs = b0 + b1;
    }

    /* ---- differential (5-bit) bases ---- */
    for (c = 0; c < 32; c++) {
        uint32_t b0 = 0xffffffffu, b1 = 0xffffffffu;
        int tt0 = 0, tt1 = 0;
        for (t = 0; t < 8; t++) {
            if (s0.c5[t][c] < b0) { b0 = s0.c5[t][c]; tt0 = t; }
            if (s1.c5[t][c] < b1) { b1 = s1.c5[t][c]; tt1 = t; }
        }
        a0[c] = b0; at0[c] = tt0;
        a1[c] = b1; at1[c] = tt1;
    }
    best_diff = 0xffffffffu;
    for (c = 0; c < 32; c++) {
        for (d = -4; d <= 3; d++) {
            int c2 = c + d;
            uint32_t e;
            if (c2 < 0 || c2 > 31) continue;   /* illegal: would need clamping */
            e = a0[c] + a1[c2];
            if (e < best_diff) {
                best_diff = e;
                diff_c0 = (uint32_t)c;
                diff_d = (uint32_t)d;
                diff_t0 = (uint32_t)at0[c];
                diff_t1 = (uint32_t)at1[c2];
            }
        }
    }

    if (best_abs <= best_diff) {
        out->err = best_abs;
        out->diff = 0;
        out->base0 = abs_c0;
        out->base1 = abs_c1;
        out->t0 = abs_t0;
        out->t1 = abs_t1;
    } else {
        out->err = best_diff;
        out->diff = 1;
        out->base0 = (int)diff_c0;
        out->base1 = (int)diff_d;
        out->t0 = (int)diff_t0;
        out->t1 = (int)diff_t1;
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

static void pack_block(uint8_t *out, int flip, const enc_result *e,
                       const uint8_t raw[16]) {
    uint64_t v = 0;
    int i, x, y;
    if (e->diff) {
        /* Base colour 5: R=59..63, G=51..55, B=43..47. */
        uint64_t c5 = (uint64_t)(e->base0 & 31);
        /* Delta 3 (two's complement in three bits): R=56..58, G=48..50,
         * B=40..42. */
        uint64_t dl = (uint64_t)(e->base1 & 7);
        v |= c5 << 59;               /* R5 */
        v |= c5 << 51;               /* G5 */
        v |= c5 << 43;               /* B5 */
        v |= dl << 56;               /* dR3 */
        v |= dl << 48;               /* dG3 */
        v |= dl << 40;               /* dB3 */
    } else {
        uint64_t c0 = (uint64_t)(e->base0 & 15);
        uint64_t c1 = (uint64_t)(e->base1 & 15);
        v |= c0 << 60;               /* R1 */
        v |= c0 << 52;               /* G1 */
        v |= c0 << 44;               /* B1 */
        v |= c1 << 56;               /* R2 */
        v |= c1 << 48;               /* G2 */
        v |= c1 << 40;               /* B2 */
    }
    v |= (uint64_t)(flip & 1) << 32;
    v |= (uint64_t)(e->diff & 1) << 33;
    v |= (uint64_t)(e->t1 & 7) << 34;
    v |= (uint64_t)(e->t0 & 7) << 37;
    for (y = 0; y < 4; y++) {
        for (x = 0; x < 4; x++) {
            int bit = x * 4 + y;
            int raw2 = raw[y * 4 + x] & 3;
            v |= (uint64_t)(raw2 & 1) << bit;
            v |= (uint64_t)((raw2 >> 1) & 1) << (16 + bit);
        }
    }
    for (i = 0; i < 8; i++) out[i] = (uint8_t)(v >> (56 - 8 * i));
}

static void selectors_for(const uint8_t *q, const int *idx, int t, int c,
                          int diff, uint8_t raw[8]) {
    int i;
    if (diff) {
        for (i = 0; i < 8; i++) raw[i] = g_sel5[t][q[idx[i]]][c];
    } else {
        for (i = 0; i < 8; i++) raw[i] = g_sel4[t][q[idx[i]]][c];
    }
}

uint64_t etc1_encode_u64(const uint8_t *q16) {
    /* Pixel index = y*4 + x (row major). */
    static const int left0[8] = {0, 1, 4, 5, 8, 9, 12, 13};
    static const int left1[8] = {2, 3, 6, 7, 10, 11, 14, 15};
    static const int top0[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    static const int top1[8] = {8, 9, 10, 11, 12, 13, 14, 15};
    enc_result e0, e1, *best;
    uint8_t raw0[8], raw1[8], raw[16], out[8];
    uint64_t v = 0;
    int i, flip;

    best_for_flip(q16, left0, left1, &e0);
    best_for_flip(q16, top0, top1, &e1);
    if (e0.err <= e1.err) {
        flip = 0;
        best = &e0;
        selectors_for(q16, left0, best->t0, best->base0, best->diff, raw0);
        selectors_for(q16, left1, best->t1,
                      best->diff ? best->base0 + best->base1 : best->base1,
                      best->diff, raw1);
        for (i = 0; i < 8; i++) {
            raw[left0[i]] = raw0[i];
            raw[left1[i]] = raw1[i];
        }
    } else {
        flip = 1;
        best = &e1;
        selectors_for(q16, top0, best->t0, best->base0, best->diff, raw0);
        selectors_for(q16, top1, best->t1,
                      best->diff ? best->base0 + best->base1 : best->base1,
                      best->diff, raw1);
        for (i = 0; i < 8; i++) {
            raw[top0[i]] = raw0[i];
            raw[top1[i]] = raw1[i];
        }
    }
    pack_block(out, flip, best, raw);
    for (i = 0; i < 8; i++) v = (v << 8) | out[i];
    return v;
}

/* Decode a block to 16 RGB triplets (48 bytes). */
void etc1_decode_rgb(const uint8_t *b, uint8_t *rgb48) {
    uint64_t v = 0;
    int i, x, y, ch;
    int flip, diff, t0, t1;
    int base0[3], base1[3];
    uint64_t raw_lsb, raw_msb;

    for (i = 0; i < 8; i++) v = (v << 8) | b[i];
    flip = (int)((v >> 32) & 1);
    diff = (int)((v >> 33) & 1);
    t0 = (int)((v >> 37) & 7);
    t1 = (int)((v >> 34) & 7);

    if (diff) {
        int c5[3], dl[3];
        c5[0] = (int)((v >> 59) & 31);
        c5[1] = (int)((v >> 51) & 31);
        c5[2] = (int)((v >> 43) & 31);
        dl[0] = (int)((v >> 56) & 7);
        dl[1] = (int)((v >> 48) & 7);
        dl[2] = (int)((v >> 40) & 7);
        for (ch = 0; ch < 3; ch++) {
            int d = dl[ch] >= 4 ? dl[ch] - 8 : dl[ch];
            int c2 = c5[ch] + d;
            if (c2 < 0) c2 = 0;
            if (c2 > 31) c2 = 31;
            base0[ch] = g_exp5[c5[ch]];
            base1[ch] = g_exp5[c2];
        }
    } else {
        int r0 = (int)((v >> 60) & 15), g0 = (int)((v >> 52) & 15);
        int b0 = (int)((v >> 44) & 15);
        int r1 = (int)((v >> 56) & 15), g1 = (int)((v >> 48) & 15);
        int b1 = (int)((v >> 40) & 15);
        base0[0] = g_exp4[r0]; base0[1] = g_exp4[g0]; base0[2] = g_exp4[b0];
        base1[0] = g_exp4[r1]; base1[1] = g_exp4[g1]; base1[2] = g_exp4[b1];
    }

    raw_lsb = v & 0xffffu;
    raw_msb = (v >> 16) & 0xffffu;
    for (y = 0; y < 4; y++) {
        for (x = 0; x < 4; x++) {
            int bit = x * 4 + y;
            int raw = (int)(((raw_lsb >> bit) & 1) | (((raw_msb >> bit) & 1) << 1));
            int slot = kSlotOfRaw[raw];
            int sb = flip ? (y >> 1) : (x >> 1);
            const int *base = sb ? base1 : base0;
            const int *m = kMod[sb ? t1 : t0];
            uint8_t *px = rgb48 + (y * 4 + x) * 3;
            for (ch = 0; ch < 3; ch++)
                px[ch] = (uint8_t)clamp255(base[ch] + m[slot]);
        }
    }
}

/*
 * Diagnostic used by the test-suite: returns 0 when the block needs clamping
 * in differential mode (i.e. its base+delta pairs are out of [0, 31]) and 1
 * when it is legal (always 1 for individual mode).
 */
int etc1_diff_valid(const uint8_t *b) {
    uint64_t v = 0;
    int i, ch;
    for (i = 0; i < 8; i++) v = (v << 8) | b[i];
    if (!((v >> 33) & 1)) return 1;
    for (ch = 0; ch < 3; ch++) {
        int shift = 59 - 8 * ch;
        int dshift = 56 - 8 * ch;
        int c5 = (int)((v >> shift) & 31);
        int dl = (int)((v >> dshift) & 7);
        int d = dl >= 4 ? dl - 8 : dl;
        int c2 = c5 + d;
        if (c2 < 0 || c2 > 31) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Convenience entry points used by the Python bindings.              */
/* ------------------------------------------------------------------ */

/*
 * Encode 16 normalized doubles in [0, 1] (row-major 4x4) and write the
 * 8 block bytes.  Values outside the unit interval are clamped, matching
 * the usual texture-compression normalization; non-finite values must be
 * rejected by the caller (they are undefined here).
 */
void etc1_encode_values(const double *v, uint8_t *out8) {
    uint8_t q[16];
    uint64_t w;
    int i;
    for (i = 0; i < 16; i++) {
        double x = v[i];
        int qi;
        if (x <= 0.0) qi = 0;
        else if (x >= 1.0) qi = 255;
        else qi = (int)(x * 255.0 + 0.5);
        q[i] = (uint8_t)qi;
    }
    w = etc1_encode_u64(q);
    for (i = 0; i < 8; i++) out8[i] = (uint8_t)(w >> (56 - 8 * i));
}

/* Decode 8 block bytes into 16 normalized doubles, row-major 4x4. */
void etc1_decode_values(const uint8_t *b, double *out16) {
    uint8_t rgb[48];
    int i;
    etc1_decode_rgb(b, rgb);
    for (i = 0; i < 16; i++) out16[i] = rgb[i * 3] / 255.0;
}

