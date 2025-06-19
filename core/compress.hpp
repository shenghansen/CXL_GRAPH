#ifndef COMPRESS_HPP
#define COMPRESS_HPP

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <string.h>
#include <thread>

#include "core/bitmap.hpp"
#include "core/type.hpp"

using namespace std;

#define newA(__E, __n) (__E*)malloc((__n) * sizeof(__E))
#define new_remote(__E, __n, __node) (__E*)numa_alloc_onnode((__n) * sizeof(__E), __node)
#define parallel_for _Pragma("omp parallel for") for

template<class E> struct identityF {
    E operator()(const E& x) { return x; }
};

template<class E> struct addF {
    E operator()(const E& a, const E& b) const { return a + b; }
};

template<class E> struct minF {
    E operator()(const E& a, const E& b) const { return (a < b) ? a : b; }
};

template<class E> struct maxF {
    E operator()(const E& a, const E& b) const { return (a > b) ? a : b; }
};

struct nonMaxF {
    bool operator()(uint& a) { return (a != UINT_MAX); }
};

// Sugar to pass in a single f and get a struct suitable for edgeMap.
template<class F> struct EdgeMap_F {
    F f;
    EdgeMap_F(F& _f)
        : f(_f) {}
    inline bool update(const uint& s, const uint& d) { return f(s, d); }

    inline bool updateAtomic(const uint& s, const uint& d) { return f(s, d); }

    inline bool cond(const uint& d) const { return true; }
};

#define LAST_BIT_SET(b) (b & (0x80))
#define EDGE_SIZE_PER_BYTE 7

#define PARALLEL_DEGREE 1000

#define _SCAN_LOG_BSIZE 10
#define _SCAN_BSIZE (1 << _SCAN_LOG_BSIZE)

template<class T> struct _seq {
    T* A;
    long n;
    _seq() {
        A = NULL;
        n = 0;
    }
    _seq(T* _A, long _n)
        : A(_A)
        , n(_n) {}
    void del() { free(A); }
};

namespace sequence {
template<class intT> struct boolGetA {
    bool* A;
    boolGetA(bool* AA)
        : A(AA) {}
    intT operator()(intT i) { return (intT)A[i]; }
};

template<class ET, class intT> struct getA {
    ET* A;
    getA(ET* AA)
        : A(AA) {}
    ET operator()(intT i) { return A[i]; }
};

template<class IT, class OT, class intT, class F> struct getAF {
    IT* A;
    F f;
    getAF(IT* AA, F ff)
        : A(AA)
        , f(ff) {}
    OT operator()(intT i) { return f(A[i]); }
};

#define nblocks(_n, _bsize) (1 + ((_n) - 1) / (_bsize))

#define blocked_for(_i, _s, _e, _bsize, _body)     \
    {                                              \
        intT _ss = _s;                             \
        intT _ee = _e;                             \
        intT _n = _ee - _ss;                       \
        intT _l = nblocks(_n, _bsize);             \
        parallel_for(intT _i = 0; _i < _l; _i++) { \
            intT _s = _ss + _i * (_bsize);         \
            intT _e = min(_s + (_bsize), _ee);     \
            _body                                  \
        }                                          \
    }

template<class OT, class intT, class F, class G> OT reduceSerial(intT s, intT e, F f, G g) {
    OT r = g(s);
    for (intT j = s + 1; j < e; j++) r = f(r, g(j));
    return r;
}

template<class OT, class intT, class F, class G> OT reduce(intT s, intT e, F f, G g) {
    intT l = nblocks(e - s, _SCAN_BSIZE);
    if (l <= 1) return reduceSerial<OT>(s, e, f, g);
    OT* Sums = newA(OT, l);
    blocked_for(i, s, e, _SCAN_BSIZE, Sums[i] = reduceSerial<OT>(s, e, f, g););
    OT r = reduce<OT>((intT)0, l, f, getA<OT, intT>(Sums));
    free(Sums);
    return r;
}

template<class OT, class intT, class F> OT reduce(OT* A, intT n, F f) {
    return reduce<OT>((intT)0, n, f, getA<OT, intT>(A));
}

template<class OT, class intT> OT plusReduce(OT* A, intT n) {
    return reduce<OT>((intT)0, n, addF<OT>(), getA<OT, intT>(A));
}

// g is the map function (applied to each element)
// f is the reduce function
// need to specify OT since it is not an argument
template<class OT, class IT, class intT, class F, class G> OT mapReduce(IT* A, intT n, F f, G g) {
    return reduce<OT>((intT)0, n, f, getAF<IT, OT, intT, G>(A, g));
}

template<class intT> intT sum(bool* In, intT n) {
    return reduce<intT>((intT)0, n, addF<intT>(), boolGetA<intT>(In));
}

template<class ET, class intT, class F, class G>
ET scanSerial(ET* Out, intT s, intT e, F f, G g, ET zero, bool inclusive, bool back) {
    ET r = zero;
    if (inclusive) {
        if (back)
            for (intT i = e - 1; i >= s; i--) Out[i] = r = f(r, g(i));
        else
            for (intT i = s; i < e; i++) Out[i] = r = f(r, g(i));
    } else {
        if (back)
            for (intT i = e - 1; i >= s; i--) {
                ET t = g(i);
                Out[i] = r;
                r = f(r, t);
            }
        else
            for (intT i = s; i < e; i++) {
                ET t = g(i);
                Out[i] = r;
                r = f(r, t);
            }
    }
    return r;
}

template<class ET, class intT, class F> ET scanSerial(ET* In, ET* Out, intT n, F f, ET zero) {
    return scanSerial(Out, (intT)0, n, f, getA<ET, intT>(In), zero, false, false);
}

// back indicates it runs in reverse direction
template<class ET, class intT, class F, class G>
ET scan(ET* Out, intT s, intT e, F f, G g, ET zero, bool inclusive, bool back) {
    intT n = e - s;
    intT l = nblocks(n, _SCAN_BSIZE);
    if (l <= 2) return scanSerial(Out, s, e, f, g, zero, inclusive, back);
    ET* Sums = newA(ET, nblocks(n, _SCAN_BSIZE));
    blocked_for(i, s, e, _SCAN_BSIZE, Sums[i] = reduceSerial<ET>(s, e, f, g););
    ET total = scan(Sums, (intT)0, l, f, getA<ET, intT>(Sums), zero, false, back);
    blocked_for(i, s, e, _SCAN_BSIZE, scanSerial(Out, s, e, f, g, Sums[i], inclusive, back););
    free(Sums);
    return total;
}

template<class ET, class intT, class F> ET scan(ET* In, ET* Out, intT n, F f, ET zero) {
    return scan(Out, (intT)0, n, f, getA<ET, intT>(In), zero, false, false);
}

template<class ET, class intT, class F> ET scanI(ET* In, ET* Out, intT n, F f, ET zero) {
    return scan(Out, (intT)0, n, f, getA<ET, intT>(In), zero, true, false);
}

template<class ET, class intT, class F> ET scanBack(ET* In, ET* Out, intT n, F f, ET zero) {
    return scan(Out, (intT)0, n, f, getA<ET, intT>(In), zero, false, true);
}

template<class ET, class intT, class F> ET scanIBack(ET* In, ET* Out, intT n, F f, ET zero) {
    return scan(Out, (intT)0, n, f, getA<ET, intT>(In), zero, true, true);
}

template<class ET, class intT> ET plusScan(ET* In, ET* Out, intT n) {
    return scan(Out, (intT)0, n, addF<ET>(), getA<ET, intT>(In), (ET)0, false, false);
}

#define _F_BSIZE (2 * _SCAN_BSIZE)

// sums a sequence of n boolean flags
// an optimized version that sums blocks of 4 booleans by treating
// them as an integer
// Only optimized when n is a multiple of 512 and Fl is 4byte aligned
template<class intT> intT sumFlagsSerial(bool* Fl, intT n) {
    intT r = 0;
    if (n >= 128 && (n & 511) == 0 && ((long)Fl & 3) == 0) {
        int* IFl = (int*)Fl;
        for (int k = 0; k < (n >> 9); k++) {
            int rr = 0;
            for (int j = 0; j < 128; j++) rr += IFl[j];
            r += (rr & 255) + ((rr >> 8) & 255) + ((rr >> 16) & 255) + ((rr >> 24) & 255);
            IFl += 128;
        }
    } else
        for (intT j = 0; j < n; j++) r += Fl[j];
    return r;
}

template<class ET, class intT, class F>
_seq<ET> packSerial(ET* Out, bool* Fl, intT s, intT e, F f) {
    if (Out == NULL) {
        intT m = sumFlagsSerial(Fl + s, e - s);
        Out = newA(ET, m);
    }
    intT k = 0;
    for (intT i = s; i < e; i++)
        if (Fl[i]) Out[k++] = f(i);
    return _seq<ET>(Out, k);
}

template<class ET, class intT, class F> _seq<ET> pack(ET* Out, bool* Fl, intT s, intT e, F f) {
    intT l = nblocks(e - s, _F_BSIZE);
    if (l <= 1) return packSerial(Out, Fl, s, e, f);
    intT* Sums = newA(intT, l);
    blocked_for(i, s, e, _F_BSIZE, Sums[i] = sumFlagsSerial(Fl + s, e - s););
    intT m = plusScan(Sums, Sums, l);
    if (Out == NULL) Out = newA(ET, m);
    blocked_for(i, s, e, _F_BSIZE, packSerial(Out + Sums[i], Fl, s, e, f););
    free(Sums);
    return _seq<ET>(Out, m);
}

template<class ET, class intT> intT pack(ET* In, ET* Out, bool* Fl, intT n) {
    return pack(Out, Fl, (intT)0, n, getA<ET, intT>(In)).n;
}

template<class intT> _seq<intT> packIndex(bool* Fl, intT n) {
    return pack((intT*)NULL, Fl, (intT)0, n, identityF<intT>());
}

template<class ET, class intT, class PRED> intT filter(ET* In, ET* Out, bool* Fl, intT n, PRED p) {
    parallel_for(intT i = 0; i < n; i++) Fl[i] = (bool)p(In[i]);
    intT m = pack(In, Out, Fl, n);
    return m;
}

template<class ET, class intT, class PRED> intT filter(ET* In, ET* Out, intT n, PRED p) {
    bool* Fl = newA(bool, n);
    intT m = filter(In, Out, Fl, n, p);
    free(Fl);
    return m;
}

}   // namespace sequence



// for compress
#define ONE_BYTE 256
#define TWO_BYTES 65536
#define THREE_BYTES 16777216
#define ONE_BYTE_SIGNED_MAX 128
#define ONE_BYTE_SIGNED_MIN -127
#define TWO_BYTES_SIGNED_MAX 32768
#define TWO_BYTES_SIGNED_MIN -32767
#define THREE_BYTES_SIGNED_MAX 8388608
#define THREE_BYTES_SIGNED_MIN -8388607

template<typename EdgeData>
void sort_adj_list(Bitmap** adj_bitmap, AdjUnit<EdgeData>** adj_list, EdgeId** adj_index,
                   int socket, int n) {
    int finished = 0;
    #pragma omp parallel for
    for (size_t j = 0; j < n; j++) {
        if (adj_bitmap[0]->get_bit(j)) {
            std::sort(&adj_list[0][adj_index[0][j]],
                      &adj_list[0][adj_index[0][j + 1]],
                      [](const AdjUnit<EdgeData>& a, const AdjUnit<EdgeData>& b) {
                          return a.neighbour < b.neighbour;
                      });
        }
    }
}

template<typename EdgeData>
void sort_adj_list(AdjUnit<EdgeData>** adj_list, CompressedAdjIndexUnit** compressed_adj_index,
                   VertexId* compressed_adj_vertices, int socket) {
#pragma omp parallel for
    for (size_t j = 0; j < compressed_adj_vertices[0]; j++) {
        std::sort(&adj_list[0][compressed_adj_index[0][j].index],
                  &adj_list[0][compressed_adj_index[0][j + 1].index],
                  [](const AdjUnit<EdgeData>& a, const AdjUnit<EdgeData>& b) {
                      return a.neighbour < b.neighbour;
                  });
    }
}

int numBytesSigned(int x) {
    if (x < ONE_BYTE_SIGNED_MAX && x > ONE_BYTE_SIGNED_MIN)
        return 1;
    else
        return 4;
}

EdgeId compress_first_edge(uint8_t* start, EdgeId offset, VertexId source, VertexId target) {
    uint8_t* saveStart = start;
    EdgeId saveOffset = offset;

    int32_t preCompress = (int32_t)target - source;
    int bytesUsed = 0;
    uint8_t firstByte = 0;
    int32_t toCompress = abs(preCompress);
    firstByte = toCompress & 0x3f;   // 0011|1111
    if (preCompress < 0) {
        firstByte |= 0x40;
    }
    toCompress = toCompress >> 6;
    if (toCompress > 0) {
        firstByte |= 0x80;
    }
    start[offset] = firstByte;
    offset++;

    uint8_t curByte = toCompress & 0x7f;
    while ((curByte > 0) || (toCompress > 0)) {
        bytesUsed++;
        uint8_t toWrite = curByte;
        toCompress = toCompress >> 7;
        // Check to see if there's any bits left to represent
        curByte = toCompress & 0x7f;
        if (toCompress > 0) {
            toWrite |= 0x80;
        }
        start[offset] = toWrite;
        offset++;
    }
    return offset;
}

template<typename EdgeData>
EdgeId compress_edges(uint8_t* start, long curOffset, AdjUnit<EdgeData>* savedEdges, VertexId edgeI,
                      int numBytes, int numBytesWeight, uint32_t runlength) {
    // header
    if constexpr (std::is_same<EdgeData, Empty>::value) {
        uint8_t header = numBytes - 1;
        header |= ((runlength - 1) << 2);
        start[curOffset++] = header;
        for (int i = 0; i < runlength; i++) {
            uint e = savedEdges[edgeI + i].neighbour - savedEdges[edgeI + i - 1].neighbour;
            int bytesUsed = 0;
            for (int j = 0; j < numBytes; j++) {
                uint8_t curByte = e & 0xff;
                e = e >> 8;
                start[curOffset++] = curByte;
                bytesUsed++;
            }
        }
        return curOffset;
    } else {
        // use 3 bits for info on bytes needed
        uint8_t header = numBytes - 1;
        if (numBytesWeight == 4) header |= 4;
        header |= ((runlength - 1) << 3);   // use 5 bits for run length
        start[curOffset++] = header;
        int bytesUsed = 0;

        for (int i = 0; i < runlength; i++) {
            uint e = savedEdges[edgeI + i].neighbour - savedEdges[edgeI + i - 1].neighbour;
            for (int j = 0; j < numBytes; j++) {
                uint8_t curByte = e & 0xff;
                e = e >> 8;
                start[curOffset++] = curByte;
                bytesUsed++;
            }
            int w = savedEdges[edgeI + i].edge_data;   // TODO
            uint wMag = abs(w);
            uint8_t curByte = wMag & 0x7f;

            wMag = wMag >> 7;
            if (w < 0)
                start[curOffset++] = curByte | 0x80;
            else
                start[curOffset++] = curByte;
            bytesUsed++;
            for (int j = 1; j < numBytesWeight; j++) {
                curByte = wMag & 0xff;
                wMag = wMag >> 8;
                start[curOffset++] = curByte;
                bytesUsed++;
            }
        }
    }
    return curOffset;
}

template<typename EdgeData>
long sequential_compress_edgeset(uint8_t* edgeArray, long currentOffset, VertexId degree,
                                 VertexId vertexNum, AdjUnit<EdgeData>* savedEdges) {
    if (degree > 0) {
        if constexpr (std::is_same<EdgeData, Empty>::value) {
            currentOffset =
                compress_first_edge(edgeArray, currentOffset, vertexNum, savedEdges[0].neighbour);
            if (degree == 1) return currentOffset;
            uint edgeI = 1;
            uint runlength = 0;
            int numBytes = 0;
            while (1) {
                uint difference = savedEdges[edgeI + runlength].neighbour -
                                  savedEdges[edgeI + runlength - 1].neighbour;
                if (difference < ONE_BYTE) {
                    if (!numBytes) {
                        numBytes = 1;
                        runlength++;
                    } else if (numBytes == 1)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(
                            edgeArray, currentOffset, savedEdges, edgeI, numBytes, 0, runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else if (difference < TWO_BYTES) {
                    if (!numBytes) {
                        numBytes = 2;
                        runlength++;
                    } else if (numBytes == 2)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(
                            edgeArray, currentOffset, savedEdges, edgeI, numBytes, 0, runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else if (difference < THREE_BYTES) {
                    if (!numBytes) {
                        numBytes = 3;
                        runlength++;
                    } else if (numBytes == 3)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(
                            edgeArray, currentOffset, savedEdges, edgeI, numBytes, 0, runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else {
                    if (!numBytes) {
                        numBytes = 4;
                        runlength++;
                    } else if (numBytes == 4)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(
                            edgeArray, currentOffset, savedEdges, edgeI, numBytes, 0, runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                }

                if (runlength == 64) {
                    currentOffset = compress_edges(
                        edgeArray, currentOffset, savedEdges, edgeI, numBytes, 0, runlength);
                    edgeI += runlength;
                    runlength = numBytes = 0;
                }

                if (runlength + edgeI == degree) {
                    currentOffset = compress_edges(
                        edgeArray, currentOffset, savedEdges, edgeI, numBytes, 0, runlength);
                    break;
                }
            }
        } else {
            currentOffset =
                compress_first_edge(edgeArray, currentOffset, vertexNum, savedEdges[0].neighbour);

            currentOffset =
                compress_first_edge(edgeArray, currentOffset, 0, savedEdges[0].edge_data);   // TODO
            if (degree == 1) return currentOffset;

            uint edgeI = 1;
            uint runlength = 0;
            int numBytes = 0, numBytesWeight = 0;
            while (1) {
                uint difference = savedEdges[edgeI + runlength].neighbour -
                                  savedEdges[edgeI + runlength - 1].neighbour;
                int weight = savedEdges[edgeI + runlength].edge_data;   // TODO
                if (difference < ONE_BYTE && numBytesSigned(weight) == 1) {
                    if (!numBytes) {
                        numBytes = 1;
                        numBytesWeight = 1;
                        runlength++;
                    } else if (numBytes == 1 && numBytesWeight == 1)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(edgeArray,
                                                       currentOffset,
                                                       savedEdges,
                                                       edgeI,
                                                       numBytes,
                                                       numBytesWeight,
                                                       runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else if (difference < ONE_BYTE && numBytesSigned(weight) == 4) {
                    if (!numBytes) {
                        numBytes = 1;
                        numBytesWeight = 4;
                        runlength++;
                    } else if (numBytes == 1 && numBytesWeight == 4)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(edgeArray,
                                                       currentOffset,
                                                       savedEdges,
                                                       edgeI,
                                                       numBytes,
                                                       numBytesWeight,
                                                       runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else if (difference < TWO_BYTES && numBytesSigned(weight) == 1) {
                    if (!numBytes) {
                        numBytes = 2;
                        numBytesWeight = 1;
                        runlength++;
                    } else if (numBytes == 2 && numBytesWeight == 1)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(edgeArray,
                                                       currentOffset,
                                                       savedEdges,
                                                       edgeI,
                                                       numBytes,
                                                       numBytesWeight,
                                                       runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else if (difference < TWO_BYTES && numBytesSigned(weight) == 4) {
                    if (!numBytes) {
                        numBytes = 2;
                        numBytesWeight = 4;
                        runlength++;
                    } else if (numBytes == 2 && numBytesWeight == 4)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(edgeArray,
                                                       currentOffset,
                                                       savedEdges,
                                                       edgeI,
                                                       numBytes,
                                                       numBytesWeight,
                                                       runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else if (difference < THREE_BYTES && numBytesSigned(weight) == 1) {
                    if (!numBytes) {
                        numBytes = 3;
                        numBytesWeight = 1;
                        runlength++;
                    } else if (numBytes == 3 && numBytesWeight == 1)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(edgeArray,
                                                       currentOffset,
                                                       savedEdges,
                                                       edgeI,
                                                       numBytes,
                                                       numBytesWeight,
                                                       runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else if (difference < THREE_BYTES && numBytesSigned(weight) == 4) {
                    if (!numBytes) {
                        numBytes = 3;
                        numBytesWeight = 4;
                        runlength++;
                    } else if (numBytes == 3 && numBytesWeight == 4)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(edgeArray,
                                                       currentOffset,
                                                       savedEdges,
                                                       edgeI,
                                                       numBytes,
                                                       numBytesWeight,
                                                       runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else if (numBytesSigned(weight) == 1) {
                    if (!numBytes) {
                        numBytes = 4;
                        numBytesWeight = 1;
                        runlength++;
                    } else if (numBytes == 4 && numBytesWeight == 1)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(edgeArray,
                                                       currentOffset,
                                                       savedEdges,
                                                       edgeI,
                                                       numBytes,
                                                       numBytesWeight,
                                                       runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                } else {
                    if (!numBytes) {
                        numBytes = 4;
                        numBytesWeight = 4;
                        runlength++;
                    } else if (numBytes == 4 && numBytesWeight == 4)
                        runlength++;
                    else {
                        // encode
                        currentOffset = compress_edges(edgeArray,
                                                       currentOffset,
                                                       savedEdges,
                                                       edgeI,
                                                       numBytes,
                                                       numBytesWeight,
                                                       runlength);
                        edgeI += runlength;
                        runlength = numBytes = 0;
                    }
                }

                if (runlength == 32) {
                    currentOffset = compress_edges(edgeArray,
                                                   currentOffset,
                                                   savedEdges,
                                                   edgeI,
                                                   numBytes,
                                                   numBytesWeight,
                                                   runlength);
                    edgeI += runlength;
                    runlength = numBytes = 0;
                }
                if (runlength + edgeI == degree) {
                    currentOffset = compress_edges(edgeArray,
                                                   currentOffset,
                                                   savedEdges,
                                                   edgeI,
                                                   numBytes,
                                                   numBytesWeight,
                                                   runlength);
                    break;
                }
            }
        }
    }
    return currentOffset;
}

/*
adj_list: the adjacency list to compress
adj_index: the index of the adjacency list
n: number of vertices
adj_edges: the edges in the adjacency list
adj_degree: the degree of each vertex
socket: the number of sockets
*/
template<typename EdgeData>
uint8_t** parallel_compress_edges(Bitmap** adj_bitmap, AdjUnit<EdgeData>** adj_list, EdgeId** adj_index, VertexId n,
                                  EdgeId* adj_edges, int socket, size_t* return_space) {
    uint8_t** finalArr = newA(uint8_t*, socket);
    for (int s_i = 0; s_i < socket; s_i++) {
        uint32_t** edgePts = newA(uint32_t*, n);
        uint64_t* charsUsedArr = newA(uint64_t, n);
        EdgeId* compressionStarts = newA(EdgeId, n + 1);
        {
            parallel_for(long i = 0; i < n; i++) {
                charsUsedArr[i] =
                    sizeof(AdjUnit<EdgeData>) *
                    (ceil(((adj_index[s_i][i + 1] - adj_index[s_i][i]) * 9) / 8) + 4);   // todo
            }
        }
        long toAlloc = sequence::plusScan(charsUsedArr, charsUsedArr, n);
        // long toAlloc = std::reduce(std::execution::par,   // 并行执行
        //                            charsUsedArr,          // 指向数组开头的指针
        //                            charsUsedArr + n,      // 指向数组结尾的指针
        //                            (uint64_t)0);          // 初始值
        // std::exclusive_scan(std::execution::par,          // 并行执行
        //                     charsUsedArr,                 // 输入范围开始
        //                     charsUsedArr + n,             // 输入范围结束
        //                     charsUsedArr,                 // 输出范围开始 (就地操作)
        //                     (uint64_t)0);                 // 初始偏移为 0

        uint32_t* iEdges = newA(uint32_t, toAlloc);
        if (iEdges == NULL) {
            cout << "Error allocating memory for iEdges" << endl;
            exit(1);
        }
        {
            parallel_for(long i = 0; i < n; i++) {
                edgePts[i] = iEdges + charsUsedArr[i];
                if (adj_bitmap[s_i]->get_bit(i) == false) {
                    // if the vertex is not present in the bitmap, set charsUsed to 0
                    charsUsedArr[i] = 0;
                    continue;
                }
                long charsUsed =
                    sequential_compress_edgeset((uint8_t*)(iEdges + charsUsedArr[i]),
                                                0,
                                                (adj_index[s_i][i + 1] - adj_index[s_i][i]),
                                                i,
                                                adj_list[s_i] + adj_index[s_i][i]);
                charsUsedArr[i] = charsUsed;
            }
        }

        // produce the total space needed for all compressed lists in chars.
        long totalSpace = sequence::plusScan(charsUsedArr, compressionStarts, n);
        // long totalSpace =
        //     std::reduce(std::execution::par, charsUsedArr, charsUsedArr + n, (uint64_t)0);
        // std::exclusive_scan(std::execution::par,
        //                     charsUsedArr,        // 输入范围开始
        //                     charsUsedArr + n,    // 输入范围结束
        //                     compressionStarts,   // <-- 输出范围开始 (不同的数组)
        //                     (uint64_t)0);

        compressionStarts[n] = totalSpace;
        free(charsUsedArr);

        finalArr[s_i] = newA(uint8_t, totalSpace);
        return_space[s_i] = totalSpace;
        cout << "total space requested is : " << totalSpace << endl;

        {
            parallel_for(long i = 0; i < n; i++) {
                long o = compressionStarts[i];
                memcpy(finalArr[s_i] + o, (uint8_t*)(edgePts[i]), compressionStarts[i + 1] - o);
                adj_index[s_i][i] = o;
            }
        }
        adj_index[s_i][n] = totalSpace;

        free(iEdges);
        free(edgePts);
        free(compressionStarts);
        cout << "finished compressing, bytes used = " << totalSpace << endl;
        cout << "would have been, " << (adj_edges[s_i] * sizeof(AdjUnit<EdgeData>)) << endl;
    }
    return finalArr;
}

/*
adj_list: the adjacency list to compress
compressed_adj_index: the compressed adjacency index
n: number of vertices
adj_edges: the edges in the adjacency list
adj_degree: the degree of each vertex
socket: the number of sockets
*/
template<typename EdgeData>
uint8_t** parallel_compress_edges(AdjUnit<EdgeData>** adj_list,
                                  CompressedAdjIndexUnit** compressed_adj_index,
                                  VertexId* compressed_adj_vertices, EdgeId* adj_edges, int socket,
                                  size_t* return_space) {
    uint8_t** finalArr = newA(uint8_t*, socket);
    for (int s_i = 0; s_i < socket; s_i++) {
        int n = compressed_adj_vertices[s_i];
        uint32_t** edgePts = newA(uint32_t*, n);
        uint64_t* charsUsedArr = newA(uint64_t, n);
        EdgeId* compressionStarts = newA(EdgeId, n + 1);
        {
            parallel_for(long i = 0; i < n; i++) {
                charsUsedArr[i] = sizeof(AdjUnit<EdgeData>) / 2 *
                                  (ceil(((compressed_adj_index[s_i][i + 1].index -
                                          compressed_adj_index[s_i][i].index) *
                                         9) /
                                        8) +
                                   4);   // todo
            }
        }
        long toAlloc = sequence::plusScan(charsUsedArr, charsUsedArr, n);
        uint32_t* iEdges = newA(uint32_t, toAlloc);
        {
            parallel_for(long i = 0; i < n; i++) {
                edgePts[i] = iEdges + charsUsedArr[i];
                long charsUsed = sequential_compress_edgeset(
                    (uint8_t*)(iEdges + charsUsedArr[i]),
                    0,
                    (compressed_adj_index[s_i][i + 1].index - compressed_adj_index[s_i][i].index),
                    compressed_adj_index[s_i][i].vertex,
                    adj_list[s_i] + compressed_adj_index[s_i][i].index);
                charsUsedArr[i] = charsUsed;
            }
        }

        // produce the total space needed for all compressed lists in chars.
        long totalSpace = sequence::plusScan(charsUsedArr, compressionStarts, n);
        compressionStarts[n] = totalSpace;
        free(charsUsedArr);

        finalArr[s_i] = newA(uint8_t, totalSpace);
        return_space[s_i] = totalSpace;
        cout << "total space requested is : " << totalSpace << endl;

        {
            parallel_for(long i = 0; i < n; i++) {
                long o = compressionStarts[i];
                memcpy(finalArr[s_i] + o, (uint8_t*)(edgePts[i]), compressionStarts[i + 1] - o);
                compressed_adj_index[s_i][i].index = o;
            }
        }
        compressed_adj_index[s_i][n].index = totalSpace;

        free(iEdges);
        free(edgePts);
        free(compressionStarts);
        cout << "finished compressing, bytes used = " << totalSpace << endl;
        cout << "would have been, " << (adj_edges[s_i] * sizeof(AdjUnit<EdgeData>)) << endl;
    }
    return finalArr;
}


/*
adj_list: the adjacency list to compress
compressed_adj_index: the compressed adjacency index
n: number of vertices
adj_edges: the edges in the adjacency list
adj_degree: the degree of each vertex
socket: the number of sockets
*/
template<typename EdgeData>
uint8_t** parallel_compress_edges(AdjUnit<EdgeData>** adj_list, EdgeId** adj_index,
                                  CompressedAdjIndexUnit** compressed_adj_index,
                                  VertexId* compressed_adj_vertices, EdgeId* adj_edges, int socket,
                                  size_t* return_space) {
    uint8_t** finalArr = newA(uint8_t*, socket);
    for (int s_i = 0; s_i < socket; s_i++) {
        int n = compressed_adj_vertices[s_i];
        uint32_t** edgePts = newA(uint32_t*, n);
        uint64_t* charsUsedArr = newA(uint64_t, n);
        EdgeId* compressionStarts = newA(EdgeId, n + 1);
        {
            parallel_for(long i = 0; i < n; i++) {
                charsUsedArr[i] = sizeof(AdjUnit<EdgeData>) / 2 *
                                  (ceil(((compressed_adj_index[s_i][i + 1].index -
                                          compressed_adj_index[s_i][i].index) *
                                         9) /
                                        8) +
                                   4);   // todo
            }
        }
        long toAlloc = sequence::plusScan(charsUsedArr, charsUsedArr, n);
        uint32_t* iEdges = newA(uint32_t, toAlloc);
        {
            parallel_for(long i = 0; i < n; i++) {
                edgePts[i] = iEdges + charsUsedArr[i];
                long charsUsed = sequential_compress_edgeset(
                    (uint8_t*)(iEdges + charsUsedArr[i]),
                    0,
                    (compressed_adj_index[s_i][i + 1].index - compressed_adj_index[s_i][i].index),
                    compressed_adj_index[s_i][i].vertex,
                    adj_list[s_i] + compressed_adj_index[s_i][i].index);
                charsUsedArr[i] = charsUsed;
            }
        }

        // produce the total space needed for all compressed lists in chars.
        long totalSpace = sequence::plusScan(charsUsedArr, compressionStarts, n);
        compressionStarts[n] = totalSpace;
        free(charsUsedArr);

        finalArr[s_i] = newA(uint8_t, totalSpace);
        return_space[s_i] = totalSpace;
        cout << "total space requested is : " << totalSpace << endl;
        {
            parallel_for(long i = 0; i < n; i++) {
                long o = compressionStarts[i];
                memcpy(finalArr[s_i] + o, (uint8_t*)(edgePts[i]), compressionStarts[i + 1] - o);
                adj_index[s_i][compressed_adj_index[s_i][i].vertex]= o;
                adj_index[s_i][compressed_adj_index[s_i][i].vertex + 1] = compressionStarts[i + 1];
            }
        }
        free(iEdges);
        free(edgePts);
        free(compressionStarts);
        cout << "finished compressing, bytes used = " << totalSpace << endl;
        cout << "would have been, " << (adj_edges[s_i] * sizeof(AdjUnit<EdgeData>)) << endl;
    }
    return finalArr;
}


inline int decode_weight(uint8_t*& start) {
    uint8_t fb = *start++;
    int edgeRead = (fb & 0x3f);
    if (LAST_BIT_SET(fb)) {
        int shiftAmount = 6;
        while (1) {
            uint8_t b = *start;
            edgeRead |= ((b & 0x7f) << shiftAmount);
            start++;
            if (LAST_BIT_SET(b))
                shiftAmount += EDGE_SIZE_PER_BYTE;
            else
                break;
        }
    }
    return (fb & 0x40) ? -edgeRead : edgeRead;
}

inline VertexId decode_first_edge(uint8_t*& start, VertexId source) {
    uint8_t fb = *start++;
    // int sign = (fb & 0x40) ? -1 : 1;
    VertexId edgeRead = (fb & 0x3f);
    if (LAST_BIT_SET(fb)) {
        int shiftAmount = 6;
        // shiftAmount += 6;
        while (1) {
            uint8_t b = *start;
            edgeRead |= ((b & 0x7f) << shiftAmount);
            start++;
            if (LAST_BIT_SET(b))
                shiftAmount += EDGE_SIZE_PER_BYTE;
            else
                break;
        }
    }
    // edgeRead *= sign;
    return (fb & 0x40) ? source - edgeRead : source + edgeRead;
}

template<typename EdgeData, typename OPT>
void decode(OPT opt, uint8_t* edgeStart, const VertexId& source, const uint& degree,
            const bool par = true) {
    uint edgesRead = 0;
    if (degree > 0) {
        // Eat first edge, which is compressed specially
        VertexId startEdge = decode_first_edge(edgeStart, source);
        // printf("startEdge: %d\n", startEdge);
        if constexpr (std::is_same<EdgeData, Empty>::value) {
            if (!opt(source, startEdge, 0, edgesRead)) {
                return;
            }
        } else {
            int weight = decode_weight(edgeStart);
            if (!opt(source, startEdge, weight, edgesRead)) {
                return;
            }
        }
        uint i = 0;
        edgesRead = 1;
        if constexpr (std::is_same<EdgeData, Empty>::value) {
            while (1) {
                // printf("edgesRead: %d, degree: %d\n", edgesRead, degree);
                if (edgesRead == degree) return;
                                  
                uint8_t header = edgeStart[i++];
                uint numbytes = 1 + (header & 0x3);
                uint runlength = 1 + (header >> 2);
                // printf("numbytes: %d, runlength: %d\n", numbytes, runlength);
                switch (numbytes) {
                case 1:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge = (uint)edgeStart[i++] + startEdge;
                        startEdge = edge;
                        if (!opt(source, edge, 0, edgesRead++)) {
                            return;
                        }
                    }
                    break;
                case 2:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge =
                            (uint)edgeStart[i] + (((uint)edgeStart[i + 1]) << 8) + startEdge;
                        i += 2;
                        startEdge = edge;
                        if (!opt(source, edge, 0, edgesRead++)) {
                            return;
                        }
                    }
                    break;
                case 3:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge = (uint)edgeStart[i] + (((uint)edgeStart[i + 1]) << 8) +
                                    (((uint)edgeStart[i + 2]) << 16) + startEdge;
                        i += 3;
                        startEdge = edge;
                        if (!opt(source, edge, 0, edgesRead++)) {
                            return;
                        }
                    }
                    break;
                default:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge = (uint)edgeStart[i] + (((uint)edgeStart[i + 1]) << 8) +
                                    (((uint)edgeStart[i + 2]) << 16) +
                                    (((uint)edgeStart[i + 3]) << 24) + startEdge;
                        i += 4;
                        startEdge = edge;
                        if (!opt(source, edge, 0, edgesRead++)) {
                            return;
                        }
                    }
                }
            }
        } else {
            while (1) {
                if (edgesRead == degree) return;
                uint8_t header = edgeStart[i++];
                uint info = header & 0x7;   // 3 bits for info
                uint runlength = 1 + (header >> 3);
                switch (info) {
                case 0:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge = (uint)edgeStart[i] + startEdge;
                        startEdge = edge;
                        uint w = edgeStart[i + 1];   // highest bit is sign bit
                        int weight = (w & 0x80) ? -(w & 0x7f) : (w & 0x7f);
                        i += 2;
                        if (!opt(source, edge, weight, edgesRead++)) {
                            return;
                        }
                    }
                    break;
                case 1:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge =
                            (uint)edgeStart[i] + (((uint)edgeStart[i + 1]) << 8) + startEdge;
                        startEdge = edge;
                        uint w = edgeStart[i + 2];   // highest bit is sign bit
                        int weight = (w & 0x80) ? -(w & 0x7f) : (w & 0x7f);
                        i += 3;
                        if (!opt(source, edge, weight, edgesRead++)) {
                            return;
                        }
                    }
                    break;
                case 2:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge = (uint)edgeStart[i] + (((uint)edgeStart[i + 1]) << 8) +
                                    (((uint)edgeStart[i + 2]) << 16) + startEdge;
                        startEdge = edge;
                        uint w = edgeStart[i + 3];   // highest bit is sign bit
                        int weight = (w & 0x80) ? -(w & 0x7f) : (w & 0x7f);
                        i += 4;
                        if (!opt(source, edge, weight, edgesRead++)) {
                            return;
                        }
                    }
                    break;
                case 3:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge = (uint)edgeStart[i] + (((uint)edgeStart[i + 1]) << 8) +
                                    (((uint)edgeStart[i + 2]) << 16) +
                                    (((uint)edgeStart[i + 3]) << 24) + startEdge;
                        startEdge = edge;
                        uint w = edgeStart[i + 4];   // highest bit is sign bit
                        int weight = (w & 0x80) ? -(w & 0x7f) : (w & 0x7f);
                        i += 5;
                        if (!opt(source, edge, weight, edgesRead++)) {
                            return;
                        }
                    }
                    break;
                case 4:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge = (uint)edgeStart[i] + startEdge;
                        startEdge = edge;
                        uint w = edgeStart[i + 1];   // highest bit is sign bit
                        int weight = (w & 0x7f) + (((uint)edgeStart[i + 2]) << 7) +
                                     (((uint)edgeStart[i + 3]) << 15) +
                                     (((uint)edgeStart[i + 4]) << 23);
                        if (w & 0x80) weight = -weight;
                        i += 5;
                        if (!opt(source, edge, weight, edgesRead++)) {
                            return;
                        }
                    }
                    break;
                case 5:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge =
                            (uint)edgeStart[i] + (((uint)edgeStart[i + 1]) << 8) + startEdge;
                        startEdge = edge;
                        uint w = edgeStart[i + 2];   // highest bit is sign bit
                        int weight = (w & 0x7f) + (((uint)edgeStart[i + 3]) << 7) +
                                     (((uint)edgeStart[i + 4]) << 15) +
                                     (((uint)edgeStart[i + 5]) << 23);
                        if (w & 0x80) weight = -weight;
                        i += 6;
                        if (!opt(source, edge, weight, edgesRead++)) {
                            return;
                        }
                    }
                    break;
                case 6:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge = (uint)edgeStart[i] + (((uint)edgeStart[i + 1]) << 8) +
                                    (((uint)edgeStart[i + 2]) << 16) + startEdge;
                        startEdge = edge;
                        uint w = edgeStart[i + 3];   // highest bit is sign bit
                        int weight = (w & 0x7f) + (((uint)edgeStart[i + 4]) << 7) +
                                     (((uint)edgeStart[i + 5]) << 15) +
                                     (((uint)edgeStart[i + 6]) << 23);
                        if (w & 0x80) weight = -weight;
                        i += 7;
                        if (!opt(source, edge, weight, edgesRead++)) {
                            return;
                        }
                    }
                default:
                    for (uint j = 0; j < runlength; j++) {
                        uint edge = (uint)edgeStart[i] + (((uint)edgeStart[i + 1]) << 8) +
                                    (((uint)edgeStart[i + 2]) << 16) +
                                    (((uint)edgeStart[i + 3]) << 24) + startEdge;
                        startEdge = edge;
                        uint w = edgeStart[i + 4];   // highest bit is sign bit
                        int weight = (w & 0x7f) + (((uint)edgeStart[i + 5]) << 7) +
                                     (((uint)edgeStart[i + 6]) << 15) +
                                     (((uint)edgeStart[i + 7]) << 23);
                        if (w & 0x80) weight = -weight;
                        i += 8;
                        if (!opt(source, edge, weight, edgesRead++)) {
                            return;
                        }
                    }
                }
            }
        }
    }
}



#endif