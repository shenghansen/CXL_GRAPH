#ifndef COMPRESS_HPP
#define COMPRESS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <fstream>
#include <iostream>
#include <numeric>

#include <stdio.h>
#include <string.h>

#include "core/type.hpp"

using namespace std;

#define newA(__E, __n) (__E*)malloc((__n) * sizeof(__E))
#define new_remote(__E, __n, __node) (__E*)numa_alloc_onnode((__n) * sizeof(__E), __node)
#define parallel_for _Pragma("omp parallel for") for

#define LAST_BIT_SET(b) (b & (0x80))
#define EDGE_SIZE_PER_BYTE 7

#define PARALLEL_DEGREE 1000

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
void sort_adj_list(AdjUnit<EdgeData>** adj_list, EdgeId** adj_index, int socket, int n) {
    for (size_t i = 0; i < socket; i++) {
        for (size_t j = 0; j < n; j++) {
            std::sort(&adj_list[i][adj_index[i][j]],
                      &adj_list[i][adj_index[i][j + 1]],
                      [](const AdjUnit<EdgeData>& a, const AdjUnit<EdgeData>& b) {
                          return a.neighbour < b.neighbour;
                      });
        }
    }
}

template<typename EdgeData>
void sort_adj_list(AdjUnit<EdgeData>** adj_list, CompressedAdjIndexUnit** compressed_adj_index,
                   VertexId* compressed_adj_vertices, int socket) {
    for (size_t i = 0; i < socket; i++) {
        for (size_t j = 0; j < compressed_adj_vertices[i]; j++) {
            std::sort(&adj_list[i][compressed_adj_index[i][j].index],
                      &adj_list[i][compressed_adj_index[i][j + 1].index],
                      [](const AdjUnit<EdgeData>& a, const AdjUnit<EdgeData>& b) {
                          return a.neighbour < b.neighbour;
                      });
        }
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
uint8_t** parallel_compress_edges(AdjUnit<EdgeData>** adj_list, EdgeId** adj_index, VertexId n,
                                  EdgeId* adj_edges, int socket, size_t* return_space) {
    uint8_t** finalArr = newA(uint8_t*, socket);
    for (int s_i = 0; s_i < socket; s_i++) {
        uint32_t** edgePts = newA(uint32_t*, n);
        uint64_t* charsUsedArr = newA(uint64_t, n);
        EdgeId* compressionStarts = newA(EdgeId, n + 1);
        {
            parallel_for(long i = 0; i < n; i++) {
                charsUsedArr[i] =
                    sizeof(AdjUnit<EdgeData>) / 2 *
                    (ceil(((adj_index[s_i][i + 1] - adj_index[s_i][i]) * 9) / 8) + 4);   // todo
            }
        }
        // long toAlloc = sequence::plusScan(charsUsedArr, charsUsedArr, n);
        long toAlloc = std::reduce(std::execution::par,   // 并行执行
                                   charsUsedArr,          // 指向数组开头的指针
                                   charsUsedArr + n,      // 指向数组结尾的指针
                                   (uint64_t)0);          // 初始值
        std::exclusive_scan(std::execution::par,          // 并行执行
                            charsUsedArr,                 // 输入范围开始
                            charsUsedArr + n,             // 输入范围结束
                            charsUsedArr,                 // 输出范围开始 (就地操作)
                            (uint64_t)0);                 // 初始偏移为 0

        uint32_t* iEdges = newA(uint32_t, toAlloc);
        {
            parallel_for(long i = 0; i < n; i++) {
                edgePts[i] = iEdges + charsUsedArr[i];
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
        // long totalSpace = sequence::plusScan(charsUsedArr, compressionStarts, n);
        long totalSpace =
            std::reduce(std::execution::par, charsUsedArr, charsUsedArr + n, (uint64_t)0);
        std::exclusive_scan(std::execution::par,
                            charsUsedArr,        // 输入范围开始
                            charsUsedArr + n,    // 输入范围结束
                            compressionStarts,   // <-- 输出范围开始 (不同的数组)
                            (uint64_t)0);
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
        // long toAlloc = sequence::plusScan(charsUsedArr, charsUsedArr, n);
        long toAlloc = std::reduce(std::execution::par,   // 并行执行
                                   charsUsedArr,          // 指向数组开头的指针
                                   charsUsedArr + n,      // 指向数组结尾的指针
                                   (uint64_t)0);          // 初始值
        std::exclusive_scan(std::execution::par,          // 并行执行
                            charsUsedArr,                 // 输入范围开始
                            charsUsedArr + n,             // 输入范围结束
                            charsUsedArr,                 // 输出范围开始 (就地操作)
                            (uint64_t)0);                 // 初始偏移为 0

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
        // long totalSpace = sequence::plusScan(charsUsedArr, compressionStarts, n);
        long totalSpace =
            std::reduce(std::execution::par, charsUsedArr, charsUsedArr + n, (uint64_t)0);
        std::exclusive_scan(std::execution::par,
                            charsUsedArr,        // 输入范围开始
                            charsUsedArr + n,    // 输入范围结束
                            compressionStarts,   // <-- 输出范围开始 (不同的数组)
                            (uint64_t)0);
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
                if (edgesRead == degree) return;
                uint8_t header = edgeStart[i++];
                uint numbytes = 1 + (header & 0x3);
                uint runlength = 1 + (header >> 2);
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