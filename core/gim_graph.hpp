/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef GRAPH_HPP
#define GRAPH_HPP
#if defined(SPARSE_MODE_UNIDIRECTIONAL) || defined(DENSE_MODE_UNIDIRECTIONAL)
#    define UNIDIRECTIONAL_MODE
#endif

#include <atomic>
#include <cstddef>
#include <fcntl.h>
#include <malloc.h>
#include <numa.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bitmap.hpp"
#include "communicate.h"
#include "core/atomic.hpp"
#include "core/bitmap.hpp"
#include "core/constants.hpp"
#include "core/filesystem.hpp"
#include "core/mpi.hpp"
#include "core/time.hpp"
#include "core/type.hpp"
#include "mpi.h"



#define THREDAS 32
#define NUMA SNC
#define REMOTE_NUMA 7

#define CAPACITY (1ull * 1024 * 1024 * 1024)

// #define  PRINT_DEBUG_MESSAGES 1
double total_process_time = 0;
bool is_first = true;
double process_edge_time[4] = {0};
bool waiting = false;

void check_blocking(std::mutex& recv_queue_mutex, int& recv_queue_size, int step) {
    waiting = true;
    while (true) {
        recv_queue_mutex.lock();
        bool condition = (recv_queue_size <= step);   // 当前分区接受完成才继续后面的计算
        recv_queue_mutex.unlock();
        // printf("recv_queue_size=%d\n", recv_queue_size);
        if (!condition) break;
        // __asm volatile("pause" ::: "memory");
        std::this_thread::yield();
    }
    waiting = false;
}

enum ThreadStatus { WORKING, STEALING };

enum MessageTag { ShuffleGraph, PassMessage, GatherVertexArray };

struct ThreadState {
    VertexId curr;
    VertexId end;
    ThreadStatus status;
};
/* origin MesaageBUffer */
struct MessageBuffer {
    size_t capacity;
    int count;   // the actual size (i.e. bytes) should be sizeof(element) * count
    char* data;
    MessageBuffer() {
        capacity = 0;
        count = 0;
        data = NULL;
    }
    void init(int socket_id) {
        capacity = 1024 * 1024;
        count = 0;
        data = (char*)numa_alloc_onnode(capacity, socket_id);
    }
    void resize(size_t new_capacity) {
        if (new_capacity > capacity) {
            char* new_data = (char*)numa_realloc(data, capacity, new_capacity);
            assert(new_data != NULL);
            data = new_data;
            capacity = new_capacity;
        }
    }
};

/* gim version */
struct GIMMessageBuffer {
    int count;
    CXL_SHM* cxl_shm;

    int host_id;
    int numa_id;
    size_t capacity;
    char* data;
    GIMMessageBuffer(CXL_SHM* cxl_shm, int host_id, int numa_id) {
        this->cxl_shm = cxl_shm;
        this->host_id = host_id;
        this->numa_id = numa_id;
        capacity = 1024 * 1024;
        count = 0;
        // data = (char*)cxl_shm->GIM_malloc(capacity, host_id, numa_id);
    }
    void init(size_t size) {
        capacity = size;
        // data = (char*)malloc(capacity);
        data = (char*)cxl_shm->GIM_malloc(capacity, host_id, numa_id);
    }
    void resize(size_t new_capacity) {
        if (new_capacity > capacity) {
            printf("resize\n\n");
            ERROR("out of capacity");
            char* new_data = (char*)cxl_shm->GIM_malloc(new_capacity * 10, host_id, numa_id);
            // char* new_data = (char*)malloc(new_capacity);
            assert(new_data != NULL);
            data = new_data;
            capacity = new_capacity;
        }
    }
};

template<typename MsgData> struct MsgUnit {
    VertexId vertex;
    MsgData msg_data;
} __attribute__((packed));

template<typename EdgeData = Empty> class Graph {
    /* 后面注释带partitions的就是全局多节点数据，只带threads的就是节点内多线程数据 */
public:
    std::unordered_map<std::string, size_t> data_size;
    int partition_id;   // 自己所属的分区id
    int partitions;     // 总分区数

    size_t alpha;   // 分块系数

    int threads;              // 单节点内总线程数
    int sockets;              // socket数
    int threads_per_socket;   // 每个socket的线程数

    size_t edge_data_size;   // edge_data的大小
    size_t unit_size;        // adj_unit的大小
    size_t edge_unit_size;   // edge_unit的大小

    bool symmetric;      // 图是否对称?无向图
    VertexId vertices;   // 点数
    EdgeId edges;        // 边数
    VertexId*
        out_degree;   // VertexId [vertices]; numa-interleaved 稠密模式下每个host内master点的出度
    VertexId*
        in_degree;   // VertexId [vertices]; numa-interleaved  稀疏模式下每个host内master点的入度

    VertexId*
        partition_offset;   // VertexId [partitions+1] 记录每个分区所在的索引,即每个分区起始顶点的ID
    // host local data
    VertexId*
        local_partition_offset;   // VertexId [sockets+1]  记录本地分区的索引,即本地分区起始顶点的ID

    VertexId owned_vertices;   // 本分区拥有多少点
    EdgeId* outgoing_edges;    // EdgeId [sockets] 稀疏模式下每个socket内出边总数
    EdgeId* incoming_edges;    // EdgeId [sockets] 稠密模式下每个socket内入边总数

    // graph data
    Bitmap** incoming_adj_bitmap;   //  [socket][vertices]   每个 socket内的每个点是否有入边
    EdgeId** incoming_adj_index;             // EdgeId [sockets] [vertices+1]; numa-aware
                                             // 每个socket内每个点入边的索引
    AdjUnit<EdgeData>** incoming_adj_list;   // AdjUnit<EdgeData> [sockets] [该socket内出边总数];
                                             // numa-aware  每个socket内每条入边的src

    Bitmap** outgoing_adj_bitmap;   // [socket][vertices]   每个 socket内的每个点是否有出边
    EdgeId** outgoing_adj_index;   // EdgeId [sockets] [vertices+1]; numa-aware
                                   // 每个socket内每个点出边的索引
    AdjUnit<EdgeData>**
        outgoing_adj_list;   // AdjUnit<EdgeData> [sockets] [该socket点的内出边总数]; numa-aware
                             // 每个socket内每条出边的dst
    // 压缩 graph data
    VertexId*
        compressed_incoming_adj_vertices;   // VertexId[socket] 稀疏模式下每个socket有入边的点的数量
    CompressedAdjIndexUnit**
        compressed_incoming_adj_index;   // CompressedAdjIndexUnit [sockets][有出边的mirror点数+1];
                                         // numa-aware
                                         // CompressedAdjIndexUnit的index用来表示入边的索引，即在
                                         // ingoing_adj_list 邻接表中入边的起始位置或者结束位置
                                         // vertex用来表示入边的dst，即哪些点有入边
    VertexId*
        compressed_outgoing_adj_vertices;   // VertexId[socket] 稀疏模式下每个socket有出边的点的数量
    CompressedAdjIndexUnit**
        compressed_outgoing_adj_index;   // CompressedAdjIndexUnit [sockets][有出边的点数+1];
                                         // numa-aware
                                         // CompressedAdjIndexUnit的index用来表示出边的索引，即在
                                         // outgoing_adj_list 邻接表中出边的起始位置或者结束位置
                                         // vertex用来表示出边的src，即哪些点有出边

    ThreadState** thread_state;   // ThreadState* [threads]; numa-aware
    // 在tune_chunk中初始化 在process_vertices中使用
    // 设置两个tuned_chunks的意义是为了应对transpose操作，不transpose就是tuned_chunks_dense
    //  transpose就是tuned_chunks_sparse
    ThreadState** tuned_chunks_dense;    // ThreadState [partitions][threads];
    ThreadState** tuned_chunks_sparse;   // ThreadState [partitions][threads];

    // 每个线程自己的发送缓冲，先写到这里再一下刷到全局的send_buffer中，这也是利用局部性
    size_t local_send_buffer_limit;
    MessageBuffer** local_send_buffer;   // MessageBuffer* [threads]; numa-aware

    int current_send_part_id;
    // MessageBuffer的二维数组，MessageBuffer是自定义数组，初始化时不分配空间，通过resize函数numa_aware分配空间
    MessageBuffer*** send_buffer;   // MessageBuffer* [partitions] [sockets]; numa-aware
    MessageBuffer*** recv_buffer;   // MessageBuffer* [partitions] [sockets]; numa-aware

    // GIM
    CXL_SHM* cxl_shm;
    GIM_comm* gim_comm;
    /* buffer */
    GIMMessageBuffer****
        gim_send_buffer;   // MessageBuffer* [partitions][partitions] [sockets]; numa-aware
    GIMMessageBuffer****
        gim_recv_buffer;   // MessageBuffer* [partitions][partitions] [sockets]; numa-aware
                           /* thread state */
    ThreadState*** gim_thread_state;
    /* edge */
    Bitmap*** gim_incoming_adj_bitmap;
    EdgeId*** gim_incoming_adj_index;
    AdjUnit<EdgeData>*** gim_incoming_adj_list;

    Bitmap*** gim_outgoing_adj_bitmap;
    EdgeId*** gim_outgoing_adj_index;
    AdjUnit<EdgeData>*** gim_outgoing_adj_list;
    /*     压缩 graph data */
    CompressedAdjIndexUnit*** gim_compressed_incoming_adj_index;
    CompressedAdjIndexUnit*** gim_compressed_outgoing_adj_index;
    /*  current_send_part_id*/
    std::atomic<int>* global_current_send_part_id;
    std::atomic<int>* stealings;
    int* stealingss;
    /* degree */
    VertexId** gim_out_degree;
    VertexId** gim_in_degree;
    /* global stealing */
    size_t*** send_count;

    /* single comm*/
    std::atomic<bool>**** completion_tags;   //  bool* [partitions][partitions][sockets]; numa-aware
    size_t*** length_array;

    Graph() {
        // threads = numa_num_configured_cpus();
        // sockets = numa_num_configured_nodes();
        // threads_per_socket = threads / sockets;
        // for simulate
        threads = THREDAS;
        sockets = NUMA;
        threads_per_socket = threads / sockets;
        init();
    }
    // for simulate
    inline int get_real_numa_id(int thread_id, int partition_id) {
        return (thread_id / threads_per_socket) + (partition_id * sockets);
    }
    inline int get_logical_thread_id() {
        int physical_thread_id = omp_get_thread_num();
        return physical_thread_id % (partition_id * threads);
    }
    void set_thread_affinity(int thread_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
    int get_thread_core_id() {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        // 获取当前线程的 CPU 亲和性
        pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        // 遍历并返回当前线程所在的物理核心
        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &cpuset)) {
                return i;   // 返回当前线程所在的物理核心编号
            }
        }
        return -1;   // 如果没有找到，返回 -1（错误标志）
    }
    // 根据threadid得知在哪个numa节点上
    inline int get_socket_id(int thread_id) { return thread_id / threads_per_socket; }
    // 根据threadid得知在numa上的第几个thread，即offset
    inline int get_socket_offset(int thread_id) { return thread_id % threads_per_socket; }

    int get_partition_id() { return partition_id; }
    // for data statistics
    void printf_data(std::string data_name, size_t size, size_t total_size) {
        printf("%s size: ", data_name.c_str());
        double size_in_gb = static_cast<double>(size) / (1024 * 1024 * 1024);
        double size_in_mb = static_cast<double>(size) / (1024 * 1024);
        double size_in_kb = static_cast<double>(size) / 1024;
        if (size_in_gb >= 1.0) {
            printf("%.2f GB", size_in_gb);
        } else if (size_in_mb >= 1.0) {
            printf("%.2f MB", size_in_mb);
        } else if (size_in_kb >= 1.0) {
            printf("%.2f KB", size_in_kb);
        } else {
            printf("%zu B", size);
        }
        printf(" , percentage : %lf%\n", double(size) / total_size * 100);
    }

    void printf_data_info() {
        data_size["out_degree"] = vertices * sizeof(VertexId);
        data_size["in_degree"] = vertices * sizeof(VertexId);
        data_size["partition_offset"] = (partitions + 1) * sizeof(VertexId);
        data_size["local_partition_offset"] = (sockets + 1) * sizeof(VertexId);
        data_size["outgoing_edges"] = sockets * sizeof(EdgeId);
        data_size["incoming_edges"] = sockets * sizeof(EdgeId);
        data_size["incoming_adj_bitmap"] = sockets * (vertices / 64 + 1) * sizeof(EdgeId);
        data_size["incoming_adj_index"] = sockets * (vertices + 1) * sizeof(EdgeId);
        data_size["outgoing_adj_bitmap"] = sockets * (vertices / 64 + 1) * sizeof(EdgeId);
        data_size["outgoing_adj_index"] = sockets * (vertices + 1) * sizeof(EdgeId);
        data_size["compressed_incoming_adj_vertices"] = sockets * sizeof(VertexId);
        data_size["compressed_outgoing_adj_vertices"] = sockets * sizeof(VertexId);
        data_size["thread_state"] = threads * sizeof(ThreadState);
        data_size["tuned_chunks_dense"] = partitions * threads * sizeof(ThreadState);
        data_size["tuned_chunks_sparse"] = partitions * threads * sizeof(ThreadState);
        size_t total_size = 0;
        for (auto it = data_size.begin(); it != data_size.end(); it++) {
            total_size += it->second;
        }
        double size_in_gb = static_cast<double>(total_size) / (1024 * 1024 * 1024);
        double size_in_mb = static_cast<double>(total_size) / (1024 * 1024);
        double size_in_kb = static_cast<double>(total_size) / 1024;

        if (size_in_gb >= 1.0) {
            printf("total size: %.2f GB\n", size_in_gb);
        } else if (size_in_mb >= 1.0) {
            printf("total size: %.2f MB\n", size_in_mb);
        } else if (size_in_kb >= 1.0) {
            printf("total size: %.2f KB\n", size_in_kb);
        } else {
            printf("total size: %zu B\n", total_size);
        }
        std::vector<std::pair<std::string, size_t>> vec(data_size.begin(), data_size.end());
        std::sort(
            vec.begin(),
            vec.end(),
            [](const std::pair<std::string, size_t>& a, const std::pair<std::string, size_t>& b) {
                return a.second > b.second;   // 降序排序
            });
        for (const auto& pair : vec) {
            printf_data(pair.first, pair.second, total_size);
        }
        double remote_mem_size = 0;
        // remote_mem_size+=data_size["outgoing_adj_index"]+data_size["outgoing_adj_list"];
        // remote_mem_size+=data_size["compressed_incoming_adj_index"]+data_size["incoming_adj_list"];
        remote_mem_size += data_size["outgoing_adj_index"] + data_size["outgoing_adj_list"] +
                           data_size["compressed_outgoing_adj_index"];
        remote_mem_size += data_size["incoming_adj_index"] + data_size["incoming_adj_list"] +
                           data_size["compressed_incoming_adj_index"];
        printf("remote memory percentage: %lf %\n", remote_mem_size / total_size * 100);
    }

    void init() {
        MPI_Comm_rank(MPI_COMM_WORLD, &partition_id);
        MPI_Comm_size(MPI_COMM_WORLD, &partitions);

        cxl_shm = new CXL_SHM(partitions, partition_id);
        gim_comm = new GIM_comm(cxl_shm);

        edge_data_size = std::is_same<EdgeData, Empty>::value ? 0 : sizeof(EdgeData);
        unit_size = sizeof(VertexId) + edge_data_size;
        edge_unit_size = sizeof(VertexId) + unit_size;

        assert(numa_available() != -1);
        assert(sizeof(unsigned long) == 8);   // assume unsigned long is 64-bit

        char nodestring[sockets * 2 + 1];
        nodestring[0] = '0' + partition_id * NUMA;
        // for (int s_i = 1; s_i < sockets; s_i++) {
        //   nodestring[s_i * 2 - 1] = ',';
        //   nodestring[s_i * 2] = '0' + s_i;
        // }
        // for simulate
        int index = 1;
        for (int s_i = partition_id * NUMA + 1; s_i < partition_id * NUMA + NUMA; s_i++) {
            nodestring[index++] = ',';
            nodestring[index++] = '0' + s_i;
        }
        nodestring[index++] = '\0';
        struct bitmask* nodemask = numa_parse_nodestring(nodestring);
        numa_set_interleave_mask(nodemask);
        // 根据nodestring得到numa配置，然后在这几个numa上交织分配内存

        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        // 禁止omp动态调度并且固定使用cpu核数对应的线程
        // thread_state = new ThreadState*[threads];
        gim_thread_state = new ThreadState**[partitions];
        for (size_t p_i = 0; p_i < partitions; p_i++) {
            gim_thread_state[p_i] = new ThreadState*[threads];
            for (size_t t_i = 0; t_i < threads; t_i++) {
                gim_thread_state[p_i][t_i] =
                    (ThreadState*)cxl_shm->GIM_malloc(sizeof(ThreadState), p_i, get_socket_id(t_i));
                // thread_state[t_i] = (ThreadState*)numa_alloc_onnode(
                // sizeof(ThreadState), get_real_numa_id(t_i, partition_id));
            }
        }


        thread_state = gim_thread_state[partition_id];
        local_send_buffer_limit = 1024;
        local_send_buffer = new MessageBuffer*[threads];
        for (int t_i = 0; t_i < threads; t_i++) {
            local_send_buffer[t_i] = (MessageBuffer*)numa_alloc_onnode(
                sizeof(MessageBuffer), get_real_numa_id(t_i, partition_id));
            local_send_buffer[t_i]->init(get_real_numa_id(t_i, partition_id));
        }
        // numa-aware初始化thread_state 和local_send_buffer
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++) {
            // int s_i = get_socket_id(t_i);
            // for simulate
            int s_i = get_real_numa_id(t_i, partition_id);
            // assert(numa_run_on_node(s_i) == 0);
#ifdef PRINT_DEBUG_MESSAGES
            if (partition_id == 3)
                printf("partition-%d thread-%d bound to socket-%d\n", partition_id, t_i, s_i);
#endif
        }
#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 3) {
            printf("partition=%d threads=%d*%d\n", partition_id, sockets, threads_per_socket);
            printf("interleave on %s\n", nodestring);
        }
#endif
        // 检查多线程能否正常的运行在节点上？
        //  #pragma omp parallel for
        //  for (int t_i = 0; t_i < threads; t_i++) {
        //   // set_thread_affinity(t_i+partition_id*threads);
        // //   if (partition_id==0)
        //    printf("partition:%d,logical thread:%d,physical
        //    thread:%d\n",partition_id,omp_get_thread_num(),get_thread_core_id());
        //  }

        /* origin send_buffer init */
        send_buffer = new MessageBuffer**[partitions];
        recv_buffer = new MessageBuffer**[partitions];
        for (int i = 0; i < partitions; i++) {
            send_buffer[i] = new MessageBuffer*[sockets];
            recv_buffer[i] = new MessageBuffer*[sockets];
            // for simulate
            for (int s_i = 0; s_i < sockets; s_i++) {
                send_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode(sizeof(MessageBuffer),
                                                                        s_i + partition_id * NUMA);
                send_buffer[i][s_i]->init(s_i);
                recv_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode(sizeof(MessageBuffer),
                                                                        s_i + partition_id * NUMA);
                recv_buffer[i][s_i]->init(s_i);
            }
        }
        // send_buffer和recv_buffer是二维的，第一维分区，第二维socket，每个socket内的线程还有自己的local_send_buffer
        alpha = 8 * (partitions - 1);
        global_current_send_part_id =
            (std::atomic<int>*)cxl_shm->CXL_SHM_malloc(sizeof(std::atomic<int>) * partition_id);
        stealings =
            (std::atomic<int>*)cxl_shm->CXL_SHM_malloc(sizeof(std::atomic<int>) * partitions);
        stealingss = (int*)cxl_shm->CXL_SHM_malloc(sizeof(int) * partitions);
        for (int i = 0; i < partitions; i++) {
            stealings[i].store(0);
            stealingss[i] = 0;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void init_gim_buffer() {

        /* gim version send_buffer init */
        size_t max_owned_vertices = 0;
        for (size_t i = 0; i < partitions; i++) {
            size_t p_i_owned_v = partition_offset[i + 1] - partition_offset[i];
            max_owned_vertices =
                p_i_owned_v > max_owned_vertices ? p_i_owned_v : max_owned_vertices;
        }
        gim_send_buffer = new GIMMessageBuffer***[partitions];
        gim_recv_buffer = new GIMMessageBuffer***[partitions];
        send_count=new size_t**[partitions];
#ifdef UNIDIRECTIONAL_MODE
            completion_tags = new std::atomic<bool>***[partitions];
        length_array = new size_t**[partitions];
#endif
        for (int i = 0; i < partitions; i++) {
            gim_send_buffer[i] = new GIMMessageBuffer**[partitions];
            gim_recv_buffer[i] = new GIMMessageBuffer**[partitions];
            send_count[i]=new size_t*[partitions];
#ifdef UNIDIRECTIONAL_MODE
            completion_tags[i] = new std::atomic<bool>**[partitions];
            length_array[i] = new size_t*[partitions];
#endif
            for (size_t j = 0; j < partitions; j++) {
                gim_send_buffer[i][j] = new GIMMessageBuffer*[sockets];
                gim_recv_buffer[i][j] = new GIMMessageBuffer*[sockets];
                send_count[i][j]=(size_t*)cxl_shm->GIM_malloc(sizeof(size_t) * sockets, i);
#ifdef UNIDIRECTIONAL_MODE
                completion_tags[i][j] = new std::atomic<bool>*[sockets];
                length_array[i][j] = (size_t*)cxl_shm->GIM_malloc(sizeof(size_t) * sockets, i);
#endif
                // for simulate
                for (int s_i = 0; s_i < sockets; s_i++) {
                    // void* ptr=cxl_shm->GIM_malloc(sizeof(GIMMessageBuffer),i,s_i);
                    // gim_send_buffer[i][j][s_i] = new(ptr) GIMMessageBuffer(cxl_shm, i, s_i);
                    gim_send_buffer[i][j][s_i] = new GIMMessageBuffer(cxl_shm, i, s_i);
                    gim_send_buffer[i][j][s_i]->init(sizeof(MsgUnit<double>) * max_owned_vertices *
                                                     sockets);
                    gim_recv_buffer[i][j][s_i] = new GIMMessageBuffer(cxl_shm, i, s_i);
                    gim_recv_buffer[i][j][s_i]->init(sizeof(MsgUnit<double>) * max_owned_vertices *
                                                     sockets);
                    send_count[i][j][s_i]=0;
#ifdef UNIDIRECTIONAL_MODE
                    // 实际上是已经分配好了一个gim，然后再把对应的位置指针返回，并没有每次都malloc
                    completion_tags[i][j][s_i] =
                        (std::atomic<bool>*)cxl_shm->GIM_malloc(sizeof(std::atomic<bool>), i, s_i);
                    completion_tags[i][j][s_i]->store(false);
#endif
                }
            }
        }
    }

    double print_total_process_time() { return total_process_time; }
    // fill a vertex array with a specific value
    template<typename T> void fill_vertex_array(T* array, T value) {
#pragma omp parallel for
        for (VertexId v_i = partition_offset[partition_id];
             v_i < partition_offset[partition_id + 1];
             v_i++) {
            array[v_i] = value;
        }
    }

    // allocate a numa-aware vertex array
    template<typename T> T* alloc_vertex_array() {
        char* array = (char*)mmap(
            NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(array != NULL);
        // 需要连续的点数组，所以先分配内存再迁移到numa，而不是直接在numa上分配
        //  for simulate
        for (int s_i = 0; s_i < sockets; s_i++) {
            numa_tonode_memory(
                array + sizeof(T) * local_partition_offset[s_i],
                sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]),
                s_i + partition_id * NUMA);
        }
        return (T*)array;
    }

    // deallocate a vertex array
    template<typename T> void dealloc_vertex_array(T* array) {
        numa_free(array, sizeof(T) * vertices);
    }

    // allocate a numa-oblivious vertex array
    template<typename T> T* alloc_interleaved_vertex_array() {
        T* array = (T*)numa_alloc_interleaved(sizeof(T) * vertices);
        assert(array != NULL);
        return array;
    }

    // dump a vertex array to path   数组内容写到文件里
    template<typename T> void dump_vertex_array(T* array, std::string path) {
        long file_length = sizeof(T) * vertices;
        if (!file_exists(path) || file_size(path) != file_length) {
            if (partition_id == 0) {
                FILE* fout = fopen(path.c_str(), "wb");
                char* buffer = new char[PAGESIZE];
                for (long offset = 0; offset < file_length;) {
                    if (file_length - offset >= PAGESIZE) {
                        fwrite(buffer, 1, PAGESIZE, fout);
                        offset += PAGESIZE;
                    } else {
                        fwrite(buffer, 1, file_length - offset, fout);
                        offset += file_length - offset;
                    }
                }
                fclose(fout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        int fd = open(path.c_str(), O_RDWR);
        assert(fd != -1);
        long offset = sizeof(T) * partition_offset[partition_id];
        long end_offset = sizeof(T) * partition_offset[partition_id + 1];
        void* data = (void*)array;
        assert(lseek(fd, offset, SEEK_SET) != -1);
        while (offset < end_offset) {
            long bytes = write(fd, data + offset, end_offset - offset);
            assert(bytes != -1);
            offset += bytes;
        }
        assert(close(fd) == 0);
    }

    // restore a vertex array from path  从文件中重新读数据到数组
    template<typename T> void restore_vertex_array(T* array, std::string path) {
        long file_length = sizeof(T) * vertices;
        if (!file_exists(path) || file_size(path) != file_length) {
            assert(false);
        }
        int fd = open(path.c_str(), O_RDWR);
        assert(fd != -1);
        long offset = sizeof(T) * partition_offset[partition_id];
        long end_offset = sizeof(T) * partition_offset[partition_id + 1];
        void* data = (void*)array;
        assert(lseek(fd, offset, SEEK_SET) != -1);
        while (offset < end_offset) {
            long bytes = read(fd, data + offset, end_offset - offset);
            assert(bytes != -1);
            offset += bytes;
        }
        assert(close(fd) == 0);
    }

    // gather a vertex array  从其他节点(分区)接受点数组到root节点
    template<typename T> void gather_vertex_array(T* array, int root) {
        if (partition_id != root) {
            MPI_Send(array + partition_offset[partition_id],
                     sizeof(T) * owned_vertices,
                     MPI_CHAR,
                     root,
                     GatherVertexArray,
                     MPI_COMM_WORLD);
        } else {
            for (int i = 0; i < partitions; i++) {
                if (i == partition_id) continue;
                MPI_Status recv_status;
                MPI_Recv(array + partition_offset[i],
                         sizeof(T) * (partition_offset[i + 1] - partition_offset[i]),
                         MPI_CHAR,
                         i,
                         GatherVertexArray,
                         MPI_COMM_WORLD,
                         &recv_status);
                int length;
                MPI_Get_count(&recv_status, MPI_CHAR, &length);
                assert(length == sizeof(T) * (partition_offset[i + 1] - partition_offset[i]));
            }
        }
    }

    // allocate a vertex subset
    VertexSubset* alloc_vertex_subset() { return new VertexSubset(vertices); }
    VertexSubset** alloc_global_vertex_subset() {
        VertexSubset** global_vertex_subset = new VertexSubset*[partitions];
        for (int i = 0; i < partitions; i++) {
            unsigned long* data = (unsigned long*)cxl_shm->GIM_malloc(
                sizeof(unsigned long) * (WORD_OFFSET(vertices) + 1), i);
            // unsigned long* data = (unsigned long*)malloc(
            // sizeof(unsigned long) * (WORD_OFFSET(vertices) + 1));
            global_vertex_subset[i] = new VertexSubset(vertices, data);
            // global_vertex_subset[i] = new VertexSubset(vertices);
        }
        return global_vertex_subset;
    }

    template<typename T> T** alloc_global_vertex_array() {
        T** global_vertex_array = new T*[partitions];
        for (int i = 0; i < partitions; i++) {
            T* data = (T*)cxl_shm->GIM_malloc(sizeof(T) * vertices, i);
            global_vertex_array[i] = data;
        }
        return global_vertex_array;
    }

    // 根据点的index得到这个点所在分区（全局）的index
    int get_partition_id(VertexId v_i) {
        for (int i = 0; i < partitions; i++) {
            if (v_i >= partition_offset[i] && v_i < partition_offset[i + 1]) {
                return i;
            }
        }
        assert(false);
    }
    // 根据点的index得到这个点所在分区（本地）的index
    int get_local_partition_id(VertexId v_i) {
        for (int s_i = 0; s_i < sockets; s_i++) {
            if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i + 1]) {
                return s_i;
            }
        }
        assert(false);
    }

    // load a directed graph and make it undirected
    //     void load_undirected_from_directed(std::string path, VertexId vertices) {
    //         double prep_time = 0;
    //         prep_time -= MPI_Wtime();

    //         symmetric = true;

    //         MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

    //         this->vertices = vertices;
    //         long total_bytes = file_size(path.c_str());
    //         this->edges = total_bytes / edge_unit_size;
    // #ifdef PRINT_DEBUG_MESSAGES
    //         if (partition_id == 0) {
    //             printf("|V| = %u, |E| = %lu\n", vertices, edges);
    //         }
    // #endif

    //         EdgeId read_edges = edges / partitions;
    //         if (partition_id == partitions - 1) {
    //             read_edges += edges % partitions;
    //         }
    //         long bytes_to_read = edge_unit_size * read_edges;
    //         long read_offset = edge_unit_size * (edges / partitions * partition_id);
    //         long read_bytes;
    //         int fin = open(path.c_str(), O_RDONLY);
    //         EdgeUnit<EdgeData>* read_edge_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    //         out_degree = alloc_interleaved_vertex_array<VertexId>();   // numa交织分配内存
    //         for (VertexId v_i = 0; v_i < vertices; v_i++) {
    //             out_degree[v_i] = 0;
    //         }
    //         assert(lseek(fin, read_offset, SEEK_SET) == read_offset); //定位到每个节点要读的位置
    //         read_bytes = 0;
    //         while (read_bytes < bytes_to_read) {   //每次读一个chunksize的边
    //             long curr_read_bytes;
    //             if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
    //                 curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
    //             } else {
    //                 curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
    //             }
    //             assert(curr_read_bytes >= 0);
    //             read_bytes += curr_read_bytes;
    //             EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
    //             // #pragma omp parallel for
    //             for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
    //                 VertexId src = read_edge_buffer[e_i].src;
    //                 VertexId dst = read_edge_buffer[e_i].dst;
    //                 __sync_fetch_and_add(&out_degree[src], 1);
    //                 __sync_fetch_and_add(&out_degree[dst], 1);
    //             }
    //         }
    //         MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);

    //         // locality-aware chunking
    //         partition_offset = new VertexId[partitions + 1];
    //         partition_offset[0] = 0;
    //         EdgeId remained_amount = edges * 2 + EdgeId(vertices) * alpha;
    //         for (int i = 0; i < partitions; i++) {
    //             VertexId remained_partitions = partitions - i;
    //             EdgeId expected_chunk_size = remained_amount / remained_partitions;
    //             if (remained_partitions == 1) {
    //                 partition_offset[i + 1] = vertices;
    //             } else {
    //                 EdgeId got_edges = 0;
    //                 for (VertexId v_i = partition_offset[i]; v_i < vertices; v_i++) {
    //                     got_edges += out_degree[v_i] + alpha;
    //                     if (got_edges > expected_chunk_size) {
    //                         partition_offset[i + 1] = v_i;
    //                         break;
    //                     }
    //                 }
    //                 partition_offset[i + 1] =
    //                     (partition_offset[i + 1]) / PAGESIZE * PAGESIZE;   // aligned with pages
    //             }
    //             for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++) {
    //                 remained_amount -= out_degree[v_i] + alpha;
    //             }
    //         }
    //         assert(partition_offset[partitions] == vertices);
    //         owned_vertices = partition_offset[partition_id + 1] - partition_offset[partition_id];
    //         // check consistency of partition boundaries
    //         VertexId* global_partition_offset = new VertexId[partitions + 1];
    //         MPI_Allreduce(partition_offset,
    //                       global_partition_offset,
    //                       partitions + 1,
    //                       vid_t,
    //                       MPI_MAX,
    //                       MPI_COMM_WORLD);
    //         for (int i = 0; i <= partitions; i++) {
    //             assert(partition_offset[i] == global_partition_offset[i]);
    //         }
    //         MPI_Allreduce(partition_offset,
    //                       global_partition_offset,
    //                       partitions + 1,
    //                       vid_t,
    //                       MPI_MIN,
    //                       MPI_COMM_WORLD);
    //         for (int i = 0; i <= partitions; i++) {
    //             assert(partition_offset[i] == global_partition_offset[i]);
    //         }
    // #ifdef PRINT_DEBUG_MESSAGES
    //         if (partition_id == 0) {
    //             for (int i = 0; i < partitions; i++) {
    //                 EdgeId part_out_edges = 0;
    //                 for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1];
    //                 v_i++) {
    //                     part_out_edges += out_degree[v_i];
    //                 }
    //                 printf("|V'_%d| = %u |E_%d| = %lu\n",
    //                        i,
    //                        partition_offset[i + 1] - partition_offset[i],
    //                        i,
    //                        part_out_edges);
    //             }
    //         }
    //         MPI_Barrier(MPI_COMM_WORLD);
    // #endif
    //         delete[] global_partition_offset;
    //         {
    //             // NUMA-aware sub-chunking
    //             local_partition_offset = new VertexId[sockets + 1];
    //             EdgeId part_out_edges = 0;
    //             for (VertexId v_i = partition_offset[partition_id];
    //                  v_i < partition_offset[partition_id + 1];
    //                  v_i++) {
    //                 part_out_edges += out_degree[v_i];
    //             }
    //             local_partition_offset[0] = partition_offset[partition_id];
    //             EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
    //             for (int s_i = 0; s_i < sockets; s_i++) {
    //                 VertexId remained_partitions = sockets - s_i;
    //                 EdgeId expected_chunk_size = remained_amount / remained_partitions;
    //                 if (remained_partitions == 1) {
    //                     local_partition_offset[s_i + 1] = partition_offset[partition_id + 1];
    //                 } else {
    //                     EdgeId got_edges = 0;
    //                     for (VertexId v_i = local_partition_offset[s_i];
    //                          v_i < partition_offset[partition_id + 1];
    //                          v_i++) {
    //                         got_edges += out_degree[v_i] + alpha;
    //                         if (got_edges > expected_chunk_size) {
    //                             local_partition_offset[s_i + 1] = v_i;
    //                             break;
    //                         }
    //                     }
    //                     local_partition_offset[s_i + 1] = (local_partition_offset[s_i + 1]) /
    //                     PAGESIZE *
    //                                                       PAGESIZE;   // aligned with pages
    //                 }
    //                 EdgeId sub_part_out_edges = 0;
    //                 for (VertexId v_i = local_partition_offset[s_i];
    //                      v_i < local_partition_offset[s_i + 1];
    //                      v_i++) {
    //                     remained_amount -= out_degree[v_i] + alpha;
    //                     sub_part_out_edges += out_degree[v_i];
    //                 }
    // #ifdef PRINT_DEBUG_MESSAGES
    //                 printf("|V'_%d_%d| = %u |E_%d| = %lu\n",
    //                        partition_id,
    //                        s_i,
    //                        local_partition_offset[s_i + 1] - local_partition_offset[s_i],
    //                        partition_id,
    //                        sub_part_out_edges);
    // #endif
    //             }
    //         }

    //         VertexId* filtered_out_degree = alloc_vertex_array<VertexId>();
    //         for (VertexId v_i = partition_offset[partition_id];
    //              v_i < partition_offset[partition_id + 1];
    //              v_i++) {
    //             filtered_out_degree[v_i] = out_degree[v_i];
    //         }
    //         numa_free(out_degree, sizeof(VertexId) * vertices);
    //         out_degree = filtered_out_degree;
    //         in_degree = out_degree;

    //         int* buffered_edges = new int[partitions];
    //         std::vector<char>* send_buffer = new std::vector<char>[partitions];
    //         for (int i = 0; i < partitions; i++) {
    //             send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    //         }
    //         EdgeUnit<EdgeData>* recv_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    //         // constructing symmetric edges
    //         EdgeId recv_outgoing_edges = 0;
    //         outgoing_edges = new EdgeId[sockets];
    //         outgoing_adj_index = new EdgeId*[sockets];
    //         outgoing_adj_list = new AdjUnit<EdgeData>*[sockets];
    //         outgoing_adj_bitmap = new Bitmap*[sockets];
    //         for (int s_i = 0; s_i < sockets; s_i++) {
    //             outgoing_adj_bitmap[s_i] = new Bitmap(vertices);
    //             outgoing_adj_bitmap[s_i]->clear();
    // #ifdef CXL_SHM
    //             outgoing_adj_index[s_i] =
    //                 (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1), REMOTE_NUMA);
    // #else
    //             outgoing_adj_index[s_i] = (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices +
    //             1),
    //                                                                  s_i + partition_id * NUMA);
    // #endif
    //         }
    //         {
    //             std::thread recv_thread_dst([&]() {
    //                 int finished_count = 0;
    //                 MPI_Status recv_status;
    //                 while (finished_count < partitions) {
    //                     MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
    //                     int i = recv_status.MPI_SOURCE;
    //                     assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
    //                     int recv_bytes;
    //                     MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
    //                     if (recv_bytes == 1) {
    //                         finished_count += 1;
    //                         char c;
    //                         MPI_Recv(
    //                             &c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD,
    //                             MPI_STATUS_IGNORE);
    //                         continue;
    //                     }
    //                     assert(recv_bytes % edge_unit_size == 0);
    //                     int recv_edges = recv_bytes / edge_unit_size;
    //                     MPI_Recv(recv_buffer,
    //                              edge_unit_size * recv_edges,
    //                              MPI_CHAR,
    //                              i,
    //                              ShuffleGraph,
    //                              MPI_COMM_WORLD,
    //                              MPI_STATUS_IGNORE);
    //                     // #pragma omp parallel for
    //                     for (EdgeId e_i = 0; e_i < recv_edges; e_i++) {
    //                         VertexId src = recv_buffer[e_i].src;
    //                         VertexId dst = recv_buffer[e_i].dst;
    //                         assert(dst >= partition_offset[partition_id] &&
    //                                dst < partition_offset[partition_id + 1]);
    //                         int dst_part = get_local_partition_id(dst);
    //                         if (!outgoing_adj_bitmap[dst_part]->get_bit(src)) {
    //                             outgoing_adj_bitmap[dst_part]->set_bit(src);
    //                             outgoing_adj_index[dst_part][src] = 0;
    //                         }
    //                         __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
    //                     }
    //                     recv_outgoing_edges += recv_edges;
    //                 }
    //             });
    //             for (int i = 0; i < partitions; i++) {
    //                 buffered_edges[i] = 0;
    //             }
    //             assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    //             read_bytes = 0;
    //             while (read_bytes < bytes_to_read) {
    //                 long curr_read_bytes;
    //                 if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
    //                     curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size *
    //                     CHUNKSIZE);
    //                 } else {
    //                     curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read -
    //                     read_bytes);
    //                 }
    //                 assert(curr_read_bytes >= 0);
    //                 read_bytes += curr_read_bytes;
    //                 EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
    //                 for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
    //                     VertexId dst = read_edge_buffer[e_i].dst;
    //                     int i = get_partition_id(dst);
    //                     memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i],
    //                            &read_edge_buffer[e_i],
    //                            edge_unit_size);
    //                     buffered_edges[i] += 1;
    //                     if (buffered_edges[i] == CHUNKSIZE) {
    //                         MPI_Send(send_buffer[i].data(),
    //                                  edge_unit_size * buffered_edges[i],
    //                                  MPI_CHAR,
    //                                  i,
    //                                  ShuffleGraph,
    //                                  MPI_COMM_WORLD);
    //                         buffered_edges[i] = 0;
    //                     }
    //                 }
    //                 for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
    //                     // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
    //                     VertexId tmp = read_edge_buffer[e_i].src;
    //                     read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
    //                     read_edge_buffer[e_i].dst = tmp;
    //                 }
    //                 for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
    //                     VertexId dst = read_edge_buffer[e_i].dst;
    //                     int i = get_partition_id(dst);
    //                     memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i],
    //                            &read_edge_buffer[e_i],
    //                            edge_unit_size);
    //                     buffered_edges[i] += 1;
    //                     if (buffered_edges[i] == CHUNKSIZE) {
    //                         MPI_Send(send_buffer[i].data(),
    //                                  edge_unit_size * buffered_edges[i],
    //                                  MPI_CHAR,
    //                                  i,
    //                                  ShuffleGraph,
    //                                  MPI_COMM_WORLD);
    //                         buffered_edges[i] = 0;
    //                     }
    //                 }
    //             }
    //             for (int i = 0; i < partitions; i++) {
    //                 if (buffered_edges[i] == 0) continue;
    //                 MPI_Send(send_buffer[i].data(),
    //                          edge_unit_size * buffered_edges[i],
    //                          MPI_CHAR,
    //                          i,
    //                          ShuffleGraph,
    //                          MPI_COMM_WORLD);
    //                 buffered_edges[i] = 0;
    //             }
    //             for (int i = 0; i < partitions; i++) {
    //                 char c = 0;
    //                 MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
    //             }
    //             recv_thread_dst.join();
    // #ifdef PRINT_DEBUG_MESSAGES
    //             printf("machine(%d) got %lu symmetric edges\n", partition_id,
    //             recv_outgoing_edges);
    // #endif
    //         }
    //         compressed_outgoing_adj_vertices = new VertexId[sockets];
    //         compressed_outgoing_adj_index = new CompressedAdjIndexUnit*[sockets];
    //         for (int s_i = 0; s_i < sockets; s_i++) {
    //             outgoing_edges[s_i] = 0;
    //             compressed_outgoing_adj_vertices[s_i] = 0;
    //             for (VertexId v_i = 0; v_i < vertices; v_i++) {
    //                 if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
    //                     outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
    //                     compressed_outgoing_adj_vertices[s_i] += 1;
    //                 }
    //             }
    //             compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit*)numa_alloc_onnode(
    //                 sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1),
    //                 s_i + partition_id * NUMA);
    //             compressed_outgoing_adj_index[s_i][0].index = 0;
    //             EdgeId last_e_i = 0;
    //             compressed_outgoing_adj_vertices[s_i] = 0;
    //             for (VertexId v_i = 0; v_i < vertices; v_i++) {
    //                 if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
    //                     outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
    //                     last_e_i = outgoing_adj_index[s_i][v_i];
    //                     compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]]
    //                         .vertex = v_i;
    //                     compressed_outgoing_adj_vertices[s_i] += 1;
    //                     compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]]
    //                         .index = last_e_i;
    //                 }
    //             }
    //             for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
    //             {
    //                 VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
    //                 outgoing_adj_index[s_i][v_i] =
    //                 compressed_outgoing_adj_index[s_i][p_v_i].index; outgoing_adj_index[s_i][v_i
    //                 + 1] =
    //                     compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
    //             }
    // #ifdef PRINT_DEBUG_MESSAGES
    //             printf(
    //                 "part(%d) E_%d has %lu symmetric edges\n", partition_id, s_i,
    //                 outgoing_edges[s_i]);
    // #endif
    //             outgoing_adj_list[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(
    //                 unit_size * outgoing_edges[s_i], s_i + partition_id * NUMA);
    //         }
    //         {
    //             std::thread recv_thread_dst([&]() {
    //                 int finished_count = 0;
    //                 MPI_Status recv_status;
    //                 while (finished_count < partitions) {
    //                     MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
    //                     int i = recv_status.MPI_SOURCE;
    //                     assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
    //                     int recv_bytes;
    //                     MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
    //                     if (recv_bytes == 1) {
    //                         finished_count += 1;
    //                         char c;
    //                         MPI_Recv(
    //                             &c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD,
    //                             MPI_STATUS_IGNORE);
    //                         continue;
    //                     }
    //                     assert(recv_bytes % edge_unit_size == 0);
    //                     int recv_edges = recv_bytes / edge_unit_size;
    //                     MPI_Recv(recv_buffer,
    //                              edge_unit_size * recv_edges,
    //                              MPI_CHAR,
    //                              i,
    //                              ShuffleGraph,
    //                              MPI_COMM_WORLD,
    //                              MPI_STATUS_IGNORE);
    // #pragma omp parallel for
    //                     for (EdgeId e_i = 0; e_i < recv_edges; e_i++) {
    //                         VertexId src = recv_buffer[e_i].src;
    //                         VertexId dst = recv_buffer[e_i].dst;
    //                         assert(dst >= partition_offset[partition_id] &&
    //                                dst < partition_offset[partition_id + 1]);
    //                         int dst_part = get_local_partition_id(dst);
    //                         EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src],
    //                         1); outgoing_adj_list[dst_part][pos].neighbour = dst; if
    //                         (!std::is_same<EdgeData, Empty>::value) {
    //                             outgoing_adj_list[dst_part][pos].edge_data =
    //                             recv_buffer[e_i].edge_data;
    //                         }
    //                     }
    //                 }
    //             });
    //             for (int i = 0; i < partitions; i++) {
    //                 buffered_edges[i] = 0;
    //             }
    //             assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    //             read_bytes = 0;
    //             while (read_bytes < bytes_to_read) {
    //                 long curr_read_bytes;
    //                 if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
    //                     curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size *
    //                     CHUNKSIZE);
    //                 } else {
    //                     curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read -
    //                     read_bytes);
    //                 }
    //                 assert(curr_read_bytes >= 0);
    //                 read_bytes += curr_read_bytes;
    //                 EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
    //                 for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
    //                     VertexId dst = read_edge_buffer[e_i].dst;
    //                     int i = get_partition_id(dst);
    //                     memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i],
    //                            &read_edge_buffer[e_i],
    //                            edge_unit_size);
    //                     buffered_edges[i] += 1;
    //                     if (buffered_edges[i] == CHUNKSIZE) {
    //                         MPI_Send(send_buffer[i].data(),
    //                                  edge_unit_size * buffered_edges[i],
    //                                  MPI_CHAR,
    //                                  i,
    //                                  ShuffleGraph,
    //                                  MPI_COMM_WORLD);
    //                         buffered_edges[i] = 0;
    //                     }
    //                 }
    //                 for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
    //                     // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
    //                     VertexId tmp = read_edge_buffer[e_i].src;
    //                     read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
    //                     read_edge_buffer[e_i].dst = tmp;
    //                 }
    //                 for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
    //                     VertexId dst = read_edge_buffer[e_i].dst;
    //                     int i = get_partition_id(dst);
    //                     memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i],
    //                            &read_edge_buffer[e_i],
    //                            edge_unit_size);
    //                     buffered_edges[i] += 1;
    //                     if (buffered_edges[i] == CHUNKSIZE) {
    //                         MPI_Send(send_buffer[i].data(),
    //                                  edge_unit_size * buffered_edges[i],
    //                                  MPI_CHAR,
    //                                  i,
    //                                  ShuffleGraph,
    //                                  MPI_COMM_WORLD);
    //                         buffered_edges[i] = 0;
    //                     }
    //                 }
    //             }
    //             for (int i = 0; i < partitions; i++) {
    //                 if (buffered_edges[i] == 0) continue;
    //                 MPI_Send(send_buffer[i].data(),
    //                          edge_unit_size * buffered_edges[i],
    //                          MPI_CHAR,
    //                          i,
    //                          ShuffleGraph,
    //                          MPI_COMM_WORLD);
    //                 buffered_edges[i] = 0;
    //             }
    //             for (int i = 0; i < partitions; i++) {
    //                 char c = 0;
    //                 MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
    //             }
    //             recv_thread_dst.join();
    //         }
    //         for (int s_i = 0; s_i < sockets; s_i++) {
    //             for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
    //             {
    //                 VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
    //                 outgoing_adj_index[s_i][v_i] =
    //                 compressed_outgoing_adj_index[s_i][p_v_i].index; outgoing_adj_index[s_i][v_i
    //                 + 1] =
    //                     compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
    //             }
    //         }
    //         MPI_Barrier(MPI_COMM_WORLD);

    //         incoming_edges = outgoing_edges;
    //         incoming_adj_index = outgoing_adj_index;
    //         incoming_adj_list = outgoing_adj_list;
    //         incoming_adj_bitmap = outgoing_adj_bitmap;
    //         compressed_incoming_adj_vertices = compressed_outgoing_adj_vertices;
    //         compressed_incoming_adj_index = compressed_outgoing_adj_index;
    //         MPI_Barrier(MPI_COMM_WORLD);

    //         delete[] buffered_edges;
    //         delete[] send_buffer;
    //         delete[] read_edge_buffer;
    //         delete[] recv_buffer;
    //         close(fin);

    //         tune_chunks();
    //         tuned_chunks_sparse = tuned_chunks_dense;

    //         //gim_buffer
    //         init_gim_buffer();

    //             prep_time += MPI_Wtime();

    // #ifdef PRINT_DEBUG_MESSAGES
    //         if (partition_id == 0) {
    //             printf("preprocessing cost: %.2lf (s)\n", prep_time);
    //         }
    // #endif
    //     }

    // transpose the graph
    void transpose() {
        std::swap(out_degree, in_degree);
        std::swap(outgoing_edges, incoming_edges);
        std::swap(outgoing_adj_index, incoming_adj_index);
        std::swap(outgoing_adj_bitmap, incoming_adj_bitmap);
        std::swap(outgoing_adj_list, incoming_adj_list);
        std::swap(tuned_chunks_dense, tuned_chunks_sparse);
        std::swap(compressed_outgoing_adj_vertices, compressed_incoming_adj_vertices);
        std::swap(compressed_outgoing_adj_index, compressed_incoming_adj_index);
        /* gim */
        std::swap(gim_out_degree, gim_in_degree);
        std::swap(gim_outgoing_adj_index, gim_incoming_adj_index);
        std::swap(gim_outgoing_adj_bitmap, gim_incoming_adj_bitmap);
        std::swap(gim_outgoing_adj_list, gim_incoming_adj_list);
        std::swap(gim_compressed_outgoing_adj_index, gim_compressed_incoming_adj_index);
    }

    // load a directed graph from path
    void load_directed(std::string path, VertexId vertices) {
        double prep_time = 0;
        prep_time -= MPI_Wtime();
        symmetric = false;

        MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

        this->vertices = vertices;

        long total_bytes = file_size(path.c_str());
        this->edges = total_bytes / edge_unit_size;
#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 0) {
            printf("|V| = %u, |E| = %lu\n", vertices, edges);
        }
#endif
        // 计算每个分区平均读多少条边
        EdgeId read_edges = edges / partitions;
        if (partition_id == partitions - 1) {
            read_edges += edges % partitions;
        }
        long bytes_to_read = edge_unit_size * read_edges;   // 每个分区读的字节数
        long read_offset =
            edge_unit_size * (edges / partitions * partition_id);   // 从文件的哪个地方开始读
        long read_bytes;                                            // 已经读的字节数
        int fin = open(path.c_str(), O_RDONLY);
        EdgeUnit<EdgeData>* read_edge_buffer =
            new EdgeUnit<EdgeData>[CHUNKSIZE];   // 将边的数据读到这里

        gim_out_degree = new VertexId*[partitions];
        for (int i = 0; i < partitions; i++) {
            gim_out_degree[i] = (VertexId*)cxl_shm->GIM_malloc(sizeof(VertexId) * vertices, i);
        }
        //  out_degree = alloc_interleaved_vertex_array<VertexId>();
        out_degree = gim_out_degree[partition_id];
        for (VertexId v_i = 0; v_i < vertices; v_i++) {
            out_degree[v_i] = 0;
        }
        assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
        read_bytes = 0;
        // 每次读一个CHUNKSIZE的边，读边是为了计算点的出度
        while (read_bytes < bytes_to_read) {
            long curr_read_bytes;
            if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
                curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
            } else {
                curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
            }
            assert(curr_read_bytes >= 0);
            read_bytes += curr_read_bytes;
            EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
#pragma omp parallel for
            for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
                VertexId src = read_edge_buffer[e_i].src;
                VertexId dst = read_edge_buffer[e_i].dst;
                __sync_fetch_and_add(&out_degree[src], 1);   // 每个节点都更新每个点的出度
            }
        }
        MPI_Allreduce(MPI_IN_PLACE,
                      out_degree,
                      vertices,
                      vid_t,
                      MPI_SUM,
                      MPI_COMM_WORLD);   // 更新全局的点的出度

        // 进行分区，确定每个节点上面的master边
        //  locality-aware chunking
        partition_offset = new VertexId[partitions + 1];
        partition_offset[0] = 0;
        EdgeId remained_amount = edges + EdgeId(vertices) * alpha;
        for (int i = 0; i < partitions; i++) {
            VertexId remained_partitions = partitions - i;
            EdgeId expected_chunk_size = remained_amount / remained_partitions;
            if (remained_partitions == 1) {
                partition_offset[i + 1] = vertices;
            } else {
                EdgeId got_edges = 0;
                for (VertexId v_i = partition_offset[i]; v_i < vertices; v_i++) {
                    got_edges += out_degree[v_i] + alpha;
                    if (got_edges > expected_chunk_size) {
                        partition_offset[i + 1] = v_i;
                        break;
                    }
                }
                partition_offset[i + 1] =
                    (partition_offset[i + 1]) / PAGESIZE * PAGESIZE;   // aligned with pages
            }
            for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++) {
                remained_amount -= out_degree[v_i] + alpha;
            }
        }
        assert(partition_offset[partitions] == vertices);
        owned_vertices = partition_offset[partition_id + 1] -
                         partition_offset[partition_id];   // 更新本节点拥有的点

        // 如果有节点计算结果不同，利用allreduce来保证每个节点上partition_offset是相同的
        //  check consistency of partition boundaries
        VertexId* global_partition_offset = new VertexId[partitions + 1];
        MPI_Allreduce(partition_offset,
                      global_partition_offset,
                      partitions + 1,
                      vid_t,
                      MPI_MAX,
                      MPI_COMM_WORLD);
        for (int i = 0; i <= partitions; i++) {
            assert(partition_offset[i] == global_partition_offset[i]);
        }
        MPI_Allreduce(partition_offset,
                      global_partition_offset,
                      partitions + 1,
                      vid_t,
                      MPI_MIN,
                      MPI_COMM_WORLD);
        for (int i = 0; i <= partitions; i++) {
            assert(partition_offset[i] == global_partition_offset[i]);
        }

#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 0) {
            for (int i = 0; i < partitions; i++) {
                EdgeId part_out_edges = 0;
                for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++) {
                    part_out_edges += out_degree[v_i];
                }
                printf("|V'_%d| = %u |E^dense_%d| = %lu\n",
                       i,
                       partition_offset[i + 1] - partition_offset[i],
                       i,
                       part_out_edges);
            }
        }
#endif
        delete[] global_partition_offset;

        // 跟上面类似，计算每个numa的点的范围
        {
            // NUMA-aware sub-chunking
            local_partition_offset = new VertexId[sockets + 1];
            EdgeId part_out_edges = 0;
            for (VertexId v_i = partition_offset[partition_id];
                 v_i < partition_offset[partition_id + 1];
                 v_i++) {
                part_out_edges += out_degree[v_i];
            }
            local_partition_offset[0] = partition_offset[partition_id];
            EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
            for (int s_i = 0; s_i < sockets; s_i++) {
                VertexId remained_partitions = sockets - s_i;
                EdgeId expected_chunk_size = remained_amount / remained_partitions;
                if (remained_partitions == 1) {
                    local_partition_offset[s_i + 1] = partition_offset[partition_id + 1];
                } else {
                    EdgeId got_edges = 0;
                    for (VertexId v_i = local_partition_offset[s_i];
                         v_i < partition_offset[partition_id + 1];
                         v_i++) {
                        got_edges += out_degree[v_i] + alpha;
                        if (got_edges > expected_chunk_size) {
                            local_partition_offset[s_i + 1] = v_i;
                            break;
                        }
                    }
                    local_partition_offset[s_i + 1] = (local_partition_offset[s_i + 1]) / PAGESIZE *
                                                      PAGESIZE;   // aligned with pages
                }
                EdgeId sub_part_out_edges = 0;
                for (VertexId v_i = local_partition_offset[s_i];
                     v_i < local_partition_offset[s_i + 1];
                     v_i++) {
                    remained_amount -= out_degree[v_i] + alpha;
                    sub_part_out_edges += out_degree[v_i];
                }
#ifdef PRINT_DEBUG_MESSAGES
                printf("|V'_%d_%d| = %u |E^dense_%d_%d| = %lu\n",
                       partition_id,
                       s_i,
                       local_partition_offset[s_i + 1] - local_partition_offset[s_i],
                       partition_id,
                       s_i,
                       sub_part_out_edges);
#endif
            }
        }
        // 确保每个分区只处理与其相关的顶点和边信息
        VertexId* filtered_out_degree =
            alloc_vertex_array<VertexId>();   // 每个节点内自己的点的出度
        for (VertexId v_i = partition_offset[partition_id];
             v_i < partition_offset[partition_id + 1];
             v_i++) {
            filtered_out_degree[v_i] = out_degree[v_i];
        }
        numa_free(out_degree, sizeof(VertexId) * vertices);
        out_degree = filtered_out_degree;   // 现在开始出度只记录自己节点的出度
        in_degree = alloc_vertex_array<VertexId>();
        for (VertexId v_i = partition_offset[partition_id];
             v_i < partition_offset[partition_id + 1];
             v_i++) {
            in_degree[v_i] = 0;
        }

        int* buffered_edges = new int[partitions];
        std::vector<char>* send_buffer =
            new std::vector<char>[partitions];   // 每个分区partitions个sendbuffer
        for (int i = 0; i < partitions; i++) {
            send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
        }
        EdgeUnit<EdgeData>* recv_buffer =
            new EdgeUnit<EdgeData>[CHUNKSIZE];   // 每个全局分区一个recvbuffer

        // 每个节点内自己的数据，numa-aware
        EdgeId recv_outgoing_edges = 0;
        outgoing_edges =
            new EdgeId[sockets];   // 存储每个 NUMA
                                   // 节点的出边数量，用于分布式图计算时确定每个节点需要处理的边。
        // outgoing_adj_index =
        //     new EdgeId*[sockets];   //邻接表索引数组，存储每个顶点的出边列表起始位置和结束位置。
        gim_outgoing_adj_index = new EdgeId**[partitions];
        for (size_t p_i = 0; p_i < partitions; p_i++) {
            gim_outgoing_adj_index[p_i] = new EdgeId*[sockets];
            for (size_t s_i = 0; s_i < sockets; s_i++) {
                gim_outgoing_adj_index[p_i][s_i] =
                    (EdgeId*)cxl_shm->CXL_SHM_malloc(sizeof(EdgeId) * (vertices + 1));

                // gim_outgoing_adj_index[p_i][s_i] = (EdgeId*)numa_alloc_onnode(
                //     sizeof(EdgeId) * (vertices + 1), s_i + partition_id * NUMA);
            }
        }
        outgoing_adj_index = gim_outgoing_adj_index[partition_id];
        // outgoing_adj_bitmap = new Bitmap*[sockets];
        gim_outgoing_adj_bitmap = new Bitmap**[partitions];
        for (size_t p_i = 0; p_i < partitions; p_i++) {
            gim_outgoing_adj_bitmap[p_i] = new Bitmap*[sockets];
            for (size_t s_i = 0; s_i < sockets; s_i++) {
                // gim_outgoing_adj_bitmap[p_i][s_i] = new Bitmap(vertices);
                unsigned long* data = (unsigned long*)cxl_shm->GIM_malloc(
                    sizeof(unsigned long) * (WORD_OFFSET(vertices) + 1), p_i);
                // unsigned long* data =
                //     (unsigned long*)cxl_shm->CXL_SHM_malloc((WORD_OFFSET(vertices) + 1));

                // unsigned long* data = new unsigned long[((WORD_OFFSET(vertices) + 1))];
                gim_outgoing_adj_bitmap[p_i][s_i] = new Bitmap(vertices, data);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        outgoing_adj_bitmap = gim_outgoing_adj_bitmap[partition_id];

        // outgoing_adj_list =
        //     new AdjUnit<EdgeData>*[sockets];   //邻接表数组，存储每个顶点的出边列表。

        // for (int s_i = 0; s_i < sockets; s_i++) {
        // outgoing_adj_bitmap[s_i] = new Bitmap(vertices);
        // outgoing_adj_bitmap[s_i]->clear();
        // for simulate
        // #ifdef CXL_SHM
        //             outgoing_adj_index[s_i] =
        //                 (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1),
        //                 REMOTE_NUMA);
        // #else
        //             outgoing_adj_index[s_i] = (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) *
        //             (vertices + 1),
        //                                                                  s_i + partition_id *
        //                                                                  NUMA);
        // #endif
        // }


        {
            // 接受线程接受所有其他一级分区的边，更新numa的CSR格式和每个节点内点的入度
            // 这里更新的是mirror点的CSR信息
            std::thread recv_thread_dst([&]() {
                int finished_count = 0;
                MPI_Status recv_status;
                while (finished_count < partitions) {
                    MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
                    int i = recv_status.MPI_SOURCE;
                    assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
                    int recv_bytes;
                    MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
                    if (recv_bytes == 1) {
                        finished_count += 1;
                        char c;
                        MPI_Recv(
                            &c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        continue;
                    }
                    assert(recv_bytes % edge_unit_size == 0);
                    int recv_edges = recv_bytes / edge_unit_size;
                    MPI_Recv(recv_buffer,
                             edge_unit_size * recv_edges,
                             MPI_CHAR,
                             i,
                             ShuffleGraph,
                             MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    // #pragma omp parallel for
                    for (EdgeId e_i = 0; e_i < recv_edges; e_i++) {
                        VertexId src = recv_buffer[e_i].src;
                        VertexId dst = recv_buffer[e_i].dst;
                        // 如果其他节点的发过来的边里面，dst在本节点的话，就更新本节点内mirror点的bitmap
                        //  index(应该有的是mirror，有的是master，只能确定dst一定是master)
                        assert(dst >= partition_offset[partition_id] &&
                               dst < partition_offset[partition_id + 1]);
                        int dst_part = get_local_partition_id(dst);   // 找到点所在在numa的id
                        // 检查并更新位图,更新邻接索引和入度
                        if (!outgoing_adj_bitmap[dst_part]->get_bit(src)) {
                            outgoing_adj_bitmap[dst_part]->set_bit(src);   // 该点存在出边
                            outgoing_adj_index[dst_part][src] = 0;
                        }
                        __sync_fetch_and_add(&outgoing_adj_index[dst_part][src],
                                             1);   // 每个socket内src的出边数
                        __sync_fetch_and_add(&in_degree[dst], 1);   // 更新每个节点内master点的入度
                    }
                    recv_outgoing_edges += recv_edges;
                }
            });

            for (int i = 0; i < partitions; i++) {
                buffered_edges[i] = 0;
            }
            assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
            read_bytes = 0;
            // 每次读一个CHUNKSIZE的边，读边是为了计算点的出度
            while (read_bytes < bytes_to_read) {
                long curr_read_bytes;
                if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
                    curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
                } else {
                    curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
                }
                assert(curr_read_bytes >= 0);
                read_bytes += curr_read_bytes;
                EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
                // 把读到每条边 根据其dst顶点发送到 所属的一级分区
                for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
                    VertexId dst = read_edge_buffer[e_i].dst;
                    int i = get_partition_id(dst);
                    memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i],
                           &read_edge_buffer[e_i],
                           edge_unit_size);
                    buffered_edges[i] += 1;
                    if (buffered_edges[i] ==
                        CHUNKSIZE) {   // 达到CHUNKSIZE就发送，因为sendbuffer的大小就是边大小乘CHUNKSIZE
                        MPI_Send(send_buffer[i].data(),
                                 edge_unit_size * buffered_edges[i],
                                 MPI_CHAR,
                                 i,
                                 ShuffleGraph,
                                 MPI_COMM_WORLD);
                        buffered_edges[i] = 0;
                    }
                }
            }
            // 继续发送send_buffer还没发完的数据
            for (int i = 0; i < partitions; i++) {
                if (buffered_edges[i] == 0) continue;
                MPI_Send(send_buffer[i].data(),
                         edge_unit_size * buffered_edges[i],
                         MPI_CHAR,
                         i,
                         ShuffleGraph,
                         MPI_COMM_WORLD);
                buffered_edges[i] = 0;
            }
            // 最后发\0表示已经发完
            //  TODO:
            for (int i = 0; i < partitions; i++) {
                char c = 0;
                MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            }
            recv_thread_dst.join();
#ifdef PRINT_DEBUG_MESSAGES
            printf("machine(%d) got %lu sparse mode edges\n", partition_id, recv_outgoing_edges);
#endif
        }

        compressed_outgoing_adj_vertices = new VertexId[sockets];
        compressed_outgoing_adj_index = new CompressedAdjIndexUnit*[sockets];
        size_t compressed_outgoing_adj_index_size = 0;
        size_t outgoing_adj_list_size = 0;
        for (int s_i = 0; s_i < sockets; s_i++) {
            outgoing_edges[s_i] = 0;   // 每个numa numa的出边数
            compressed_outgoing_adj_vertices[s_i] = 0;
            for (VertexId v_i = 0; v_i < vertices; v_i++) {   // 遍历所有的点
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                    outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];   // 每个socket出边总数
                    compressed_outgoing_adj_vertices[s_i] += 1;            // 有出边的点数
                }
            }
        }
        int max_compressed_outgoing_adj_vertices = 0;
        for (size_t i = 0; i < sockets; i++) {
            max_compressed_outgoing_adj_vertices =
                max_compressed_outgoing_adj_vertices > compressed_outgoing_adj_vertices[i]
                    ? max_compressed_outgoing_adj_vertices
                    : compressed_outgoing_adj_vertices[i];
        }
    int global_max=0;
        MPI_Allreduce(&max_compressed_outgoing_adj_vertices,
                      &global_max,
                      1,
                      MPI_INT,
                      MPI_MAX,
                      MPI_COMM_WORLD);
        gim_compressed_outgoing_adj_index = new CompressedAdjIndexUnit**[partitions];
        for (size_t p_i = 0; p_i < partitions; p_i++) {
            gim_compressed_outgoing_adj_index[p_i] = new CompressedAdjIndexUnit*[sockets];
            for (size_t s_i = 0; s_i < sockets; s_i++) {
                gim_compressed_outgoing_adj_index[p_i][s_i] =
                    (CompressedAdjIndexUnit*)cxl_shm->CXL_SHM_malloc(
                        (global_max + 1) * sizeof(CompressedAdjIndexUnit));
                // gim_incoming_adj_list[p_i][s_i] = (AdjUnit<EdgeData>*)malloc(
                //     global_max*unit_size);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        compressed_outgoing_adj_index = gim_compressed_outgoing_adj_index[partition_id];
        MPI_Barrier(MPI_COMM_WORLD);

        for (int s_i = 0; s_i < sockets; s_i++) {   // 遍历每个numa

            // for simulate
            // compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit*)numa_alloc_onnode(
            //     sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1),
            //     s_i + partition_id * NUMA);
            compressed_outgoing_adj_index_size +=
                sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1);
            compressed_outgoing_adj_index[s_i][0].index = 0;
            EdgeId last_e_i = 0;
            compressed_outgoing_adj_vertices[s_i] = 0;
            for (VertexId v_i = 0; v_i < vertices; v_i++) {
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {   // 如果mirror v_i有出边
                    outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
                    last_e_i = outgoing_adj_index[s_i][v_i];
                    compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]]
                        .vertex = v_i;
                    compressed_outgoing_adj_vertices[s_i] += 1;
                    compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]]
                        .index = last_e_i;
                }
            }
            for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i];
                 p_v_i++) {   // 对于socket内每一条出边
                VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;   // master点
                // 利用compressed_outgoing_adj_index出边的索引重新更新outgoing_adj_index的索引
                // 前面的代码中outgoing_adj_index记录的不是索引，而是出边数，现在才是真正的索引，你是会写代码的
                outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
                outgoing_adj_index[s_i][v_i + 1] =
                    compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
            }
#ifdef PRINT_DEBUG_MESSAGES
            printf("part(%d) E_%d has %lu sparse mode edges\n",
                   partition_id,
                   s_i,
                   outgoing_edges[s_i]);
#endif

            // gim_outgoing_adj_list[partition_id][s_i]= (AdjUnit<EdgeData>*)numa_alloc_onnode(
            //         unit_size * outgoing_edges[s_i], s_i + partition_id * NUMA);
            // for simulate
            // #ifdef CXL_SHM
            //                 outgoing_adj_list[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(
            //                     unit_size * outgoing_edges[s_i], REMOTE_NUMA);
            // #else
            //                 outgoing_adj_list[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(
            //                     unit_size * outgoing_edges[s_i], s_i + partition_id * NUMA);
            // #endif
            outgoing_adj_list_size += unit_size * outgoing_edges[s_i];
        }

        int max_out_going_edges = 0;
        for (size_t i = 0; i < sockets; i++) {
            max_out_going_edges =
                max_out_going_edges > outgoing_edges[i] ? max_out_going_edges : outgoing_edges[i];
        }
         
        MPI_Allreduce(&max_out_going_edges, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);


        gim_outgoing_adj_list = new AdjUnit<EdgeData>**[partitions];
        for (size_t p_i = 0; p_i < partitions; p_i++) {
            gim_outgoing_adj_list[p_i] = new AdjUnit<EdgeData>*[sockets];
            for (size_t s_i = 0; s_i < sockets; s_i++) {
                gim_outgoing_adj_list[p_i][s_i] =
                    (AdjUnit<EdgeData>*)cxl_shm->CXL_SHM_malloc(global_max * unit_size);
                // gim_outgoing_adj_list[p_i][s_i] = (AdjUnit<EdgeData>*)malloc(
                //     global_max*unit_size);
            }
        }
        outgoing_adj_list = gim_outgoing_adj_list[partition_id];
        // calculate size
        data_size["outgoing_adj_list"] = outgoing_adj_list_size;
        data_size["compressed_outgoing_adj_index"] = compressed_outgoing_adj_index_size;
        // if (partition_id==0) {
        //   printf("outgoing_adj_list_size:%d\n",outgoing_adj_list_size/1024/1024);
        //   printf("compressed_outgoing_adj_index_size:%d\n",compressed_outgoing_adj_index_size/1024/1024);
        // }

        {
            std::thread recv_thread_dst([&]() {
                int finished_count = 0;
                MPI_Status recv_status;
                while (finished_count < partitions) {
                    MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
                    int i = recv_status.MPI_SOURCE;
                    assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
                    int recv_bytes;
                    MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
                    if (recv_bytes == 1) {
                        finished_count += 1;
                        char c;
                        MPI_Recv(
                            &c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        continue;
                    }
                    assert(recv_bytes % edge_unit_size == 0);
                    int recv_edges = recv_bytes / edge_unit_size;
                    MPI_Recv(recv_buffer,
                             edge_unit_size * recv_edges,
                             MPI_CHAR,
                             i,
                             ShuffleGraph,
                             MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
#pragma omp parallel for
                    for (EdgeId e_i = 0; e_i < recv_edges; e_i++) {
                        VertexId src = recv_buffer[e_i].src;
                        VertexId dst = recv_buffer[e_i].dst;
                        assert(dst >= partition_offset[partition_id] &&
                               dst <
                                   partition_offset[partition_id + 1]);   // 如果dst是本host的master
                        int dst_part = get_local_partition_id(dst);
                        EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
                        outgoing_adj_list[dst_part][pos].neighbour = dst;   // 每个socket内
                        if (!std::is_same<EdgeData,
                                          Empty>::value) {   // 如果EdgeData不是空要初始化edge_data
                            outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
                        }
                    }
                }
            });
            for (int i = 0; i < partitions; i++) {
                buffered_edges[i] = 0;
            }
            assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
            read_bytes = 0;
            while (read_bytes < bytes_to_read) {
                long curr_read_bytes;
                if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
                    curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
                } else {
                    curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
                }
                assert(curr_read_bytes >= 0);
                read_bytes += curr_read_bytes;
                EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
                for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
                    VertexId dst = read_edge_buffer[e_i].dst;
                    int i = get_partition_id(dst);   // 跟上面一样也是发送边到dst所在节点
                    memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i],
                           &read_edge_buffer[e_i],
                           edge_unit_size);
                    buffered_edges[i] += 1;
                    if (buffered_edges[i] == CHUNKSIZE) {
                        MPI_Send(send_buffer[i].data(),
                                 edge_unit_size * buffered_edges[i],
                                 MPI_CHAR,
                                 i,
                                 ShuffleGraph,
                                 MPI_COMM_WORLD);
                        buffered_edges[i] = 0;
                    }
                }
            }
            // 发送没发完的
            for (int i = 0; i < partitions; i++) {
                if (buffered_edges[i] == 0) continue;
                MPI_Send(send_buffer[i].data(),
                         edge_unit_size * buffered_edges[i],
                         MPI_CHAR,
                         i,
                         ShuffleGraph,
                         MPI_COMM_WORLD);
                buffered_edges[i] = 0;
            }
            for (int i = 0; i < partitions; i++) {
                char c = 0;
                MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            }
            recv_thread_dst.join();
        }
        // 因为上面用outgoing_adj_index做别的用途，值改变了，这里重新赋回来，可读性极差的写法！
        for (int s_i = 0; s_i < sockets; s_i++) {
            for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++) {
                VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
                outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
                outgoing_adj_index[s_i][v_i + 1] =
                    compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);



        EdgeId recv_incoming_edges = 0;
        incoming_edges = new EdgeId[sockets];
        incoming_adj_index = new EdgeId*[sockets];
        incoming_adj_list = new AdjUnit<EdgeData>*[sockets];
        // incoming_adj_bitmap = new Bitmap*[sockets];

        for (int s_i = 0; s_i < sockets; s_i++) {
            // incoming_adj_bitmap[s_i] = new Bitmap(vertices);
            // incoming_adj_bitmap[s_i]->clear();
            // for simulate
            incoming_adj_index[s_i] = (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1),
                                                                 s_i + partition_id * NUMA);
        }


        gim_incoming_adj_bitmap = new Bitmap**[partitions];
        for (size_t p_i = 0; p_i < partitions; p_i++) {
            gim_incoming_adj_bitmap[p_i] = new Bitmap*[sockets];
            for (size_t s_i = 0; s_i < sockets; s_i++) {
                // gim_incoming_adj_bitmap[p_i][s_i] = new Bitmap(vertices);
                unsigned long* data = (unsigned long*)cxl_shm->GIM_malloc(
                    sizeof(unsigned long) * (WORD_OFFSET(vertices) + 1), p_i);
                gim_incoming_adj_bitmap[p_i][s_i] = new Bitmap(vertices, data);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        incoming_adj_bitmap = gim_incoming_adj_bitmap[partition_id];


        {
            std::thread recv_thread_src([&]() {
                int finished_count = 0;
                MPI_Status recv_status;
                while (finished_count < partitions) {
                    MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
                    int i = recv_status.MPI_SOURCE;
                    assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
                    int recv_bytes;
                    MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
                    if (recv_bytes == 1) {
                        finished_count += 1;
                        char c;
                        MPI_Recv(
                            &c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        continue;
                    }
                    assert(recv_bytes % edge_unit_size == 0);
                    int recv_edges = recv_bytes / edge_unit_size;
                    MPI_Recv(recv_buffer,
                             edge_unit_size * recv_edges,
                             MPI_CHAR,
                             i,
                             ShuffleGraph,
                             MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    // #pragma omp parallel for
                    for (EdgeId e_i = 0; e_i < recv_edges; e_i++) {
                        VertexId src = recv_buffer[e_i].src;
                        VertexId dst = recv_buffer[e_i].dst;
                        assert(src >= partition_offset[partition_id] &&
                               src < partition_offset[partition_id + 1]);   // 本host内master的出边
                        int src_part = get_local_partition_id(src);
                        // mirror的入边
                        if (!incoming_adj_bitmap[src_part]->get_bit(dst)) {
                            incoming_adj_bitmap[src_part]->set_bit(dst);   // 该dst存在入边
                            incoming_adj_index[src_part][dst] = 0;   // 记录dst入边的数量
                        }
                        __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
                    }
                    recv_incoming_edges += recv_edges;
                }
            });
            for (int i = 0; i < partitions; i++) {
                buffered_edges[i] = 0;
            }
            assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
            read_bytes = 0;
            while (read_bytes < bytes_to_read) {
                long curr_read_bytes;
                if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
                    curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
                } else {
                    curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
                }
                assert(curr_read_bytes >= 0);
                read_bytes += curr_read_bytes;
                EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
                for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
                    VertexId src = read_edge_buffer[e_i].src;
                    int i = get_partition_id(src);
                    memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i],
                           &read_edge_buffer[e_i],
                           edge_unit_size);
                    buffered_edges[i] += 1;
                    if (buffered_edges[i] == CHUNKSIZE) {
                        MPI_Send(send_buffer[i].data(),
                                 edge_unit_size * buffered_edges[i],
                                 MPI_CHAR,
                                 i,
                                 ShuffleGraph,
                                 MPI_COMM_WORLD);
                        buffered_edges[i] = 0;
                    }
                }
            }
            // 发送剩下的边
            for (int i = 0; i < partitions; i++) {
                if (buffered_edges[i] == 0) continue;
                MPI_Send(send_buffer[i].data(),
                         edge_unit_size * buffered_edges[i],
                         MPI_CHAR,
                         i,
                         ShuffleGraph,
                         MPI_COMM_WORLD);
                buffered_edges[i] = 0;
            }
            for (int i = 0; i < partitions; i++) {
                char c = 0;
                MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            }
            recv_thread_src.join();
#ifdef PRINT_DEBUG_MESSAGES
            printf("machine(%d) got %lu dense mode edges\n", partition_id, recv_incoming_edges);
#endif
        }
        compressed_incoming_adj_vertices = new VertexId[sockets];
        compressed_incoming_adj_index = new CompressedAdjIndexUnit*[sockets];
        size_t incoming_adj_list_size = 0;
        size_t compressed_incoming_adj_index_size = 0;

        // init incoming_edges and compressed_incoming_adj_vertices
        for (int s_i = 0; s_i < sockets; s_i++) {   // 遍历numa
            incoming_edges[s_i] = 0;
            compressed_incoming_adj_vertices[s_i] = 0;
            for (VertexId v_i = 0; v_i < vertices; v_i++) {                // 遍历所有的点
                if (incoming_adj_bitmap[s_i]->get_bit(v_i)) {              // 如果dst有入边
                    incoming_edges[s_i] += incoming_adj_index[s_i][v_i];   // 每个socket的入边总数
                    compressed_incoming_adj_vertices[s_i] += 1;            // 有入边的点数
                }
            }
        }
        int max_compressed_incoming_adj_vertices = 0;
        for (size_t i = 0; i < sockets; i++) {
            max_compressed_incoming_adj_vertices =
                max_compressed_incoming_adj_vertices > compressed_incoming_adj_vertices[i]
                    ? max_compressed_incoming_adj_vertices
                    : compressed_incoming_adj_vertices[i];
        }

        MPI_Allreduce(&max_compressed_incoming_adj_vertices,
                      &global_max,
                      1,
                      MPI_INT,
                      MPI_MAX,
                      MPI_COMM_WORLD);

        gim_compressed_incoming_adj_index = new CompressedAdjIndexUnit**[partitions];
        for (size_t p_i = 0; p_i < partitions; p_i++) {
            gim_compressed_incoming_adj_index[p_i] = new CompressedAdjIndexUnit*[sockets];
            for (size_t s_i = 0; s_i < sockets; s_i++) {
                gim_compressed_incoming_adj_index[p_i][s_i] =
                    (CompressedAdjIndexUnit*)cxl_shm->CXL_SHM_malloc(
                        (global_max + 1) * sizeof(CompressedAdjIndexUnit));
                // gim_incoming_adj_list[p_i][s_i] = (AdjUnit<EdgeData>*)malloc(
                //     global_max*unit_size);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        compressed_incoming_adj_index = gim_compressed_incoming_adj_index[partition_id];
        MPI_Barrier(MPI_COMM_WORLD);
        for (int s_i = 0; s_i < sockets; s_i++) {   // 遍历numa
            incoming_edges[s_i] = 0;
            compressed_incoming_adj_vertices[s_i] = 0;
            for (VertexId v_i = 0; v_i < vertices; v_i++) {                // 遍历所有的点
                if (incoming_adj_bitmap[s_i]->get_bit(v_i)) {              // 如果dst有入边
                    incoming_edges[s_i] += incoming_adj_index[s_i][v_i];   // 每个socket的入边总数
                    compressed_incoming_adj_vertices[s_i] += 1;            // 有入边的点数
                }
            }
            // for simulate
            // #ifdef CXL_SHM
            //             compressed_incoming_adj_index[s_i] =
            //             (CompressedAdjIndexUnit*)numa_alloc_onnode(
            //                 sizeof(CompressedAdjIndexUnit) *
            //                 (compressed_incoming_adj_vertices[s_i] + 1), REMOTE_NUMA);
            // #else
            //             compressed_incoming_adj_index[s_i] =
            //             (CompressedAdjIndexUnit*)numa_alloc_onnode(
            //                 sizeof(CompressedAdjIndexUnit) *
            //                 (compressed_incoming_adj_vertices[s_i] + 1), s_i + partition_id *
            //                 NUMA);
            // #endif
            compressed_incoming_adj_index_size +=
                sizeof(CompressedAdjIndexUnit) * (compressed_incoming_adj_vertices[s_i] + 1);
            compressed_incoming_adj_index[s_i][0].index = 0;
            EdgeId last_e_i = 0;
            compressed_incoming_adj_vertices[s_i] = 0;
            for (VertexId v_i = 0; v_i < vertices; v_i++) {     // 遍历所有的点
                if (incoming_adj_bitmap[s_i]->get_bit(v_i)) {   // 如果dst有入边
                    incoming_adj_index[s_i][v_i] = last_e_i + incoming_adj_index[s_i][v_i];
                    last_e_i = incoming_adj_index[s_i][v_i];
                    compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]]
                        .vertex = v_i;
                    compressed_incoming_adj_vertices[s_i] += 1;
                    compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]]
                        .index = last_e_i;
                }
            }
            for (VertexId p_v_i = 0; p_v_i < compressed_incoming_adj_vertices[s_i]; p_v_i++) {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
                incoming_adj_index[s_i][v_i + 1] =
                    compressed_incoming_adj_index[s_i][p_v_i + 1].index;
            }
#ifdef PRINT_DEBUG_MESSAGES
            printf(
                "part(%d) E_%d has %lu dense mode edges\n", partition_id, s_i, incoming_edges[s_i]);
#endif
            incoming_adj_list_size += unit_size * incoming_edges[s_i];
        }
        int max_incoming_edges = 0;
        for (size_t i = 0; i < sockets; i++) {
            max_incoming_edges =
                max_incoming_edges > incoming_edges[i] ? max_incoming_edges : incoming_edges[i];
        }
        global_max;
        MPI_Allreduce(&max_incoming_edges, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);


        gim_incoming_adj_list = new AdjUnit<EdgeData>**[partitions];
        for (size_t p_i = 0; p_i < partitions; p_i++) {
            gim_incoming_adj_list[p_i] = new AdjUnit<EdgeData>*[sockets];
            for (size_t s_i = 0; s_i < sockets; s_i++) {
                gim_incoming_adj_list[p_i][s_i] =
                    (AdjUnit<EdgeData>*)cxl_shm->CXL_SHM_malloc(global_max * unit_size);
                // gim_incoming_adj_list[p_i][s_i] = (AdjUnit<EdgeData>*)malloc(
                //     global_max*unit_size);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        incoming_adj_list = gim_incoming_adj_list[partition_id];
        data_size["incoming_adj_list"] = incoming_adj_list_size;
        data_size["compressed_incoming_adj_index"] = compressed_incoming_adj_index_size;
        // if (partition_id==0) {
        //   printf("incoming_adj_list_size:%d\n",incoming_adj_list_size/1024/1024);
        //   printf("compressed_incoming_adj_index_size:%d\n",compressed_incoming_adj_index_size/1024/1024);
        // }

        {
            std::thread recv_thread_src([&]() {
                int finished_count = 0;
                MPI_Status recv_status;
                while (finished_count < partitions) {
                    MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
                    int i = recv_status.MPI_SOURCE;
                    assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
                    int recv_bytes;
                    MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
                    if (recv_bytes == 1) {
                        finished_count += 1;
                        char c;
                        MPI_Recv(
                            &c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        continue;
                    }
                    assert(recv_bytes % edge_unit_size == 0);
                    int recv_edges = recv_bytes / edge_unit_size;
                    MPI_Recv(recv_buffer,
                             edge_unit_size * recv_edges,
                             MPI_CHAR,
                             i,
                             ShuffleGraph,
                             MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
#pragma omp parallel for
                    for (EdgeId e_i = 0; e_i < recv_edges; e_i++) {
                        VertexId src = recv_buffer[e_i].src;
                        VertexId dst = recv_buffer[e_i].dst;
                        assert(src >= partition_offset[partition_id] &&
                               src < partition_offset[partition_id + 1]);
                        int src_part = get_local_partition_id(src);
                        EdgeId pos = __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
                        incoming_adj_list[src_part][pos].neighbour = src;
                        if (!std::is_same<EdgeData, Empty>::value) {
                            incoming_adj_list[src_part][pos].edge_data = recv_buffer[e_i].edge_data;
                        }
                    }
                }
            });
            for (int i = 0; i < partitions; i++) {
                buffered_edges[i] = 0;
            }
            assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
            read_bytes = 0;
            while (read_bytes < bytes_to_read) {
                long curr_read_bytes;
                if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
                    curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
                } else {
                    curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
                }
                assert(curr_read_bytes >= 0);
                read_bytes += curr_read_bytes;
                EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
                for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
                    VertexId src = read_edge_buffer[e_i].src;
                    int i = get_partition_id(src);
                    memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i],
                           &read_edge_buffer[e_i],
                           edge_unit_size);
                    buffered_edges[i] += 1;
                    if (buffered_edges[i] == CHUNKSIZE) {
                        MPI_Send(send_buffer[i].data(),
                                 edge_unit_size * buffered_edges[i],
                                 MPI_CHAR,
                                 i,
                                 ShuffleGraph,
                                 MPI_COMM_WORLD);
                        buffered_edges[i] = 0;
                    }
                }
            }
            // 发送没发完的
            for (int i = 0; i < partitions; i++) {
                if (buffered_edges[i] == 0) continue;
                MPI_Send(send_buffer[i].data(),
                         edge_unit_size * buffered_edges[i],
                         MPI_CHAR,
                         i,
                         ShuffleGraph,
                         MPI_COMM_WORLD);
                buffered_edges[i] = 0;
            }
            for (int i = 0; i < partitions; i++) {
                char c = 0;
                MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            }
            recv_thread_src.join();
        }
        for (int s_i = 0; s_i < sockets; s_i++) {
            for (VertexId p_v_i = 0; p_v_i < compressed_incoming_adj_vertices[s_i]; p_v_i++) {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
                incoming_adj_index[s_i][v_i + 1] =
                    compressed_incoming_adj_index[s_i][p_v_i + 1].index;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        delete[] buffered_edges;
        delete[] send_buffer;
        delete[] read_edge_buffer;
        delete[] recv_buffer;
        close(fin);

        transpose();
        tune_chunks();   // tuned_chunks_dense init
        transpose();     // exchange tuned_chunks_dense and tuned_chunks_sparse
        // tuned_chunks_sparse = tuned_chunks_dense
        tune_chunks();   // tuned_chunks_dense init
        // gim_buffer
        init_gim_buffer();
        prep_time += MPI_Wtime();

#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 0) {
            printf("preprocessing cost: %.2lf (s)\n", prep_time);
        }
#endif
        }

    // 为线程分配chunk工作块，确定每个线程负责哪些点
    void tune_chunks() {
        tuned_chunks_dense = new ThreadState*[partitions];
        int current_send_part_id = partition_id;
        for (int step = 0; step < partitions; step++) {   // 遍历每个host
            current_send_part_id = (current_send_part_id + 1) % partitions;
            int i = current_send_part_id;
            tuned_chunks_dense[i] = new ThreadState[threads];   // 为当前分区创建ThreadState
            EdgeId remained_edges;                      // 当前分区中剩余的边数量。
            int remained_partitions;                    // 剩余分配的分区数。
            VertexId last_p_v_i;                        // 记录当前线程开始处理的顶点
            VertexId end_p_v_i;                         // 记录当前线程结束处理的顶点
            for (int t_i = 0; t_i < threads; t_i++) {   // 遍历host的每个线程
                tuned_chunks_dense[i][t_i].status = WORKING;   // 赋值ststus为WORKING
                int s_i = get_socket_id(t_i);                  // 线程在第几个socket
                int s_j = get_socket_offset(t_i);              // 线程是本socket的第几个
                if (s_j == 0) {
                    VertexId p_v_i = 0;
                    while (p_v_i < compressed_incoming_adj_vertices[s_i]) {
                        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                        if (v_i >= partition_offset[i]) {
                            break;
                        }
                        p_v_i++;
                    }
                    last_p_v_i = p_v_i;
                    while (p_v_i < compressed_incoming_adj_vertices[s_i]) {
                        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                        if (v_i >= partition_offset[i + 1]) {
                            break;
                        }
                        p_v_i++;
                    }
                    end_p_v_i = p_v_i;
                    // 确定当前分区 i 的顶点范围，last_p_v_i 是起点，end_p_v_i
                    // 是终点。这个范围内的顶点由当前分区处理。
                    remained_edges = 0;
                    for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++) {
                        remained_edges += compressed_incoming_adj_index[s_i][p_v_i + 1].index -
                                          compressed_incoming_adj_index[s_i][p_v_i].index;
                        remained_edges +=
                            alpha;   // 计算当前分区和线程的剩余边的数量，包括预留的 alpha
                    }   // 计算在该分区内剩余的边数，使用顶点 p_v_i
                        // 的相邻边索引之差得到边的数量，并将其累计到 remained_edges。alpha
                        // 是一个边的附加数量，可能用于任务平衡。
                }
                tuned_chunks_dense[i][t_i].curr = last_p_v_i;
                tuned_chunks_dense[i][t_i].end = last_p_v_i;   // 线程处理点的范围
                remained_partitions = threads_per_socket - s_j;
                EdgeId expected_chunk_size =
                    remained_edges / remained_partitions;   // 表示当前线程预计要处理的边数量

                if (remained_partitions ==
                    1) {   // 如果只剩一个分区要分配，直接将当前线程的终止顶点设置为 end_p_v_i
                    tuned_chunks_dense[i][t_i].end = end_p_v_i;
                } else {
                    EdgeId got_edges = 0;
                    for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++) {
                        got_edges += compressed_incoming_adj_index[s_i][p_v_i + 1].index -
                                     compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
                        if (got_edges >= expected_chunk_size) {
                            tuned_chunks_dense[i][t_i].end = p_v_i;   // 更新每个线程的终止顶点;
                            last_p_v_i =
                                tuned_chunks_dense[i][t_i].end;   // 更新下一个线程的起始顶点;
                            break;
                        }
                    }
                    got_edges = 0;
                    for (VertexId p_v_i = tuned_chunks_dense[i][t_i].curr;
                         p_v_i < tuned_chunks_dense[i][t_i].end;
                         p_v_i++) {
                        got_edges += compressed_incoming_adj_index[s_i][p_v_i + 1].index -
                                     compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
                    }
                    remained_edges -= got_edges;
                }
            }
        }
        // if(partition_id==0){
        //   for (int i=0;i<partitions;i++){
        //     for(int j=0;j<threads;j++){
        //       printf("part:%d thread:%d curr:%d
        //       end:%d\n",i,j,tuned_chunks_dense[i][j].curr,tuned_chunks_dense[i][j].end);
        //     }
        //   }
        // }
    }

    // 对于acrive中的点执行process任务，启用了工作窃取
    //  process vertices
    template<typename R> R process_vertices(std::function<R(VertexId)> process, Bitmap* active) {
        double stream_time = 0;
        stream_time -= MPI_Wtime();

        R reducer = 0;
        size_t basic_chunk = 64;   // 每次处理的顶点数，和WORD的位数是对应的
        for (int t_i = 0; t_i < threads; t_i++) {
            int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = local_partition_offset[s_i + 1] -
                                      local_partition_offset[s_i];   // 每个分区的点的数量
            thread_state[t_i]->curr =
                local_partition_offset[s_i] +
                partition_size / threads_per_socket / basic_chunk * basic_chunk *
                    s_j;   // 设置线程的curr，处理点的起始位置，和basic_chunk对齐
            thread_state[t_i]->end =
                local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk *
                                                  basic_chunk * (s_j + 1);   // 设置线程的end
            if (s_j == threads_per_socket - 1) {
                thread_state[t_i]->end = local_partition_offset[s_i + 1];   // 设置最后一个线程的end
            }
            thread_state[t_i]->status = WORKING;
        }
#pragma omp parallel reduction(+ : reducer)
        {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            while (true) {
                VertexId v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
                if (v_i >=
                    thread_state[thread_id]->end)   // 遍历当前线程要处理的点，一次处理basic_chunk
                    break;
                unsigned long word =
                    active->data[WORD_OFFSET(v_i)];   // 根据Bitmap *active决定是否执行process
                while (word != 0) {
                    if (word & 1) {
                        local_reducer += process(v_i);
                    }
                    v_i++;
                    word = word >> 1;
                }
            }
            // 当前线程的任务处理完后，将状态设置为 STEALING，表示它可能会去“偷取”其他线程的任务
            thread_state[thread_id]->status = STEALING;
            for (int t_offset = 1; t_offset < threads; t_offset++) {
                int t_i = (thread_id + t_offset) % threads;
                while (thread_state[t_i]->status !=
                       STEALING) {   // 如果目标线程的状态不是 STEALING，则尝试窃取工作。
                    VertexId v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                    if (v_i >= thread_state[t_i]->end) continue;
                    unsigned long word = active->data[WORD_OFFSET(v_i)];
                    while (word != 0) {
                        if (word & 1) {
                            local_reducer += process(v_i);
                        }
                        v_i++;
                        word = word >> 1;
                    }
                }
            }
            reducer += local_reducer;
        }
        // 当本host的任务都完成，开始全局规约
        R global_reducer;
        MPI_Datatype dt = get_mpi_data_type<R>();
        MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
        stream_time += MPI_Wtime();
        total_process_time += stream_time;
#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 0) {
            printf("process_vertices took %lf (s)\n", stream_time);
        }
#endif
        return global_reducer;
    }

    template<typename R>
    R process_vertices_global(std::function<R(VertexId, int)> process, Bitmap** active) {
        double stream_time = 0;
        stream_time -= MPI_Wtime();

        R reducer = 0;
        size_t basic_chunk = 64;   // 每次处理的顶点数，和WORD的位数是对应的
        for (int t_i = 0; t_i < threads; t_i++) {
            int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = local_partition_offset[s_i + 1] -
                                      local_partition_offset[s_i];   // 每个分区的点的数量
            thread_state[t_i]->curr =
                local_partition_offset[s_i] +
                partition_size / threads_per_socket / basic_chunk * basic_chunk *
                    s_j;   // 设置线程的curr，处理点的起始位置，和basic_chunk对齐
            thread_state[t_i]->end =
                local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk *
                                                  basic_chunk * (s_j + 1);   // 设置线程的end
            if (s_j == threads_per_socket - 1) {
                thread_state[t_i]->end = local_partition_offset[s_i + 1];   // 设置最后一个线程的end
            }
            thread_state[t_i]->status = WORKING;
        }
#pragma omp parallel reduction(+ : reducer)
        {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            while (true) {
                VertexId v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
                if (v_i >=
                    thread_state[thread_id]->end)   // 遍历当前线程要处理的点，一次处理basic_chunk
                    break;
                unsigned long word =
                    active[partition_id]
                        ->data[WORD_OFFSET(v_i)];   // 根据Bitmap *active决定是否执行process
                while (word != 0) {
                    if (word & 1) {
                        local_reducer += process(v_i, -1);
                    }
                    v_i++;
                    word = word >> 1;
                }
            }
            // 当前线程的任务处理完后，将状态设置为 STEALING，表示它可能会去“偷取”其他线程的任务
            thread_state[thread_id]->status = STEALING;
            for (int t_offset = 1; t_offset < threads; t_offset++) {
                int t_i = (thread_id + t_offset) % threads;
                while (thread_state[t_i]->status !=
                       STEALING) {   // 如果目标线程的状态不是 STEALING，则尝试窃取工作。
                    VertexId v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                    if (v_i >= thread_state[t_i]->end) continue;
                    unsigned long word = active[partition_id]->data[WORD_OFFSET(v_i)];
                    while (word != 0) {
                        if (word & 1) {
                            local_reducer += process(v_i, -1);
                        }
                        v_i++;
                        word = word >> 1;
                    }
                }
            }
            reducer += local_reducer;
        }
        // 全局工作窃取
        for (int step = 1; step < partitions; step++) {
            int i = (partition_id - step + partitions) % partitions;
#pragma omp parallel reduction(+ : reducer)
            {
                R local_reducer = 0;
                int thread_id = omp_get_thread_num();
                for (int t_offset = 0; t_offset < threads; t_offset++) {
                    int t_i = (thread_id + t_offset) % threads;
                    while (gim_thread_state[i][t_i]->status !=
                           STEALING) {   // 如果目标线程的状态不是 STEALING，则尝试窃取工作。
                        VertexId v_i =
                            __sync_fetch_and_add(&gim_thread_state[i][t_i]->curr, basic_chunk);
                        if (v_i >= gim_thread_state[i][t_i]->end) continue;
                        unsigned long word = active[i]->data[WORD_OFFSET(v_i)];
                        while (word != 0) {
                            if (word & 1) {
                                local_reducer += process(v_i, i);
                            }
                            v_i++;
                            word = word >> 1;
                        }
                    }
                }
                reducer += local_reducer;
            }
        }
        // 当本host的任务都完成，开始全局规约
        R global_reducer;
        MPI_Datatype dt = get_mpi_data_type<R>();
        MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
        stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 0) {
            printf("process_vertices took %lf (s)\n", stream_time);
        }
#endif
        return global_reducer;
    }

    // 将本地线程的发送缓冲区的数据刷新到全局发送缓冲区
    // 全局的send_buffer怎么知道local_send_bufferd的位置呢，不是每个大小都是 local_send_buffer_limit
    //  template<typename M> void flush_local_send_buffer(int t_i) {
    //      int s_i = get_socket_id(t_i);   //线程在那个socket
    //      int pos = __sync_fetch_and_add(
    //          &send_buffer[current_send_part_id][s_i]->count,
    //          local_send_buffer[t_i]
    //              ->count);   //通过faa找到全局的send_buffer[current_send_part_id] [t_i]的指针
    //      memcpy(send_buffer[current_send_part_id][s_i]->data + sizeof(MsgUnit<M>) * pos,
    //             local_send_buffer[t_i]->data,
    //             sizeof(MsgUnit<M>) * local_send_buffer[t_i]->count);
    //      local_send_buffer[t_i]->count = 0;
    //  }

    template<typename M> void flush_local_send_buffer(int t_i) {
        int s_i = get_socket_id(t_i);   // 线程在那个socket
        int pos = __sync_fetch_and_add(
            &send_count[partition_id][current_send_part_id][s_i],
            local_send_buffer[t_i]
                ->count);   // 通过faa找到全局的send_buffer[current_send_part_id] [t_i]的指针
        memcpy(gim_send_buffer[partition_id][current_send_part_id][s_i]->data +
                   sizeof(MsgUnit<M>) * pos,
               local_send_buffer[t_i]->data,
               sizeof(MsgUnit<M>) * local_send_buffer[t_i]->count);
        local_send_buffer[t_i]->count = 0;
    }
    // 向特定的顶点发送消息。它首先将消息及其目标顶点 ID 存储到当前线程的本地发送缓冲区中。
    // 如果缓冲区达到限制，则会调用刷新函数将消息发送到全局缓冲区。
    //  emit a message to a vertex's master (dense) / mirror (sparse)
    template<typename M> void emit(VertexId vtx, M msg) {
        int t_i = omp_get_thread_num();
        MsgUnit<M>* buffer = (MsgUnit<M>*)local_send_buffer[t_i]->data;
        buffer[local_send_buffer[t_i]->count].vertex = vtx;
        buffer[local_send_buffer[t_i]->count].msg_data = msg;
        local_send_buffer[t_i]->count += 1;
        if (local_send_buffer[t_i]->count == local_send_buffer_limit) {
            flush_local_send_buffer<M>(t_i);
        }
    }

    // need rewrite current_send_part_id ,make it shared
    template<typename M> void flush_local_send_buffer_to_other(int t_i, int id) {
        int s_i = get_socket_id(t_i);   // 线程在那个socket
        // printf("%d steal %d current part%d\n",
        //        partition_id,
        //        id,
        //        global_current_send_part_id[id].load());
        int pos = __sync_fetch_and_add(
            &send_count[id][global_current_send_part_id[id]][s_i],
            local_send_buffer[t_i]
                ->count);   // 通过faa找到全局的send_buffer[current_send_part_id] [t_i]的指针
        memcpy(gim_send_buffer[id][global_current_send_part_id[id]][s_i]->data +
                   sizeof(MsgUnit<M>) * pos,
               local_send_buffer[t_i]->data,
               sizeof(MsgUnit<M>) * local_send_buffer[t_i]->count);
        local_send_buffer[t_i]->count = 0;
    }
    template<typename M> void emit_other(VertexId vtx, M msg, int partition_id) {
        int t_i = omp_get_thread_num();
        MsgUnit<M>* buffer = (MsgUnit<M>*)local_send_buffer[t_i]->data;
        buffer[local_send_buffer[t_i]->count].vertex = vtx;
        buffer[local_send_buffer[t_i]->count].msg_data = msg;
        local_send_buffer[t_i]->count += 1;
        if (local_send_buffer[t_i]->count == local_send_buffer_limit) {
            flush_local_send_buffer_to_other<M>(t_i, partition_id);
        }
    }


    // process edges
    template<typename R, typename M>
    R process_edges(
        std::function<void(VertexId)> sparse_signal,
        std::function<R(VertexId, M, VertexAdjList<EdgeData>, int partiton_id)> sparse_slot,
        std::function<void(VertexId, VertexAdjList<EdgeData>, int partiton_id)> dense_signal,
        std::function<R(VertexId, M)> dense_slot, Bitmap* active,
        Bitmap* dense_selective = nullptr) {
        double stream_time = 0;
        stream_time -= MPI_Wtime();
        for (int i = 0; i < 4; i++) {
            process_edge_time[i] = 0;
        }
        size_t local_send_buffer_size = 0;
        for (int t_i = 0; t_i < threads; t_i++) {   // 调整local_send_buffer的大小
            local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
            local_send_buffer_size += sizeof(MsgUnit<M>) * local_send_buffer_limit;
            local_send_buffer[t_i]->count = 0;
        }

        R reducer = 0;
        EdgeId active_edges = process_vertices<EdgeId>(   // 计算active中的点的出度之和
            [&](VertexId vtx) { return (EdgeId)out_degree[vtx]; },
            active);
        bool sparse = (active_edges < edges / 20);

        // if (partition_id == 0) printf("spare:%d\n", sparse);
        size_t send_buffer_size = 0;
        size_t recv_buffer_size = 0;
        if (sparse) {
            for (int i = 0; i < partitions; i++) {   // 稀疏模式每个host要向外发送数据
                for (int s_i = 0; s_i < sockets; s_i++) {

                    recv_buffer[i][s_i]->resize(
                        sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) *
                        sockets);   // 接受区间是msgu的大小乘socket数乘第i个host拥有点数
                    send_buffer[partition_id][s_i]->resize(
                        sizeof(MsgUnit<M>) * owned_vertices *
                        sockets);   // 发送区间都一样，msgu的大小乘socket数*自己拥有的点
                    send_buffer[partition_id][s_i]->count = 0;
                    recv_buffer[i][s_i]->count = 0;

                    gim_recv_buffer[partition_id][i][s_i]->resize(
                        sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) *
                        sockets);   // 接受区间是msgu的大小乘socket数乘第i个host拥有点数
                    gim_send_buffer[partition_id][partition_id][s_i]->resize(
                        sizeof(MsgUnit<M>) * owned_vertices *
                        sockets);   // 发送区间都一样，msgu的大小乘socket数*自己拥有的点
                    send_count[partition_id][i][s_i] = 0;
                    gim_send_buffer[partition_id][partition_id][s_i]->count = 0;
                    gim_recv_buffer[partition_id][i][s_i]->count = 0;

                    send_buffer_size += sizeof(MsgUnit<M>) * owned_vertices * sockets;
                    recv_buffer_size += sizeof(MsgUnit<M>) *
                                        (partition_offset[i + 1] - partition_offset[i]) * sockets;
#ifdef UNIDIRECTIONAL_MODE
                    // placement new
                    // 需要显式调用析构函数，它直接构造对象，不进行内存分配，但是没有用new管理导致不会被析构，用delete导致未定义行为
                    new (completion_tags[partition_id][i][s_i]) std::atomic<bool>(false);
#endif
                }
            }
        } else {
            for (int i = 0; i < partitions; i++) {   // 稠密模式每个host要接受数据
                for (int s_i = 0; s_i < sockets; s_i++) {


                    recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices *
                                                sockets);   // 跟上面相反
                    send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) *
                                                (partition_offset[i + 1] - partition_offset[i]) *
                                                sockets);
                    send_buffer[i][s_i]->count = 0;
                    recv_buffer[i][s_i]->count = 0;

                    gim_recv_buffer[partition_id][i][s_i]->resize(
                        sizeof(MsgUnit<M>) * owned_vertices *
                        sockets);   // 接受区间是msgu的大小乘socket数乘第i个host拥有点数
                    gim_send_buffer[partition_id][partition_id][s_i]->resize(
                        sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) *
                        sockets);   // 发送区间都一样，msgu的大小乘socket数*自己拥有的点
                    send_count[partition_id][i][s_i] = 0;
                    gim_send_buffer[partition_id][i][s_i]->count = 0;
                    gim_recv_buffer[partition_id][i][s_i]->count = 0;
                    recv_buffer_size += sizeof(MsgUnit<M>) * owned_vertices * sockets;
                    send_buffer_size += sizeof(MsgUnit<M>) *
                                        (partition_offset[i + 1] - partition_offset[i]) * sockets;
#ifdef UNIDIRECTIONAL_MODE
                    new (completion_tags[partition_id][i][s_i]) std::atomic<bool>(false);
#endif
                }
            }
        }
        data_size["local_send_buffer"] = local_send_buffer_size;
        data_size["send_buffer"] = send_buffer_size;
        data_size["recv_buffer"] = recv_buffer_size;
        // if (partition_id==0) {
        // printf("process_edge:local_send_buffer_size:%d\n",local_send_buffer_size);
        // printf("process_edge:send_buffer_size:%d,recv_buffer_size:%d\n",send_buffer_size/1024/1024,recv_buffer_size/1024/1024);
        // }
        size_t basic_chunk = 64;
#if defined(SPARSE_MODE_UNIDIRECTIONAL) || defined(DENSE_MODE_UNIDIRECTIONAL)
        // 初始化完成后再进行操作
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        if (sparse) {
#ifdef PRINT_DEBUG_MESSAGES
            if (partition_id == 0) {
                printf("sparse mode\n");
            }
#endif
            int* recv_queue = new int[partitions];
            int recv_queue_size = 0;
            std::mutex recv_queue_mutex;

            current_send_part_id = partition_id;
            // 对自己host的点进行遍历，以basic_chunk为单位
#pragma omp parallel for
            for (VertexId begin_v_i = partition_offset[partition_id];
                 begin_v_i < partition_offset[partition_id + 1];
                 begin_v_i += basic_chunk) {
                VertexId v_i = begin_v_i;
                unsigned long word = active->data[WORD_OFFSET(v_i)];
                while (word != 0) {
                    if (word & 1) {
                        sparse_signal(v_i);   // 每个点都执行sparse_signal
                    }
                    v_i++;
                    word = word >> 1;
                }
            }
#pragma omp parallel for
            for (int t_i = 0; t_i < threads; t_i++) {
                flush_local_send_buffer<M>(t_i);
            }
            // 此时current_send_part_id的send_buffer准备好了
            process_edge_time[0] = MPI_Wtime() + stream_time;

#ifdef SPARSE_MODE_UNIDIRECTIONAL
            std::thread comm_thread([&]() {
                for (int step = 1; step < partitions; step++) {
                    int i = (partition_id - step + partitions) %
                            partitions;   // 确保i进程是除了自己以外的所有进程
                    for (int s_i = 0; s_i < sockets; s_i++) {
                        // memcpy的读操作是线程安全的
                        memcpy(gim_recv_buffer[i][partition_id][s_i]->data,
                               gim_send_buffer[partition_id][partition_id][s_i]->data,
                               sizeof(MsgUnit<M>) *
                                   send_count[partition_id][partition_id][s_i]);
                        // if(sizeof(MsgUnit<M>)
                        // *send_count[partition_id][partition_id][s_i]>0)

                        // ll_DMA_memcpy((uint8_t*)gim_send_buffer[partition_id][partition_id][s_i]->data,
                        //                 (uint8_t*)gim_recv_buffer[i][partition_id][s_i]->data,
                        //                 sizeof(MsgUnit<M>) *
                        // send_count[partition_id][partition_id][s_i],partition_id );

                        length_array[i][partition_id][s_i] =
                            send_count[partition_id][partition_id][s_i];
                        completion_tags[i][partition_id][s_i]->store(true,
                                                                     std::memory_order_release);
                    }
                }
            });

            int expected_partition = partition_id + 1;
            // 标记有接收自己的消息
            completion_tags[partition_id][partition_id][0]->store(true, std::memory_order_relaxed);
#else
            recv_queue[recv_queue_size] = partition_id;
            recv_queue_mutex.lock();   // 为啥这里用锁了不用原子变量
            recv_queue_size += 1;
            recv_queue_mutex.unlock();

            std::thread send_thread([&]() {
                for (int step = 1; step < partitions; step++) {   // 遍历host
                    int i = (partition_id - step + partitions) %
                            partitions;   // 确保i进程是除了自己以外的所有进程
                    for (int s_i = 0; s_i < sockets; s_i++) {   // 遍历所有socket
                        // MPI_Send(send_buffer[partition_id][s_i]->data,
                        //          sizeof(MsgUnit<M>) * send_buffer[partition_id][s_i]->count,
                        //          MPI_CHAR,
                        //          i,
                        //          PassMessage,
                        //          MPI_COMM_WORLD);

                        MPI_Send(gim_send_buffer[partition_id][partition_id][s_i]->data,
                                 sizeof(MsgUnit<M>) * send_count[partition_id][partition_id][s_i],
                                 MPI_CHAR,
                                 i,
                                 PassMessage,
                                 MPI_COMM_WORLD);
                        // gim_comm->GIM_Send(
                        //     gim_send_buffer[partition_id][partition_id][s_i]->data,
                        //     sizeof(MsgUnit<M>) *
                        //         send_count[partition_id][partition_id][s_i],
                        //     i,
                        //     0,
                        //     gim_recv_buffer[i][partition_id][s_i]->data);
                    }
                }
            });
            std::thread recv_thread([&]() {
                for (int step = 1; step < partitions; step++) {   // 遍历host
                    int i = (partition_id + step) % partitions;   // 除了自己以外的所有进程
                    for (int s_i = 0; s_i < sockets; s_i++) {     // 遍历所有socket
                        MPI_Status recv_status;
                        MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
                        // MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
                        // MPI_Recv(recv_buffer[i][s_i]->data,
                        //          recv_buffer[i][s_i]->count,
                        //          MPI_CHAR,
                        //          i,
                        //          PassMessage,
                        //          MPI_COMM_WORLD,
                        //          MPI_STATUS_IGNORE);
                        // recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);

                        MPI_Get_count(
                            &recv_status, MPI_CHAR, &gim_recv_buffer[partition_id][i][s_i]->count);
                        MPI_Recv(gim_recv_buffer[partition_id][i][s_i]->data,
                                 gim_recv_buffer[partition_id][i][s_i]->count,
                                 MPI_CHAR,
                                 i,
                                 PassMessage,
                                 MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        gim_recv_buffer[partition_id][i][s_i]->count /= sizeof(MsgUnit<M>);


                        // gim_comm->GIM_Probe(i, 0);
                        // gim_comm->GIM_Get_count(i, 0,
                        // gim_recv_buffer[partition_id][i][s_i]->count);
                        // gim_comm->GIM_Recv(gim_recv_buffer[partition_id][i][s_i]->count, i, 0);
                        // gim_recv_buffer[partition_id][i][s_i]->count /= sizeof(MsgUnit<M>);
                    }
                    recv_queue[recv_queue_size] = i;
                    recv_queue_mutex.lock();
                    recv_queue_size += 1;
                    recv_queue_mutex.unlock();
                }
            });
#endif

            // 现在数据都到每个host的recv_buffer里了
            std::thread check_thread;
            // check_thread = std::thread(
            //     check_blocking, std::ref(recv_queue_mutex), std::ref(recv_queue_size), 0);
            for (int step = 0; step < partitions; step++) {   // 遍历所有分区，这里是串行的
#ifdef SPARSE_MODE_UNIDIRECTIONAL
                while (true) {
                    // 最先处理自己的消息
                    expected_partition = (expected_partition - 1 + partitions) % partitions;

                    bool expected = true;

                    // 只看第一个socket
                    if (completion_tags[partition_id][expected_partition][0]
                            ->compare_exchange_strong(expected, false))
                        break;
                }

                int i = expected_partition;
#else

                // while (true) {
                //     recv_queue_mutex.lock();
                //     bool condition =
                //         (recv_queue_size <= step);   // 当前分区接受完成才继续后面的计算
                //     recv_queue_mutex.unlock();
                //     if (!condition) break;
                //     __asm volatile("pause" ::: "memory");
                // }
                if(check_thread.joinable())check_thread.join();
                if (step < partitions - 1) {
                    check_thread = std::thread(check_blocking,
                                               std::ref(recv_queue_mutex),
                                               std::ref(recv_queue_size),
                                               step + 1);
                }
                int i = recv_queue[step];
#endif

                global_current_send_part_id[partition_id] = i;
                // MessageBuffer** used_buffer;
                GIMMessageBuffer** used_buffer;
                if (i == partition_id) {
                    // used_buffer = send_buffer[i];   // MessageBuffer**
                    used_buffer = gim_send_buffer[partition_id][i];
                } else {
                    // used_buffer = recv_buffer[i];
                    used_buffer = gim_recv_buffer[partition_id][i];
                }
                for (int s_i = 0; s_i < sockets; s_i++) {   // 遍历所有socket
                    // 如果socket的消息还没有收到,等待，i==partition时已经准备好了
                    while (
                        i != partition_id && s_i &&
                        !completion_tags[partition_id][i][s_i]->load(std::memory_order_relaxed)) {
                        __asm volatile("pause" ::: "memory");
                    }

                    MsgUnit<M>* buffer = (MsgUnit<M>*)used_buffer[s_i]
                                             ->data;   // 将(MessageBuffer*)->data转为(MsgUnit<M>
                                                       //  s_i表示第s_i个socket的buffer
#ifdef SPARSE_MODE_UNIDIRECTIONAL
                    size_t buffer_size =
                        (i == partition_id)
                            ? send_count[partition_id][partition_id][s_i]
                            : length_array[partition_id][i]
                                          [s_i];   // send_buffer[i][s_i]->count 第i个host
                                                   // s_i个socket的buffersize
#else
                    size_t buffer_size =
                        (i == partition_id)
                            ? send_count[partition_id][partition_id][s_i]
                            : used_buffer[s_i]->count;   // send_buffer[i][s_i]->count 第i个host
                                                         // s_i个socket的buffersize
#endif
                    for (int t_i = 0; t_i < threads; t_i++) {   // 遍历所有线程
                        // 确定每个线程负责buffer的哪个部分,每个socket内的线程分工，不同socket的线程可能处理同样的任务
                        //  int s_i = get_socket_id(t_i);
                        int s_j = get_socket_offset(t_i);
                        VertexId partition_size = buffer_size;
                        thread_state[t_i]->curr =
                            partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
                        thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk *
                                                 basic_chunk * (s_j + 1);
                        if (s_j == threads_per_socket - 1) {
                            thread_state[t_i]->end = buffer_size;
                        }
                        thread_state[t_i]->status = WORKING;
                    }
#pragma omp parallel reduction(+ : reducer)
                    {
                        R local_reducer = 0;
                        int thread_id = omp_get_thread_num();
                        int s_i = get_socket_id(thread_id);
                        // 执行sparse_slot
                        while (true) {
                            // 线程开始从自己负责的部分开始获取任务
                            VertexId b_i =
                                __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
                            if (b_i >= thread_state[thread_id]->end) break;
                            VertexId begin_b_i = b_i;
                            VertexId end_b_i = b_i + basic_chunk;
                            if (end_b_i > thread_state[thread_id]->end) {
                                end_b_i = thread_state[thread_id]->end;
                            }
                            // 获取到basic_chunk个任务后，如果buffer里面点在线程所属numa(二级分区)内，就需要执行sparse_slot
                            for (b_i = begin_b_i; b_i < end_b_i; b_i++) {
                                VertexId v_i = buffer[b_i].vertex;
                                M msg_data = buffer[b_i].msg_data;
                                if (outgoing_adj_bitmap[s_i]->get_bit(
                                        v_i)) {   // 如果v_i在s_i socket内有出边
                                    local_reducer += sparse_slot(
                                        v_i,
                                        msg_data,
                                        VertexAdjList<EdgeData>(   // 遍历v_i在s_i
                                                                   //  socket内的出边的dst
                                            outgoing_adj_list[s_i] +
                                                outgoing_adj_index[s_i]
                                                                  [v_i],   // s_i内v_i的出边开始
                                            outgoing_adj_list[s_i] +
                                                outgoing_adj_index[s_i][v_i + 1]),
                                        -1);   // s_i内v_i的出边结束
                                }
                            }
                        }
                        // 工作窃取
                        thread_state[thread_id]->status = STEALING;
                        for (int t_offset = 1; t_offset < threads; t_offset++) {
                            int t_i = (thread_id + t_offset) % threads;
                            if (thread_state[t_i]->status == STEALING) continue;
                            while (true) {
                                VertexId b_i =
                                    __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                                if (b_i >= thread_state[t_i]->end) break;
                                VertexId begin_b_i = b_i;
                                VertexId end_b_i = b_i + basic_chunk;
                                if (end_b_i > thread_state[t_i]->end) {
                                    end_b_i = thread_state[t_i]->end;
                                }
                                int s_i = get_socket_id(t_i);
                                for (b_i = begin_b_i; b_i < end_b_i; b_i++) {
                                    VertexId v_i = buffer[b_i].vertex;
                                    M msg_data = buffer[b_i].msg_data;
                                    if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                                        local_reducer += sparse_slot(
                                            v_i,
                                            msg_data,
                                            VertexAdjList<
                                                EdgeData>(   // 这里只是两个指针，具体数据是存放邻居的数组
                                                outgoing_adj_list[s_i] +
                                                    outgoing_adj_index[s_i][v_i],
                                                outgoing_adj_list[s_i] +
                                                    outgoing_adj_index[s_i][v_i + 1]),
                                            -1);
                                    }
                                }
                            }
                        }
                        reducer += local_reducer;
                    }
                    while (stealingss[partition_id] != 0) {
                        // printf("stealings:%d\n", stealingss[partition_id]);
                        __asm volatile("pause" ::: "memory");
                    }
                }
#ifdef GLOBAL_STEALING_SPRASE
                for (int step = 1; step < partitions; step++) {
                    int i = (partition_id - step + partitions) % partitions;
                    // 怎么判断这个节点需不需要工作窃取
                    if (!waiting) break;
                    __sync_fetch_and_add(&stealingss[i], 1);
                    //  stealings[i]++;
#    pragma omp parallel reduction(+ : reducer)
                    {
                        R local_reducer = 0;
                        int thread_id = omp_get_thread_num();

                        for (int t_offset = 0; t_offset < threads; t_offset++) {
                            if (!waiting) break;
                            int t_i = (thread_id + t_offset) % threads;
                            if (gim_thread_state[i][t_i]->status == STEALING) continue;
                            while (waiting) {
                                VertexId b_i = __sync_fetch_and_add(&gim_thread_state[i][t_i]->curr,
                                                                    basic_chunk);
                                if (b_i >= gim_thread_state[i][t_i]->end) break;
                                VertexId begin_b_i = b_i;
                                VertexId end_b_i = b_i + basic_chunk;
                                if (end_b_i > gim_thread_state[i][t_i]->end) {
                                    end_b_i = gim_thread_state[i][t_i]->end;
                                }
                                int s_i = get_socket_id(t_i);
                                for (b_i = begin_b_i; b_i < end_b_i; b_i++) {
                                    // VertexId v_i = buffer[b_i].vertex;
                                    // M msg_data = buffer[b_i].msg_data;
                                    MsgUnit<M>* buffer;
                                    if (global_current_send_part_id[i] == i) {
                                        buffer = (MsgUnit<M>*)gim_send_buffer[i][i][0]->data;
                                    } else {
                                        buffer =
                                            (MsgUnit<M>*)
                                                gim_recv_buffer[i][global_current_send_part_id[i]]
                                                               [0]
                                                                   ->data;
                                    }
                                    VertexId v_i = buffer[b_i].vertex;
                                    M msg_data = buffer[b_i].msg_data;
                                    if (gim_outgoing_adj_bitmap[i][s_i]->get_bit(v_i)) {
                                        local_reducer += sparse_slot(
                                            v_i,
                                            msg_data,
                                            VertexAdjList<EdgeData>(

                                                gim_outgoing_adj_list[i][s_i] +
                                                    gim_outgoing_adj_index[i][s_i][v_i],
                                                gim_outgoing_adj_list[i][s_i] +
                                                    gim_outgoing_adj_index[i][s_i][v_i + 1]),
                                            i);
                                    }
                                }
                            }
                        }
                        reducer += local_reducer;
                    }

                    // stealings[i]--;
                    __sync_fetch_and_add(&stealingss[i], -1);
                }
#endif
            }
#ifdef SPARSE_MODE_UNIDIRECTIONAL
            comm_thread.join();
#endif
            // 全局工作窃取
            // TODO: fix socket
            // #ifdef GLOBAL_STEALING_SPRASE
            //             for (int step = 1; step < partitions; step++) {
            //                 int i = (partition_id - step + partitions) % partitions;
            //                 // 怎么判断这个节点需不需要工作窃取
            //                 __sync_fetch_and_add(&stealingss[i], 1);
            //                 //  stealings[i]++;
            // #    pragma omp parallel reduction(+ : reducer)
            //                 {
            //                     R local_reducer = 0;
            //                     int thread_id = omp_get_thread_num();

            //                     for (int t_offset = 0; t_offset < threads; t_offset++) {
            //                         int t_i = (thread_id + t_offset) % threads;
            //                         if (gim_thread_state[i][t_i]->status == STEALING) continue;
            //                         while (true) {
            //                             VertexId b_i =
            //                                 __sync_fetch_and_add(&gim_thread_state[i][t_i]->curr,
            //                                 basic_chunk);
            //                             if (b_i >= gim_thread_state[i][t_i]->end) break;
            //                             VertexId begin_b_i = b_i;
            //                             VertexId end_b_i = b_i + basic_chunk;
            //                             if (end_b_i > gim_thread_state[i][t_i]->end) {
            //                                 end_b_i = gim_thread_state[i][t_i]->end;
            //                             }
            //                             int s_i = get_socket_id(t_i);
            //                             for (b_i = begin_b_i; b_i < end_b_i; b_i++) {
            //                                 // VertexId v_i = buffer[b_i].vertex;
            //                                 // M msg_data = buffer[b_i].msg_data;
            //                                 MsgUnit<M>* buffer;
            //                                 if (global_current_send_part_id[i] == i) {
            //                                     buffer =
            //                                     (MsgUnit<M>*)gim_send_buffer[i][i][0]->data;
            //                                 } else {
            //                                     buffer =
            //                                         (MsgUnit<M>*)
            //                                             gim_recv_buffer[i][global_current_send_part_id[i]][0]
            //                                                 ->data;
            //                                 }
            //                                 VertexId v_i = buffer[b_i].vertex;
            //                                 M msg_data = buffer[b_i].msg_data;
            //                                 if (gim_outgoing_adj_bitmap[i][s_i]->get_bit(v_i)) {
            //                                     local_reducer += sparse_slot(
            //                                         v_i,
            //                                         msg_data,
            //                                         VertexAdjList<
            //                                             EdgeData>(   //
            //                                             这里只是两个指针，具体数据是存放邻居的数组
            //                                             gim_outgoing_adj_list[i][s_i] +
            //                                                 gim_outgoing_adj_index[i][s_i][v_i],
            //                                             gim_outgoing_adj_list[i][s_i] +
            //                                                 gim_outgoing_adj_index[i][s_i][v_i +
            //                                                 1]),
            //                                         i);
            //                                 }
            //                             }
            //                         }
            //                     }
            //                     reducer += local_reducer;
            //                 }

            //                 // stealings[i]--;
            //                 __sync_fetch_and_add(&stealingss[i], -1);
            //             }
            // #endif
#ifndef SPARSE_MODE_UNIDIRECTIONAL
            send_thread.join();
            recv_thread.join();
#endif
            process_edge_time[1] = MPI_Wtime() + stream_time - process_edge_time[0];
            delete[] recv_queue;
        } else {
            // dense selective bitmap
            if (dense_selective != nullptr && partitions > 1) {   // 根本没被用到
                double sync_time = 0;
                sync_time -= get_time();
                std::thread send_thread([&]() {
                    for (int step = 1; step < partitions; step++) {
                        int recipient_id = (partition_id + step) % partitions;
                        MPI_Send(
                            dense_selective->data + WORD_OFFSET(partition_offset[partition_id]),
                            owned_vertices / 64,
                            MPI_UNSIGNED_LONG,
                            recipient_id,
                            PassMessage,
                            MPI_COMM_WORLD);
                    }
                });
                std::thread recv_thread([&]() {
                    for (int step = 1; step < partitions; step++) {
                        int sender_id = (partition_id - step + partitions) % partitions;
                        MPI_Recv(
                            dense_selective->data + WORD_OFFSET(partition_offset[sender_id]),
                            (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64,
                            MPI_UNSIGNED_LONG,
                            sender_id,
                            PassMessage,
                            MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE);
                    }
                });
                send_thread.join();
                recv_thread.join();
                MPI_Barrier(MPI_COMM_WORLD);
                sync_time += get_time();
#ifdef PRINT_DEBUG_MESSAGES
                if (partition_id == 0) {
                    printf("sync_time = %lf\n", sync_time);
                }
#endif
            }
#ifdef PRINT_DEBUG_MESSAGES
            if (partition_id == 0) {
                printf("dense mode\n");
            }
#endif
            /*
            稠密模式下不同的一点是提前创建了一个发送线程和一个接受线程，接受线程创建partitions-1个接受子线程。
            发送线程顺序发送partitions-1个数据，等待dense_signal完成，完成一个partitions就发送一个
            接受子线程一直MPI_RECV阻塞接受，然后接受线程顺序等待接受partitions-1个数据
            每有一个partitions的数据被接受，就启动一个partitions的dense_slot
            */
            int* send_queue = new int[partitions];
            int* recv_queue = new int[partitions];
            volatile int send_queue_size = 0;
            volatile int recv_queue_size = 0;
            std::mutex send_queue_mutex;
            std::mutex recv_queue_mutex;

#ifdef DENSE_MODE_UNIDIRECTIONAL
            std::thread comm_thread([&] {
                for (int step = 0; step < partitions - 1; step++) {
                    while (true) {
                        send_queue_mutex.lock();
                        bool condition = (send_queue_size <= step);
                        send_queue_mutex.unlock();
                        if (!condition) break;
                        __asm volatile("pause" ::: "memory");
                    }
                    int i = send_queue[step];
                    for (int s_i = 0; s_i < sockets; s_i++) {
                        memcpy(gim_recv_buffer[i][partition_id][s_i]->data,
                               gim_send_buffer[partition_id][i][s_i]->data,
                               sizeof(MsgUnit<M>) * send_count[partition_id][i][s_i]);

                        length_array[i][partition_id][s_i] = send_count[partition_id][i][s_i];
                        completion_tags[i][partition_id][s_i]->store(true,
                                                                     std::memory_order_release);
                    }
                }
            });
#else
            std::thread send_thread([&]() {
                for (int step = 0; step < partitions; step++) {
                    if (step == partitions - 1) {
                        break;
                    }
                    while (true) {
                        send_queue_mutex.lock();
                        bool condition =
                            (send_queue_size <=
                             step);   // 当前分区发送出去才继续后面的，每一次发送完成才发送下一次
                        send_queue_mutex.unlock();
                        if (!condition) break;
                        __asm volatile("pause" ::: "memory");
                    }
                    int i = send_queue[step];
                    for (int s_i = 0; s_i < sockets; s_i++) {
                        // MPI_Send(send_buffer[i][s_i]->data,
                        //          sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count,
                        //          MPI_CHAR,
                        //          i,
                        //          PassMessage,
                        //          MPI_COMM_WORLD);
                        MPI_Send(gim_send_buffer[partition_id][i][s_i]->data,
                                 sizeof(MsgUnit<M>) * send_count[partition_id][i][s_i],
                                 MPI_CHAR,
                                 i,
                                 PassMessage,
                                 MPI_COMM_WORLD);
                        // gim_comm->GIM_Send(
                        //     gim_send_buffer[partition_id][i][s_i]->data,
                        //     sizeof(MsgUnit<M>) * send_count[partition_id][i][s_i],
                        //     i,
                        //     0,
                        //     gim_recv_buffer[i][partition_id][s_i]->data);
                    }
                }
            });
            std::thread recv_thread([&]() {
                std::vector<std::thread> threads;
                for (int step = 1; step < partitions; step++) {
                    int i = (partition_id - step + partitions) %
                            partitions;   // 确保i进程是除了自己以外的所有进程
                    threads
                        .emplace_back(   // partitions-1个线程，每个线程的任务是接受数据，共接受partitions-1
                            [&](int i) {   // i是分区
                                for (int s_i = 0; s_i < sockets; s_i++) {
                                    MPI_Status recv_status;
                                    MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
                                    // MPI_Get_count(
                                    //     &recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
                                    // MPI_Recv(recv_buffer[i][s_i]->data,
                                    //          recv_buffer[i][s_i]->count,
                                    //          MPI_CHAR,
                                    //          i,
                                    //          PassMessage,
                                    //          MPI_COMM_WORLD,
                                    //          MPI_STATUS_IGNORE);
                                    // recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);

                                    MPI_Get_count(&recv_status,
                                                  MPI_CHAR,
                                                  &gim_recv_buffer[partition_id][i][s_i]->count);
                                    MPI_Recv(gim_recv_buffer[partition_id][i][s_i]->data,
                                             gim_recv_buffer[partition_id][i][s_i]->count,
                                             MPI_CHAR,
                                             i,
                                             PassMessage,
                                             MPI_COMM_WORLD,
                                             MPI_STATUS_IGNORE);
                                    gim_recv_buffer[partition_id][i][s_i]->count /=
                                        sizeof(MsgUnit<M>);


                                    // gim_comm->GIM_Probe(i, 0);
                                    // gim_comm->GIM_Get_count(
                                    //     i, 0, gim_recv_buffer[partition_id][i][s_i]->count);
                                    // gim_comm->GIM_Recv(
                                    //     gim_recv_buffer[partition_id][i][s_i]->count, i, 0);
                                    // gim_recv_buffer[partition_id][i][s_i]->count /=
                                    //     sizeof(MsgUnit<M>);
                                }
                            },
                            i);
                }
                for (int step = 1; step < partitions; step++) {
                    int i = (partition_id - step + partitions) % partitions;
                    threads[step - 1].join();
                    recv_queue[recv_queue_size] = i;
                    recv_queue_mutex.lock();
                    recv_queue_size += 1;
                    recv_queue_mutex.unlock();
                }
                recv_queue[recv_queue_size] = partition_id;
                recv_queue_mutex.lock();
                recv_queue_size += 1;
                recv_queue_mutex.unlock();
            });
#endif
            current_send_part_id = partition_id;
            for (int step = 0; step < partitions; step++) {
                current_send_part_id = (current_send_part_id + 1) % partitions;
                global_current_send_part_id[partition_id] =
                    current_send_part_id;   // 存疑，要不要锁
                int i = current_send_part_id;
                for (int t_i = 0; t_i < threads; t_i++) {
                    *thread_state[t_i] = tuned_chunks_dense[i][t_i];
                }
                // 每个点对邻居执行dense_signal
#pragma omp parallel
                {
                    int thread_id = omp_get_thread_num();
                    int s_i = get_socket_id(thread_id);
                    VertexId final_p_v_i = thread_state[thread_id]->end;
                    while (true) {
                        // 线程开始从自己负责的部分开始获取任务
                        VertexId begin_p_v_i =
                            __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
                        if (begin_p_v_i >= final_p_v_i) break;
                        VertexId end_p_v_i = begin_p_v_i + basic_chunk;
                        if (end_p_v_i > final_p_v_i) {
                            end_p_v_i = final_p_v_i;
                        }
                        for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++) {
                            VertexId v_i =
                                compressed_incoming_adj_index[s_i][p_v_i]
                                    .vertex;   // 对于s_i socket中的v_i，对所有邻居执行dense_signal
                            dense_signal(
                                v_i,
                                VertexAdjList<EdgeData>(
                                    incoming_adj_list[s_i] +
                                        compressed_incoming_adj_index[s_i][p_v_i].index,
                                    incoming_adj_list[s_i] +
                                        compressed_incoming_adj_index[s_i][p_v_i + 1].index),
                                -1);
                        }
                    }
                    // 线程窃取
                    thread_state[thread_id]->status = STEALING;
                    for (int t_offset = 1; t_offset < threads; t_offset++) {
                        int t_i = (thread_id + t_offset) % threads;
                        int s_i = get_socket_id(t_i);
                        while (thread_state[t_i]->status != STEALING) {
                            VertexId begin_p_v_i =
                                __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                            if (begin_p_v_i >= thread_state[t_i]->end) break;
                            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
                            if (end_p_v_i > thread_state[t_i]->end) {
                                end_p_v_i = thread_state[t_i]->end;
                            }
                            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++) {
                                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                                dense_signal(
                                    v_i,
                                    VertexAdjList<EdgeData>(
                                        incoming_adj_list[s_i] +
                                            compressed_incoming_adj_index[s_i][p_v_i]
                                                .index,   // s_i出边开始
                                        incoming_adj_list[s_i] +
                                            compressed_incoming_adj_index[s_i][p_v_i +
                                                                               1]   // s_i出边结束
                                                .index),
                                    -1);
                            }
                        }
                    }
                }

                // 全部执行完后再flush
#pragma omp parallel for
                for (int t_i = 0; t_i < threads; t_i++) {
                    flush_local_send_buffer<M>(t_i);
                }
                // 确保其他节点窃取的任务完成
                while (stealingss[partition_id] != 0) {
                    // printf("stealings:%d ,send_part:%d.%d\n",
                    //        stealingss[partition_id],
                    //        current_send_part_id,
                    //        global_current_send_part_id[partition_id].load());
                    __asm volatile("pause" ::: "memory");
                }
                // 这里是启动发送线程的关键，处理完一个分区就发送一个分区，其实和稀疏模式一样
                if (i != partition_id) {
                    send_queue[send_queue_size] = i;
                    send_queue_mutex.lock();
                    send_queue_size += 1;
                    send_queue_mutex.unlock();
                }
            }
            // 全局工作窃取
#ifdef GLOBAL_STEALING_DENSE
            for (int step = 1; step < partitions; step++) {
                int i = (partition_id - step + partitions) % partitions;
                // 怎么判断这个节点需不需要工作窃取
                if (global_current_send_part_id[i] == i) {
                    continue;
                }
                __sync_fetch_and_add(&stealingss[i], 1);
                //  stealings[i]++;
#    pragma omp parallel
                {
                    int thread_id = omp_get_thread_num();

                    for (int t_offset = 0; t_offset < threads; t_offset++) {
                        int t_i = (thread_id + t_offset) % threads;
                        int s_i = get_socket_id(t_i);
                        while (gim_thread_state[i][t_i]->status != STEALING) {
                            VertexId begin_p_v_i =
                                __sync_fetch_and_add(&gim_thread_state[i][t_i]->curr, basic_chunk);
                            if (begin_p_v_i >= gim_thread_state[i][t_i]->end) break;
                            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
                            if (end_p_v_i > gim_thread_state[i][t_i]->end) {
                                end_p_v_i = gim_thread_state[i][t_i]->end;
                            }
                            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++) {
                                VertexId v_i =
                                    gim_compressed_incoming_adj_index[i][s_i][p_v_i].vertex;
                                dense_signal(
                                    v_i,
                                    VertexAdjList<EdgeData>(
                                        gim_incoming_adj_list[i][s_i] +
                                            gim_compressed_incoming_adj_index[i][s_i][p_v_i]
                                                .index,   // s_i出边开始
                                        gim_incoming_adj_list[i][s_i] +
                                            gim_compressed_incoming_adj_index[i][s_i][p_v_i + 1]
                                                .index),
                                    i);
                            }
                        }
                    }
                }
#    pragma omp parallel for
                for (int t_i = 0; t_i < threads; t_i++) {
                    flush_local_send_buffer_to_other<M>(t_i, i);
                }
                // stealings[i]--;
                __sync_fetch_and_add(&stealingss[i], -1);
            }
#endif

            process_edge_time[2] = MPI_Wtime() + stream_time;

#ifdef DENSE_MODE_UNIDIRECTIONAL
            int expected_partition = partition_id + 1;
            // 标记有接收自己的消息
            completion_tags[partition_id][partition_id][0]->store(true, std::memory_order_relaxed);
#endif
            // dense_slot
            for (int step = 0; step < partitions; step++) {
#ifdef DENSE_MODE_UNIDIRECTIONAL
                while (true) {
                    // 最先处理自己的消息
                    expected_partition = (expected_partition - 1 + partitions) % partitions;

                    bool expected = true;

                    // 只看第一个socket
                    if (completion_tags[partition_id][expected_partition][0]
                            ->compare_exchange_strong(expected, false))
                        break;
                }

                int i = expected_partition;
#else
                while (true) {
                    recv_queue_mutex.lock();
                    bool condition = (recv_queue_size <= step);
                    recv_queue_mutex.unlock();
                    if (!condition) break;
                    __asm volatile("pause" ::: "memory");
                }   // 接受完一个就启动计算
                int i = recv_queue[step];
#endif
                // MessageBuffer** used_buffer;
                GIMMessageBuffer** used_buffer;
                if (i == partition_id) {
                    // used_buffer = send_buffer[i];
                    used_buffer = gim_send_buffer[partition_id][i];
                } else {
                    // used_buffer = recv_buffer[i];
                    used_buffer = gim_recv_buffer[partition_id][i];
                }
                // 确定每个线程负责buffer的哪个部分
                for (int t_i = 0; t_i < threads; t_i++) {
                    int s_i = get_socket_id(t_i);
                    while (
                        i != partition_id && s_i &&
                        !completion_tags[partition_id][i][s_i]->load(std::memory_order_relaxed)) {
                        __asm volatile("pause" ::: "memory");
                    }
                    int s_j = get_socket_offset(t_i);
#ifdef DENSE_MODE_UNIDIRECTIONAL
                    size_t buffer_size = (i == partition_id)
                                             ? send_count[partition_id][partition_id][s_i]
                                             : length_array[partition_id][i][s_i];
#else
                    size_t buffer_size = (i == partition_id)
                                             ? send_count[partition_id][i][s_i]
                                             : used_buffer[s_i]->count;
#endif
                    VertexId partition_size = buffer_size;
                    thread_state[t_i]->curr =
                        partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
                    thread_state[t_i]->end =
                        partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
                    if (s_j == threads_per_socket - 1) {
                        thread_state[t_i]->end = buffer_size;
                    }
                    thread_state[t_i]->status = WORKING;
                }
#pragma omp parallel reduction(+ : reducer)
                {
                    R local_reducer = 0;
                    int thread_id = omp_get_thread_num();
                    int s_i = get_socket_id(thread_id);
                    MsgUnit<M>* buffer = (MsgUnit<M>*)used_buffer[s_i]->data;
                    while (true) {
                        // 线程开始从自己负责的部分开始获取任务
                        VertexId b_i =
                            __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
                        if (b_i >= thread_state[thread_id]->end) break;
                        VertexId begin_b_i = b_i;
                        VertexId end_b_i = b_i + basic_chunk;
                        if (end_b_i > thread_state[thread_id]->end) {
                            end_b_i = thread_state[thread_id]->end;
                        }
                        for (b_i = begin_b_i; b_i < end_b_i; b_i++) {
                            VertexId v_i = buffer[b_i].vertex;
                            M msg_data = buffer[b_i].msg_data;
                            local_reducer += dense_slot(v_i, msg_data);   // 对每个vi执行dense_slot
                        }
                    }
                    thread_state[thread_id]->status = STEALING;
                    reducer += local_reducer;
                }
            }
#ifdef DENSE_MODE_UNIDIRECTIONAL
            comm_thread.join();
#else
            send_thread.join();
            recv_thread.join();
#endif
            delete[] send_queue;
            delete[] recv_queue;
        }
        process_edge_time[3] = MPI_Wtime() - process_edge_time[2] + stream_time;

        stream_time += MPI_Wtime();
        total_process_time += stream_time;
        // printf("partition_id: %d,process_edges took %lf (s)\n",partition_id, stream_time);
        // printf("partition_id: %d,spare slot: %lf (s) ,spare signal :%lf (s),dense signal: %lf
        // (s),dense slot: %lf (s)\n",partition_id,
        // process_edge_time[0],process_edge_time[1],process_edge_time[2],process_edge_time[3]);

        R global_reducer;
        MPI_Datatype dt = get_mpi_data_type<R>();
        MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 0) {
            printf("process_edges took %lf (s)\n", stream_time);
        }
#endif
        if (is_first && partition_id == 0) {
            // printf_data_info();
            is_first = false;
        }
        return global_reducer;
    }
};

#endif
