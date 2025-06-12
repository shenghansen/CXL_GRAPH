/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

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

#include <stdio.h>
#include <stdlib.h>

#include "compress.hpp"
#include "core/gim_graph.hpp"
#include "mpi.h"

typedef int Weight;
double exec_time = 0;
std::vector<double> times;
void compute(Graph<Weight>* graph, VertexId root) {
    exec_time = 0;
    exec_time -= get_time();

    // Weight * distance = graph->alloc_vertex_array<Weight>();
    // VertexSubset * active_in = graph->alloc_vertex_subset();
    // VertexSubset * active_out = graph->alloc_vertex_subset();
    Weight** global_distance = graph->alloc_global_vertex_array<Weight>();
    VertexSubset** global_active_in = graph->alloc_global_vertex_subset();
    VertexSubset** global_active_out = graph->alloc_global_vertex_subset();
    Weight* distance = global_distance[graph->partition_id];
    VertexSubset* active_in = global_active_in[graph->partition_id];
    VertexSubset* active_out = global_active_out[graph->partition_id];
    MPI_Barrier(MPI_COMM_WORLD);
    active_in->clear();
    active_in->set_bit(root);
    graph->fill_vertex_array(distance, (Weight)1e9);
    distance[root] = (Weight)0;
    VertexId active_vertices = 1;

    for (int i_i = 0; active_vertices > 0; i_i++) {
#ifdef SHOW_RESULT
        if (graph->partition_id == 0) {
            printf("active(%d)>=%u\n", i_i, active_vertices);
            // graph->print_process_data();
        }
        // printf("partition_id:%d, sequence:\n",graph->partition_id);
        // graph->print_get_sequence();
#endif
        active_out->clear();
        active_vertices = graph->process_edges<VertexId, Weight>(
            [&](VertexId src) { graph->emit(src, distance[src]); },
#ifdef COMPRESS
            [&](VertexId src, Weight msg, uint8_t* compressed_list, int degree, int partition_id) {
                if (partition_id == -1) {
                    VertexId activated = 0;
                    decode<Weight>(
                        [&](VertexId src, VertexId dst, int weight, int edgeRead) -> bool {
                            Weight relax_dist = msg + weight;
                            if (relax_dist < distance[dst]) {
                                if (write_min(&distance[dst], relax_dist)) {
                                    active_out->set_bit(dst);
                                    activated += 1;
                                }
                            }
                            return true;
                        },
                        compressed_list,
                        src,
                        degree);
                    return activated;
                } else {
                    VertexId activated = 0;
                    decode<Weight>(
                        [&](VertexId src, VertexId dst, int weight, int edgeRead) -> bool {
                            Weight relax_dist = msg + weight;
                            if (relax_dist < global_distance[partition_id][dst]) {
                                if (write_min(&global_distance[partition_id][dst], relax_dist)) {
                                    global_active_out[partition_id]->set_bit(dst);
                                    activated += 1;
                                }
                            }
                            return true;
                        },
                        compressed_list,
                        src,
                        degree);
                    return activated;
                }
            },
            [&](VertexId dst, uint8_t* compressed_list, int degree, int partition_id) {
                if (partition_id == -1) {
                    Weight msg = 1e9;
                    decode<Weight>(
                        [&](VertexId dst, VertexId src, int weight, int edgeRead) -> bool {
                            Weight relax_dist = distance[src] + weight;
                            if (relax_dist < msg) {
                                msg = relax_dist;
                            }
                            return true;
                        },
                        compressed_list,
                        dst,
                        degree);
                    if (msg < 1e9) graph->emit(dst, msg);
                } else {
                    Weight msg = 1e9;
                    decode<Weight>(
                        [&](VertexId dst, VertexId src, int weight, int edgeRead) -> bool {
                            Weight relax_dist = global_distance[partition_id][src] + weight;
                            if (relax_dist < msg) {
                                msg = relax_dist;
                            }
                            return true;
                        },
                        compressed_list,
                        dst,
                        degree);
                    if (msg < 1e9) graph->emit_other(dst, msg, partition_id);
                }
            },
#else
            [&](VertexId src, Weight msg, VertexAdjList<Weight> outgoing_adj, int partition_id) {
                if (partition_id == -1) {
                    VertexId activated = 0;
                    for (AdjUnit<Weight>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end;
                         ptr++) {
                        VertexId dst = ptr->neighbour;
                        Weight relax_dist = msg + ptr->edge_data;
                        CXL_PREFETCH
                        if (relax_dist < distance[dst]) {
                            if (write_min(&distance[dst], relax_dist)) {
                                active_out->set_bit(dst);
                                activated += 1;
                            }
                        }
                    }
                    return activated;
                } else {
                    VertexId activated = 0;
                    for (AdjUnit<Weight>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end;
                         ptr++) {
                        VertexId dst = ptr->neighbour;
                        Weight relax_dist = msg + ptr->edge_data;
                        CXL_PREFETCH
                        if (relax_dist < global_distance[partition_id][dst]) {
                            if (write_min(&global_distance[partition_id][dst], relax_dist)) {
                                global_active_out[partition_id]->set_bit(dst);
                                activated += 1;
                            }
                        }
                    }
                    return activated;
                }
            },
            [&](VertexId dst, VertexAdjList<Weight> incoming_adj, int partition_id) {
                if (partition_id == -1) {
                    Weight msg = 1e9;
#    ifdef OMP_SIMD
#        pragma omp simd reduction(min : msg)
#    endif
                    for (AdjUnit<Weight>* ptr = incoming_adj.begin; ptr != incoming_adj.end;
                         ptr++) {
                        VertexId src = ptr->neighbour;
                        Weight relax_dist = distance[src] + ptr->edge_data;
                        CXL_PREFETCH
                        if (relax_dist < msg) {
                            msg = relax_dist;
                        }
                    }
                    if (msg < 1e9) graph->emit(dst, msg);
                } else {
                    Weight msg = 1e9;
#    ifdef OMP_SIMD
#        pragma omp simd reduction(min : msg)
#    endif
                    for (AdjUnit<Weight>* ptr = incoming_adj.begin; ptr != incoming_adj.end;
                         ptr++) {
                        VertexId src = ptr->neighbour;
                        CXL_PREFETCH
                        Weight relax_dist = global_distance[partition_id][src] + ptr->edge_data;
                        if (relax_dist < msg) {
                            msg = relax_dist;
                        }
                    }
                    if (msg < 1e9) graph->emit_other(dst, msg, partition_id);
                }
            },
#endif
            [&](VertexId dst, Weight msg) {
                if (msg < distance[dst]) {
                    write_min(&distance[dst], msg);
                    active_out->set_bit(dst);
                    return 1;
                }
                return 0;
            },
            active_in);
        std::swap(active_in, active_out);
        std::swap(global_active_in, global_active_out);
    }

    exec_time += get_time();
    times.push_back(exec_time);
    // if (graph->partition_id==0) {
    //   printf("exec_time=%lf(s)\n", exec_time);
    // }
    // printf("partition: %d,exec_time=%lf(s)\n", graph->get_partition_id(), exec_time);
    graph->gather_vertex_array(distance, 0);
#ifdef SHOW_RESULT
    if (graph->partition_id == 0) {
        VertexId max_v_i = root;
        for (VertexId v_i = 0; v_i < graph->vertices; v_i++) {
            if (distance[v_i] < 1e9 && distance[v_i] > distance[max_v_i]) {
                max_v_i = v_i;
            }
        }
        printf("distance[%u]=%f\n", max_v_i, distance[max_v_i]);
    }
#endif
    graph->dealloc_vertex_array(distance);
    delete active_in;
    delete active_out;
}

int main(int argc, char** argv) {
    MPI_Instance mpi(&argc, &argv);

    if (argc < 4) {
        printf("sssp [file] [vertices] [root]\n");
        exit(-1);
    }

    Graph<Weight>* graph;
    graph = new Graph<Weight>();
    VertexId vertices = std::atoi(argv[2]);
    std::string base_filename = argv[1];

    bool loaded_from_preprocessed = graph->load_preprocessed_graph(base_filename + ".with_data");
    // bool loaded_from_preprocessed = false;
    if (!loaded_from_preprocessed) {
        // if (graph->get_partition_id() == 0) {
        //     printf("Loading graph from original file: %s\n", base_filename.c_str());
        // }
        // 根据图是否有向或对称选择加载函数
        graph->load_directed(base_filename, vertices);   // 或者 load_undirected_from_directed

        // 可选：在第一次加载后保存预处理文件
        // if (graph->get_partition_id() == 0) {
        //     printf("Saving preprocessed graph data...\n");
        // }
        // graph->save_preprocessed_graph(base_filename);
        // if (graph->get_partition_id() == 0) {
        //     printf("Finished saving preprocessed graph data.\n");
        // }
    }
    // graph->load_directed(argv[1], std::atoi(argv[2]));
    VertexId root = std::atoi(argv[3]);
    for (size_t i = 0; i < EXEC_TIMES; i++) {
        compute(graph, root);
    }
    double average_time = 0;
    for (auto i : times) {
        average_time += i;
    }
    average_time /= EXEC_TIMES;

#if OUTPUT_LEVEL == 0
    if (graph->partition_id == 0) {
        printf("exec_time=%lf(s)\n", exec_time);
    }
    printf("partiton_id: %d, total_process_time  =%lf(s)\n",
           graph->get_partition_id(),
           graph->print_total_process_time() / EXEC_TIMES);
    printf("partiton_id: %d, average  =%lf(s)\n", graph->get_partition_id(), average_time);
#elif OUTPUT_LEVEL == 1
    printf("partiton_id: %d, average  =%lf(s)\n", graph->get_partition_id(), average_time);
#elif OUTPUT_LEVEL == 2
    double max_average_time = 0;
    MPI_Allreduce(&average_time, &max_average_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (graph->partition_id == 0) {
        printf("%lf\n", max_average_time);
    }
#elif OUTPUT_LEVEL == 3
    printf("partiton_id: %d, total_process_time  =%lf(s)\n",
           graph->get_partition_id(),
           graph->print_total_process_time() / EXEC_TIMES);
    double max_total_process_time = 0;
    double total_process_time = graph->print_total_process_time() / EXEC_TIMES;
    MPI_Allreduce(
        &total_process_time, &max_total_process_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (graph->partition_id == 0) {
        printf("%lf\n", max_total_process_time);
    }
#endif

    delete graph;
    _exit(0);
    return 0;
}
