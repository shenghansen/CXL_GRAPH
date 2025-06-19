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

#include <cstdint>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "core/gim_graph.hpp"
#include "type.hpp"

double exec_time = 0;
std::vector<double> times;
void compute(Graph<Empty>* graph, VertexId root) {
    exec_time = 0;
    exec_time -= get_time();

    // VertexId* parent = graph->alloc_vertex_array<VertexId>();
    // VertexSubset* visited = graph->alloc_vertex_subset();
    // VertexSubset* active_in = graph->alloc_vertex_subset();
    // VertexSubset* active_out = graph->alloc_vertex_subset();
    MPI_Barrier(MPI_COMM_WORLD);
    VertexId** global_parent = graph->alloc_global_vertex_array<VertexId>();
    VertexSubset** global_visited = graph->alloc_global_vertex_subset();
    VertexSubset** global_active_in = graph->alloc_global_vertex_subset();
    VertexSubset** global_active_out = graph->alloc_global_vertex_subset();

    MPI_Barrier(MPI_COMM_WORLD);
    VertexSubset* visited = global_visited[graph->partition_id];
    VertexSubset* active_in = global_active_in[graph->partition_id];
    VertexSubset* active_out = global_active_out[graph->partition_id];
    VertexId* parent = global_parent[graph->partition_id];
    MPI_Barrier(MPI_COMM_WORLD);


    visited->clear();
    visited->set_bit(root);
    active_in->clear();
    active_in->set_bit(root);
    graph->fill_vertex_array(parent, graph->vertices);
    parent[root] = root;
    MPI_Barrier(MPI_COMM_WORLD);
    VertexId active_vertices = 1;

    for (int i_i = 0; active_vertices > 0; i_i++) {
#ifdef SHOW_RESULT
        if (graph->partition_id == 0) {
            printf("active(%d)>=%u\n", i_i, active_vertices);
            //   graph->print_process_data();
        }
// printf("partition_id:%d, sequence:\n",graph->partition_id);
// graph->print_get_sequence();
#endif
        active_out->clear();
        active_vertices = graph->process_edges<VertexId, VertexId>(
            [&](VertexId src) { graph->emit(src, src); },
#ifdef COMPRESS
            [&](VertexId src,
                VertexId msg,
                uint8_t* compressed_list,
                int degree,
                int partition_id) {
                if (partition_id == -1) {
                    VertexId activated = 0;
                    // printf("partition: %d, src: %d, degree: %d\n", graph->partition_id, src, degree);
                    decode<Empty>(
                        [&](VertexId src, VertexId dst, int weight, int edgeRead) -> bool {
                            if (parent[dst] == graph->vertices &&
                                cas(&parent[dst], graph->vertices, src)) {
                                active_out->set_bit(dst);
                                activated += 1;
                            }
                                // printf("partition: %d,degree:%d,src:%d, dst: %d,eageread: %d,activate:%d\n",graph->partition_id, degree, src, dst,edgeRead, activated);
                            return true;
                        },
                        compressed_list,
                        src,
                        degree);
                        // printf("partition: %d, decode finished\n", graph->partition_id);
                    return activated;
                } else {
                    VertexId activated = 0;
                    decode<Empty>(
                        [&](VertexId src, VertexId dst, int weight, int edgeRead) -> bool {
                            if (global_parent[partition_id][dst] == graph->vertices &&
                                cas(&global_parent[partition_id][dst], graph->vertices, src)) {
                                global_active_out[partition_id]->set_bit(dst);
                                activated += 1;
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
                    if (visited->get_bit(dst)) return;
                    decode<Empty>(
                        [&](VertexId decode_dst, VertexId src, int weight, int edgeRead) -> bool {
                            if (active_in->get_bit(src)) {
                                graph->emit(decode_dst, src);
                                return false;
                            }
                            // if (graph->partition_id == 0) {
                                // printf("partition: %d,degree:%d,dst:%d, src: %d,eageread: %d\n",graph->partition_id, degree, dst, src,edgeRead);
                            // }
                            return true;
                        },
                        compressed_list,
                        dst,
                        degree);
                    // printf("decode finished\n");
                } else {
                    if (global_visited[partition_id]->get_bit(dst)) return;
                    decode<Empty>(
                        [&](VertexId dst, VertexId src, int weight, int edgeRead) -> bool {
                            if (global_active_in[partition_id]->get_bit(src)) {
                                graph->emit_other(dst, src, partition_id);
                                
                                return false;
                            }
                            return true;
                        },
                        compressed_list,
                        dst,
                        degree);
                }
            },
#else
            [&](VertexId src, VertexId msg, VertexAdjList<Empty> outgoing_adj, int partition_id) {
                if (partition_id == -1) {
                    VertexId activated = 0;
                    // printf("partition: %d,src: %d, degree:%d\n", graph->partition_id,src, outgoing_adj.end - outgoing_adj.begin);
                    for (AdjUnit<Empty>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        VertexId dst = ptr->neighbour;
                        CXL_PREFETCH
                        if (parent[dst] == graph->vertices &&
                            cas(&parent[dst], graph->vertices, src)) {
                            active_out->set_bit(dst);
                            activated += 1;
                        }
                    }
                    return activated;
                } else {
                    VertexId activated = 0;
                    for (AdjUnit<Empty>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        VertexId dst = ptr->neighbour;
                        CXL_PREFETCH
                        if (global_parent[partition_id][dst] == graph->vertices &&
                            cas(&global_parent[partition_id][dst], graph->vertices, src)) {
                            global_active_out[partition_id]->set_bit(dst);
                            activated += 1;
                        }
                    }
                    return activated;
                }
            },
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj, int partition_id) {
                if (partition_id == -1) {
                    if (visited->get_bit(dst)) return;
#    ifdef OMP_SIMD
#    endif
                    for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        VertexId src = ptr->neighbour;
                        CXL_PREFETCH
                        if (active_in->get_bit(src)) {
                            graph->emit(dst, src);
                            break;
                        }
                    }
                } else {
                    if (global_visited[partition_id]->get_bit(dst)) return;
#    ifdef OMP_SIMD
#    endif
                    for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        VertexId src = ptr->neighbour;
                        CXL_PREFETCH
                        if (global_active_in[partition_id]->get_bit(src)) {
                            graph->emit_other(dst, src, partition_id);
                            break;
                        }
                    }
                }
            },
#endif
            [&](VertexId dst, VertexId msg) {
                if (cas(&parent[dst], graph->vertices, msg)) {
                    active_out->set_bit(dst);
                    return 1;
                }
                return 0;
            },
            active_in,
            visited);
#ifndef GLOBAL_STEALING_VERTICES
        active_vertices = graph->process_vertices<VertexId>(   // 用来标记点访问过了
            [&](VertexId vtx) {
                visited->set_bit(vtx);
                return 1;
            },
            active_out);
#else
        active_vertices = graph->process_vertices_global<VertexId>(   // 用来标记点访问过了
            [&](VertexId vtx, int partition_id) {
                if (partition_id == -1) {
                    visited->set_bit(vtx);
                    return 1;
                } else {
                    global_visited[partition_id]->set_bit(vtx);
                    return 1;
                }
            },
            global_active_out);
#endif
        std::swap(active_in, active_out);
        std::swap(global_active_in, global_active_out);
    }
    exec_time += get_time();
    times.push_back(exec_time);
    // if (graph->partition_id==0) {
    //   printf("exec_time=%lf(s)\n", exec_time);
    // }
    //   printf("partition: %d,exec_time=%lf(s)\n", graph->get_partition_id(), exec_time);
    graph->gather_vertex_array(parent, 0);
#ifdef SHOW_RESULT
    if (graph->partition_id == 0) {
        VertexId found_vertices = 0;
        for (VertexId v_i = 0; v_i < graph->vertices; v_i++) {
            if (parent[v_i] < graph->vertices) {
                found_vertices += 1;
            }
        }
        printf("found_vertices = %u\n", found_vertices);
    }
#endif
    graph->dealloc_vertex_array(parent);
    delete active_in;
    delete active_out;
    delete visited;
}

int main(int argc, char** argv) {
    MPI_Instance mpi(&argc, &argv);

    if (argc < 4) {
        printf("bfs [file] [vertices] [root]\n");
        exit(-1);
    }

    Graph<Empty>* graph;
    graph = new Graph<Empty>();
    VertexId root = std::atoi(argv[3]);
    VertexId vertices = std::atoi(argv[2]);
    std::string base_filename = argv[1];
    // bool loaded_from_preprocessed = graph->load_preprocessed_graph(base_filename);
    // if (!loaded_from_preprocessed) {
        // if (graph->get_partition_id() == 0) {
        //     printf("Loading graph from original file: %s\n", base_filename.c_str());
        // }
        // 根据图是否有向或对称选择加载函数
        graph->load_directed(base_filename, vertices);   // 或者 load_undirected_from_directed
        // 可选：在第一次加载后保存预处理文件
        // if (graph->get_partition_id() == 0) {
        //      printf("Saving preprocessed graph data...\n");
        // }
        // graph->save_preprocessed_graph(base_filename);
        //  if (graph->get_partition_id() == 0) {
        //      printf("Finished saving preprocessed graph data.\n");
        // }
    // }

    //   graph->load_directed(argv[1], std::atoi(argv[2]));
 MPI_Barrier(MPI_COMM_WORLD);
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
    //   delete graph;
    // _exit(0);
    return 0;
}
