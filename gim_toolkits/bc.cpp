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

#include "core/gim_graph.hpp"
#include "mpi.h"

#define COMPACT 0
double exec_time = 0;
std::vector<double> times;
void compute(Graph<Empty>* graph, VertexId root) {
    exec_time = 0;
    exec_time -= get_time();

    // double * num_paths = graph->alloc_vertex_array<double>();
    // double * dependencies = graph->alloc_vertex_array<double>();
    // VertexSubset * active_all = graph->alloc_vertex_subset();
    // VertexSubset * visited = graph->alloc_vertex_subset();
    // VertexSubset * active_in = graph->alloc_vertex_subset();

    double** global_num_paths = graph->alloc_global_vertex_array<double>();
    double** global_dependencies = graph->alloc_global_vertex_array<double>();
    VertexSubset** global_active_all = graph->alloc_global_vertex_subset();
    VertexSubset** global_visited = graph->alloc_global_vertex_subset();
    VertexSubset** global_active_in = graph->alloc_global_vertex_subset();
    MPI_Barrier(MPI_COMM_WORLD);
    double* num_paths = global_num_paths[graph->partition_id];
    double* dependencies = global_dependencies[graph->partition_id];
    VertexSubset* active_all = global_active_all[graph->partition_id];
    VertexSubset* visited = global_visited[graph->partition_id];
    VertexSubset* active_in = global_active_in[graph->partition_id];


    MPI_Barrier(MPI_COMM_WORLD);
    active_all->fill();
    std::vector<VertexSubset*> levels;
    std::vector<VertexSubset**> global_levels;

    VertexId active_vertices = 1;
    visited->clear();
    visited->set_bit(root);
    active_in->clear();
    active_in->set_bit(root);
    levels.push_back(active_in);
    global_levels.push_back(global_active_in);
    graph->fill_vertex_array(num_paths, 0.0);
    num_paths[root] = 1.0;
    VertexId i_i;
    if (graph->partition_id == 0) {
        // printf("forward\n");
    }
    for (i_i = 0; active_vertices > 0; i_i++) {
#ifdef SHOW_RESULT
        if (graph->partition_id == 0) {
            printf("active(%d)>=%u\n", i_i, active_vertices);
            // graph->print_process_data();
        }
        // printf("partition_id:%d, sequence:\n",graph->partition_id);
        // graph->print_get_sequence();
        #endif
        VertexSubset** global_active_out = graph->alloc_global_vertex_subset();
        MPI_Barrier(MPI_COMM_WORLD);
        // VertexSubset* active_out = graph->alloc_vertex_subset();
        VertexSubset* active_out = global_active_out[graph->partition_id];
        active_out->clear();
        graph->process_edges<VertexId, double>(
            [&](VertexId src) { graph->emit(src, num_paths[src]); },
            [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj, int partition_id) {
                if (partition_id == -1) {
                    for (AdjUnit<Empty>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        VertexId dst = ptr->neighbour;
                        CXL_PREFETCH
                        if (!visited->get_bit(dst)) {
                            if (num_paths[dst] == 0) {
                                active_out->set_bit(dst);
                            }
                            write_add(&num_paths[dst], msg);
                        }
                    }
                    return 0;
                } else {
                    for (AdjUnit<Empty>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        VertexId dst = ptr->neighbour;
                        CXL_PREFETCH
                        if (!global_visited[partition_id]->get_bit(dst)) {
                            if (global_num_paths[partition_id][dst] == 0) {
                                global_active_out[partition_id]->set_bit(dst);
                            }
                            write_add(&global_num_paths[partition_id][dst], msg);
                        }
                    }
                    return 0;
                }
            },
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj, int partition_id) {
                if(partition_id==-1){
                    if (visited->get_bit(dst)) return;
                    double sum = 0;
#ifdef OMP_SIMD
#    pragma omp simd reduction(+ : sum)
#endif
                    for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        VertexId src = ptr->neighbour;
                        CXL_PREFETCH
                        if (active_in->get_bit(src)) {
                            sum += num_paths[src];
                        }
                    }
                    if (sum > 0) {
                        graph->emit(dst, sum);
                    }
                }else{
                    if (global_visited[partition_id]->get_bit(dst)) return;
                    double sum = 0;
#ifdef OMP_SIMD
#    pragma omp simd reduction(+ : sum)
#endif
                    for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        VertexId src = ptr->neighbour;
                        CXL_PREFETCH
                        if (global_active_in[partition_id]->get_bit(src)) {
                            sum += global_num_paths[partition_id][src];
                        }
                    }
                    if (sum > 0) {
                        graph->emit_other(dst, sum,partition_id);
                    }
                }
            },
            [&](VertexId dst, double msg) {
                if (!visited->get_bit(dst)) {
                    active_out->set_bit(dst);
                    write_add(&num_paths[dst], msg);
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
        levels.push_back(active_out);
        global_levels.push_back(global_active_out);
        active_in = active_out;
        global_active_in = global_active_out;
    }

    double* inv_num_paths = num_paths;
#ifndef GLOBAL_STEALING_VERTICES
    graph->process_vertices<VertexId>(
        [&](VertexId vtx) {
            inv_num_paths[vtx] = 1 / num_paths[vtx];
            dependencies[vtx] = 0;
            return 1;
        },
        active_all);
    visited->clear();
    graph->process_vertices<VertexId>(
        [&](VertexId vtx) {
            visited->set_bit(vtx);
            dependencies[vtx] += inv_num_paths[vtx];
            return 1;
        },
        levels.back());
#else
    graph->process_vertices_global<VertexId>(
        [&](VertexId vtx, int partition_id) {
            if (partition_id == -1) {
                inv_num_paths[vtx] = 1 / num_paths[vtx];
                dependencies[vtx] = 0;
                return 1;
            } else {
                global_num_paths[partition_id][vtx] = 1 / global_num_paths[partition_id][vtx];
                global_dependencies[partition_id][vtx] = 0;
                return 1;
            }
        },
        global_active_all);
    visited->clear();
    graph->process_vertices_global<VertexId>(
        [&](VertexId vtx, int partition_id) {
            if (partition_id == -1) {
                visited->set_bit(vtx);
                dependencies[vtx] += inv_num_paths[vtx];
                return 1;
            } else {
                global_visited[partition_id]->set_bit(vtx);
                global_dependencies[partition_id][vtx] += global_num_paths[partition_id][vtx];
                return 1;
            }
        },
        global_levels.back());
#endif


    graph->transpose();
    // if (graph->partition_id == 0) {
    //     printf("backward\n");
    // }
    while (levels.size() > 1) {
        graph->process_edges<VertexId, double>(
            [&](VertexId src) { graph->emit(src, dependencies[src]); },
            [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj, int partition_id) {
                if (partition_id == -1) {
                    for (AdjUnit<Empty>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        VertexId dst = ptr->neighbour;
                        if (!visited->get_bit(dst)) {
                            write_add(&dependencies[dst], msg);
                        }
                    }
                    return 0;
                } else {
                    for (AdjUnit<Empty>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                        VertexId dst = ptr->neighbour;
                        if (!global_visited[partition_id]->get_bit(dst)) {
                            write_add(&global_dependencies[partition_id][dst], msg);
                        }
                    }
                    return 0;
                }
            },
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj, int partition_id) {
                if(partition_id==-1){
                    if (visited->get_bit(dst)) return;
                    double sum = 0;
                    for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        VertexId src = ptr->neighbour;
                        if (levels.back()->get_bit(src)) {
                            sum += dependencies[src];
                        }
                    }
                    graph->emit(dst, sum);
                }else{
                    if (global_visited[partition_id]->get_bit(dst)) return;
                    double sum = 0;
                    for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        VertexId src = ptr->neighbour;
                        if (global_levels.back()[partition_id]->get_bit(src)) {
                            sum += global_dependencies[partition_id][src];
                        }
                    }
                    graph->emit_other(dst, sum,partition_id);
                }
            },
            [&](VertexId dst, double msg) {
                if (!visited->get_bit(dst)) {
                    write_add(&dependencies[dst], msg);
                }
                return 0;
            },
            levels.back(),
            visited);
        delete levels.back();
        levels.pop_back();
        global_levels.pop_back();
#ifndef GLOBAL_STEALING_VERTICES
        graph->process_vertices<VertexId>(
            [&](VertexId vtx) {
                visited->set_bit(vtx);
                dependencies[vtx] += inv_num_paths[vtx];
                return 1;
            },
            levels.back());
#else
        graph->process_vertices_global<VertexId>(
            [&](VertexId vtx, int partition_id) {
                if (partition_id == -1) {
                    visited->set_bit(vtx);
                    dependencies[vtx] += inv_num_paths[vtx];
                    return 1;
                } else {
                    global_visited[partition_id]->set_bit(vtx);
                    global_dependencies[partition_id][vtx] += global_num_paths[partition_id][vtx];
                    return 1;
                }
            },
            global_levels.back());
#endif
    }
#ifndef GLOBAL_STEALING_VERTICES
    graph->process_vertices<VertexId>(
        [&](VertexId vtx) {
            dependencies[vtx] = (dependencies[vtx] - inv_num_paths[vtx]) / inv_num_paths[vtx];
            return 1;
        },
        active_all);
#else
    graph->process_vertices_global<VertexId>(
        [&](VertexId vtx, int partition_id) {
            if (partition_id == -1) {
                dependencies[vtx] = (dependencies[vtx] - inv_num_paths[vtx]) / inv_num_paths[vtx];
                return 1;
            } else {
                global_dependencies[partition_id][vtx] =
                    (global_dependencies[partition_id][vtx] - global_num_paths[partition_id][vtx]) /
                    global_num_paths[partition_id][vtx];
                return 1;
            }
        },
        global_active_all);
#endif
    graph->transpose();

    exec_time += get_time();
    times.push_back(exec_time);
    // if (graph->partition_id==0) {
    //   printf("exec_time=%lf(s)\n", exec_time);
    // }
    // printf("partition: %d,exec_time=%lf(s)\n", graph->get_partition_id(), exec_time);
    graph->gather_vertex_array(dependencies, 0);
    graph->gather_vertex_array(inv_num_paths, 0);
#ifdef SHOW_RESULT
    if (graph->partition_id == 0) {
        for (VertexId v_i = 0; v_i < 20; v_i++) {
            printf("%lf %lf\n", dependencies[v_i], 1 / inv_num_paths[v_i]);
        }
    }
    #endif

    graph->dealloc_vertex_array(dependencies);
    graph->dealloc_vertex_array(inv_num_paths);
    delete visited;
    delete active_all;
}

// an implementation which uses an array to store the levels instead of multiple bitmaps
void compute_compact(Graph<Empty>* graph, VertexId root) {
    double exec_time = 0;
    exec_time -= get_time();

    double* num_paths = graph->alloc_vertex_array<double>();
    double* dependencies = graph->alloc_vertex_array<double>();
    VertexSubset* active_all = graph->alloc_vertex_subset();
    active_all->fill();
    VertexSubset* visited = graph->alloc_vertex_subset();
    VertexId* level = graph->alloc_vertex_array<VertexId>();
    VertexSubset* active_in = graph->alloc_vertex_subset();
    VertexSubset* active_out = graph->alloc_vertex_subset();

    visited->clear();
    visited->set_bit(root);
    active_in->clear();
    active_in->set_bit(root);
    VertexId active_vertices = graph->process_vertices<VertexId>(
        [&](VertexId vtx) {
            if (active_in->get_bit(vtx)) {
                level[vtx] = 0;
                return 1;
            } else {
                level[vtx] = graph->vertices;
                return 0;
            }
        },
        active_all);
    graph->fill_vertex_array(num_paths, 0.0);
    num_paths[root] = 1.0;
    VertexId i_i;
    if (graph->partition_id == 0) {
        // printf("forward\n");
    }
    for (i_i = 0; active_vertices > 0; i_i++) {
        // if (graph->partition_id==0) {
        //   printf("active(%d)>=%u\n", i_i, active_vertices);
        // }
        active_out->clear();
        graph->process_edges<VertexId, double>(
            [&](VertexId src) { graph->emit(src, num_paths[src]); },
            [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj, int partition_id) {
                for (AdjUnit<Empty>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                    VertexId dst = ptr->neighbour;
                    if (!visited->get_bit(dst)) {
                        if (num_paths[dst] == 0) {
                            active_out->set_bit(dst);
                        }
                        write_add(&num_paths[dst], msg);
                    }
                }
                return 0;
            },
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj, int partition_id) {
                if (visited->get_bit(dst)) return;
                double sum = 0;
                for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                    VertexId src = ptr->neighbour;
                    if (active_in->get_bit(src)) {
                        sum += num_paths[src];
                    }
                }
                if (sum > 0) {
                    graph->emit(dst, sum);
                }
            },
            [&](VertexId dst, double msg) {
                if (!visited->get_bit(dst)) {
                    active_out->set_bit(dst);
                    write_add(&num_paths[dst], msg);
                }
                return 0;
            },
            active_in,
            visited);
        active_vertices = graph->process_vertices<VertexId>(
            [&](VertexId vtx) {
                visited->set_bit(vtx);
                level[vtx] = i_i + 1;
                return 1;
            },
            active_out);
        std::swap(active_in, active_out);
    }

    double* inv_num_paths = num_paths;
    graph->process_vertices<VertexId>(
        [&](VertexId vtx) {
            inv_num_paths[vtx] = 1 / num_paths[vtx];
            dependencies[vtx] = 0;
            return 1;
        },
        active_all);
    visited->clear();
    active_in->clear();
    graph->process_vertices<VertexId>(
        [&](VertexId vtx) {
            if (level[vtx] == i_i) {
                active_in->set_bit(vtx);
                return 1;
            }
            return 0;
        },
        active_all);
    graph->process_vertices<VertexId>(
        [&](VertexId vtx) {
            visited->set_bit(vtx);
            dependencies[vtx] += inv_num_paths[vtx];
            return 1;
        },
        active_in);
    graph->transpose();
    if (graph->partition_id == 0) {
        // printf("backward\n");
    }
    while (i_i > 0) {
        graph->process_edges<VertexId, double>(
            [&](VertexId src) { graph->emit(src, dependencies[src]); },
            [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj, int partition_id) {
                for (AdjUnit<Empty>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                    VertexId dst = ptr->neighbour;
                    if (!visited->get_bit(dst)) {
                        write_add(&dependencies[dst], msg);
                    }
                }
                return 0;
            },
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj, int partition_id) {
                if (visited->get_bit(dst)) return;
                double sum = 0;
                for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                    VertexId src = ptr->neighbour;
                    if (active_in->get_bit(src)) {
                        sum += dependencies[src];
                    }
                }
                graph->emit(dst, sum);
            },
            [&](VertexId dst, double msg) {
                if (!visited->get_bit(dst)) {
                    write_add(&dependencies[dst], msg);
                }
                return 0;
            },
            active_in,
            visited);
        i_i--;
        active_in->clear();
        active_vertices = graph->process_vertices<VertexId>(
            [&](VertexId vtx) {
                if (level[vtx] == i_i) {
                    active_in->set_bit(vtx);
                    return 1;
                }
                return 0;
            },
            active_all);
        graph->process_vertices<VertexId>(
            [&](VertexId vtx) {
                visited->set_bit(vtx);
                dependencies[vtx] += inv_num_paths[vtx];
                return 1;
            },
            active_in);
    }

    graph->process_vertices<VertexId>(
        [&](VertexId vtx) {
            dependencies[vtx] = (dependencies[vtx] - inv_num_paths[vtx]) / inv_num_paths[vtx];
            return 1;
        },
        active_all);
    graph->transpose();

    exec_time += get_time();
    // if (graph->partition_id==0) {
    //   printf("exec_time=%lf(s)\n", exec_time);
    // }
    printf("partition: %d,exec_time=%lf(s)\n", graph->get_partition_id(), exec_time);
    graph->gather_vertex_array(dependencies, 0);
    graph->gather_vertex_array(inv_num_paths, 0);
    if (graph->partition_id == 0) {
        // for (VertexId v_i=0;v_i<20;v_i++) {
        //   printf("%lf %lf\n", dependencies[v_i], 1 / inv_num_paths[v_i]);
        // }
    }

    graph->dealloc_vertex_array(dependencies);
    graph->dealloc_vertex_array(inv_num_paths);
    delete visited;
    delete active_all;
    delete active_in;
    delete active_out;
}

int main(int argc, char** argv) {
    MPI_Instance mpi(&argc, &argv);

    if (argc < 4) {
        printf("bc [file] [vertices] [root]\n");
        exit(-1);
    }

    Graph<Empty>* graph;
    graph = new Graph<Empty>();
    VertexId root = std::atoi(argv[3]);
    VertexId vertices = std::atoi(argv[2]);
    std::string base_filename = argv[1];
    bool loaded_from_preprocessed = graph->load_preprocessed_graph(base_filename);
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

#if COMPACT
    compute_compact(graph, root);
#else
    for (size_t i = 0; i < EXEC_TIMES; i++) {
        compute(graph, root);
    }
#endif
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
    return 0;
}
