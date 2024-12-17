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

double exec_time = 0;
void compute(Graph<Empty> * graph, VertexId root) {
    // double exec_time = 0;
    exec_time -= get_time();

    // VertexId* parent = graph->alloc_vertex_array<VertexId>();
    // VertexSubset* visited = graph->alloc_vertex_subset();
    // VertexSubset* active_in = graph->alloc_vertex_subset();
    // VertexSubset* active_out = graph->alloc_vertex_subset();
    VertexSubset** global_visited = graph->alloc_global_vertex_subset();
    VertexSubset** global_active_in = graph->alloc_global_vertex_subset();
    VertexSubset** global_active_out = graph->alloc_global_vertex_subset();
    VertexSubset* visited = global_visited[graph->partition_id];
    VertexSubset* active_in = global_active_in[graph->partition_id];
    VertexSubset* active_out = global_active_out[graph->partition_id];
    VertexId** global_parent = graph->alloc_global_vertex_array<VertexId>();
    VertexId* parent = global_parent[graph->partition_id];



    visited->clear();
    visited->set_bit(root);
    active_in->clear();
    active_in->set_bit(root);
    graph->fill_vertex_array(parent, graph->vertices);
    parent[root] = root;
    MPI_Barrier(MPI_COMM_WORLD);
    VertexId active_vertices = 1;

    for (int i_i = 0; active_vertices > 0; i_i++) {
        if (graph->partition_id==0) {
          printf("active(%d)>=%u\n", i_i, active_vertices);
        }
        active_out->clear();
        active_vertices = graph->process_edges<VertexId, VertexId>(
            [&](VertexId src) { graph->emit(src, src); },
            [&](VertexId src, VertexId msg, VertexAdjList<Empty> outgoing_adj, int partition_id) {
                VertexId activated = 0;
                for (AdjUnit<Empty>* ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {
                    VertexId dst = ptr->neighbour;
                    if (parent[dst] == graph->vertices && cas(&parent[dst], graph->vertices, src)) {
                        active_out->set_bit(dst);
                        activated += 1;
                    }
                }
                return activated;
            },
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj, int partition_id) {
                if(partition_id==-1){
                    if (visited->get_bit(dst)) return;
                    for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        VertexId src = ptr->neighbour;
                        if (active_in->get_bit(src)) {
                            graph->emit(dst, src);
                            break;
                        }
                    }
                }else{
                    if (global_visited[partition_id]->get_bit(dst)) return;
                    for (AdjUnit<Empty>* ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {
                        VertexId src = ptr->neighbour;
                        if (global_active_in[partition_id]->get_bit(src)) {
                            graph->emit_other(dst, src,partition_id);
                            break;
                        }
                    }
                }
                
            },
            [&](VertexId dst, VertexId msg) {
                if (cas(&parent[dst], graph->vertices, msg)) {
                    active_out->set_bit(dst);
                    return 1;
                }
                return 0;
            },
            active_in,
            visited);
        active_vertices = graph->process_vertices<VertexId>(   //用来标记点访问过了
            [&](VertexId vtx) {
                visited->set_bit(vtx);
                return 1;
            },
            active_out);
        std::swap(active_in, active_out);
    }

  exec_time += get_time();
  // if (graph->partition_id==0) {
  //   printf("exec_time=%lf(s)\n", exec_time);
  // }
  printf("partition: %d,exec_time=%lf(s)\n", graph->get_partition_id(), exec_time);
  graph->gather_vertex_array(parent, 0);

  // if (graph->partition_id==0) {
  //   VertexId found_vertices = 0;
  //   for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
  //     if (parent[v_i] < graph->vertices) {
  //       found_vertices += 1;
  //     }
  //   }
  //   printf("found_vertices = %u\n", found_vertices);
  // }

  graph->dealloc_vertex_array(parent);
  delete active_in;
  delete active_out;
  delete visited;
}

int main(int argc, char ** argv) {
  MPI_Instance mpi(&argc, &argv);

  if (argc<4) {
    printf("bfs [file] [vertices] [root]\n");
    exit(-1);
  }

  Graph<Empty> * graph;
  graph = new Graph<Empty>();
  VertexId root = std::atoi(argv[3]);
  double load_time = 0;
  load_time -= get_time();
  graph->load_directed(argv[1], std::atoi(argv[2]));
  printf("load complete\n");
  load_time += get_time();
  // printf("load_time=%lf(s)\n", load_time);
  compute(graph, root);
  printf("partiton_id: %d, total_process_time  =%lf(s)\n",
         graph->get_partition_id(),
         graph->print_total_process_time());
  // for (int run=0;run<5;run++) {
  //   compute(graph, root);
  // }

//   delete graph;
  return 0;
}
