#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;

/************************************************
  * 이 프로그램은 포즈 그래프 최적화를 위해 g2o 솔버를 사용하는 방법을 보여줍니다.
  * sphere.g2o는 인위적으로 생성된 포즈 그래프이므로 최적화해 보겠습니다.
  * 전체 그래프는 load 함수를 통해 직접 읽을 수 있지만 더 깊은 이해를 위해 읽기 코드를 직접 구현합니다.
  * 여기서 g2o/types/slam3d/의 SE3는 포즈를 나타내는 데 사용되며, 이는 본질적으로 lie 대수가 아니라 쿼터니언입니다.
 * **********************************************/

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // g2o 설정
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 그래프 모델
    optimizer.setAlgorithm(solver);   // 솔버 설정
    optimizer.setVerbose(true);       // 디버그 출력 설정

    int vertexCnt = 0, edgeCnt = 0; // vertex와 edge의 수
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // SE3 vertex
            g2o::VertexSE3 *v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 edge
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int idx1, idx2;     // 연결된 두 vertex
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    cout << "saving optimization results ..." << endl;
    optimizer.save("result.g2o");

    return 0;
}