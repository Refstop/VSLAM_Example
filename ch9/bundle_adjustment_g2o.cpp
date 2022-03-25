#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

/// 카메라 pose와 내부 파라미터
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    /// set from given data address
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    /// 추정치를 메모리에 저장
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    SO3d rotation;
    Vector3d translation = Vector3d::Zero();
    double focal = 0;
    double k1 = 0, k2 = 0;
};

/// 포즈의 정점과 카메라 내부 매개변수, 9차원, 처음 3차원은 so3이고 다음은 t, f, k1, k2입니다.
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() {}

    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();
    }

    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    /// 추정치를 기반으로 점 투영
    Vector2d project(const Vector3d &point) {
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> { // edge의 생김새
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement; // 
    }

    // use numeric derivatives
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

};

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    cout << endl;
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size(); // 랜드마크 차원, 3
    const int camera_block_size = bal_problem.camera_block_size(); // 카메라 포즈 차원, 9
    double *points = bal_problem.mutable_points(); // 랜드마크의 정보가 저장된 배열의 시작 주소
    double *cameras = bal_problem.mutable_cameras(); // 카메라 포즈의 정보가 저장된 배열의 시작 주소
    cout << "bal_problem.num_cameras(): " << bal_problem.num_cameras() << endl;


    // pose dimension 9(x,y,z,r,p,y,f,k1,k2), landmark is 3(x,y,z)
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType; // 블록 솔버
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType; // 선형 솔버
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())); // optimizer 솔버로서 LM지정
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    /// build g2o problem
    const double *observations = bal_problem.observations();
    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics(); // 카메라 자세와 내부 파라미터(줄여서 내파) vertex
        double *camera = cameras + camera_block_size * i; // 카메라 포즈&내파가 저장된 배열의 주소
        v->setId(i); // 몇번 카메라인가
        v->setEstimate(PoseAndIntrinsics(camera)); // camera: 카메라 포즈&내파(x,y,z,r,p,y,f,k1,k2)
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }

    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i; // 랜드마크가 저장된 배열의 주소
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // g2o는 정점을 BA에서 Marg로 수동으로 설정해야 합니다.
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]); // edge 화살표 출발
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]); // edge 화살표 도착
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1])); // 관측값 저장
        edge->setInformation(Matrix2d::Identity()); // information matrix, 정보행렬(공분산과 비슷한 역할)
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge); // 엣지 추가, reprojection error는 edgeprojection 클래스 안에 computeerror 함수에서 담당
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }
}
