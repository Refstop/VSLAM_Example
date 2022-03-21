#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
using namespace std; 

// 곡선 모델의 정점, 템플릿 매개변수: 가변 차원 및 데이터 유형 최적화
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() // 초기화
    {
        _estimate << 0,0,0;
    }
    
    virtual void oplusImpl( const double* update ) // 고쳐 쓰다
    {
        _estimate += Eigen::Vector3d(update);
    }
    // 저장하고 읽기: 공백으로 둡니다.
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
};

// 오류 모델 템플릿 매개변수: 관찰 차원, 유형, 연결 꼭짓점 유형
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double x ): BaseUnaryEdge(), _x(x) {}
    // 곡선 모델 오류 계산
    void computeError()
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0,0) = _measurement - std::exp( abc(0,0)*_x*_x + abc(1,0)*_x + abc(2,0) ) ;
    }
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
public:
    double _x;  // x 값, y 값 _측정
};

int main( int argc, char** argv )
{
    double a=1.0, b=2.0, c=1.0;         // 실제 매개변수 값
    int N=100;                          // 데이터 포인트
    double w_sigma=1.0;                 // 노이즈 시그마 값
    cv::RNG rng;                        // OpenCV 난수 생성기
    double abc[3] = {0,0,0};            // abc 매개변수의 예상 값

    vector<double> x_data, y_data;      // 데이터
    
    cout<<"generating data: "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );
        y_data.push_back (
            exp ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma )
        );
        cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }

    // 그래프 최적화 빌드, 먼저 g2o 설정
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > Block;  // 각 오차항 최적화 변수의 차원은 3이고 오차값의 차원은 1
    // old
    // Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); // 선형 방정식 솔버
    // Block* solver_ptr = new Block( linearSolver );      // 행렬 블록 솔버
    // new
    std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverDense<Block::PoseMatrixType>()); // 선형 방정식 솔버
    std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver))); // 행렬 블록 솔버

    // 경사하강법, GN, LM, DogLeg 중에서 선택
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );
    g2o::SparseOptimizer optimizer;     // 그래프 모델
    optimizer.setAlgorithm( solver );   // 솔버 설정
    optimizer.setVerbose( true );       // 디버그 출력 켜기
    
    // 그래프에 정점 추가
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate( Eigen::Vector3d(0,0,0) );
    v->setId(0);
    optimizer.addVertex( v );
    
    // 그래프에 간선 추가
    for ( int i=0; i<N; i++ )
    {
        CurveFittingEdge* edge = new CurveFittingEdge( x_data[i] );
        edge->setId(i);
        edge->setVertex( 0, v );                // 연결된 정점 설정
        edge->setMeasurement( y_data[i] );      // 관측값
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma) ); // 정보 행렬: 공분산 행렬의 역
        optimizer.addEdge( edge );
    }
    
    // 최적화 수행
    cout<<"start optimization"<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
    
    // 출력 최적화 값
    Eigen::Vector3d abc_estimate = v->estimate();
    cout<<"estimated model: "<<abc_estimate.transpose()<<endl;
    
    return 0;
}