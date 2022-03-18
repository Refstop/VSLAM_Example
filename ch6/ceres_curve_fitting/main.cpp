#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// 비용 함수의 계산 모델
struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST ( double x, double y ) : _x ( x ), _y ( y ) {}
    // 잔차 계산
    template <typename T>
    bool operator() (
        const T* const abc,     // 3차원 모델 매개변수
        T* residual ) const     // 잔여
    {
        residual[0] = T ( _y ) - ceres::exp ( abc[0]*T ( _x ) *T ( _x ) + abc[1]*T ( _x ) + abc[2] ); // y-exp(ax^2+bx+c)
        return true;
    }
    const double _x, _y;    // x,y 데이터
};


int main ( int argc, char** argv )
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

    // 최소제곱 문제 만들기
    ceres::Problem problem;
    for ( int i=0; i<N; i++ )
    {
        problem.AddResidualBlock (     // 질문에 오류 용어 추가
        // 자동 파생 사용, 템플릿 매개변수: 오류 유형, 출력 차원, 입력 차원, 차원은 이전 구조체와 일치해야 합니다.
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3> ( 
                new CURVE_FITTING_COST ( x_data[i], y_data[i] )
            ),
            nullptr,            // 여기에서 사용되지 않는 커널 함수, 비어 있음
            abc                 // 추정할 매개변수
        );
    }

    // 솔버 구성
    ceres::Solver::Options options;     // 채울 수 있는 구성 항목이 많이 있습니다.
    options.linear_solver_type = ceres::DENSE_QR;  // 증분 방정식을 푸는 방법
    options.minimizer_progress_to_stdout = true;   // cout에 출력

    ceres::Solver::Summary summary;                // 최적화 정보
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 최적화 시작
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 출력 결과
    cout<<summary.BriefReport() <<endl;
    cout<<"estimated a,b,c = ";
    for ( auto a:abc ) cout<<a<<" ";
    cout<<endl;

    return 0;
}

