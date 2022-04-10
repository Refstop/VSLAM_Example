#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>

// for sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

/**********************************************
* 이 프로그램은 알려진 궤적을 가진 단안 카메라의 조밀한 깊이 추정을 보여줍니다.
* 책의 섹션 12.2에 해당하는 polar search + NCC matching 방법 사용
* 이 프로그램은 완벽하지 않습니다. 분명히 개선할 수 있습니다. 실제로 의도적으로 몇 가지 문제를 노출하고 있습니다(변명입니다).
***********************************************/

// ------------------------------------------------------------------
// parameters
const int boarder = 20;         // 가장자리 너비
const int width = 640;          // 이미지 너비
const int height = 480;         // 이미지 높이
const double fx = 481.2f;       // 카메라 내부 매개변수
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;    // NCC에서 가져온 창의 절반 너비
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC 창 영역
const double min_cov = 0.1;     // 수렴 판정: 최소 분산
const double max_cov = 10;      // 발산 판정: 최대 분산

// ------------------------------------------------------------------
// 중요한 함수
// REMODE 데이터셋에서 데이터 읽기
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

/**
 * 새 이미지를 기반으로 깊이 추정 업데이트
 * 
 * @param ref           참조 이미지
 * @param curr          현재 이미지
 * @param T_C_R         참조 이미지에서 현재 이미지까지의 포즈
 * @param depth         깊이
 * @param depth_cov     깊이 분산
 * @return              성공 여부
 */
bool update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 극지 탐색
 * 
 * @param ref           참조 이미지
 * @param curr          현재 이미지
 * @param T_C_R         포즈
 * @param pt_ref        참조 이미지에서 점의 위치
 * @param depth_mu      깊이 평균
 * @param depth_cov     깊이 분산
 * @param pt_curr       현재 지점
 * @param epipolar_direction  에피폴라 라인의 방향?
 * @return              성공 여부
 */
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction
);

/**
 * 깊이 필터 업데이트
 * @param pt_ref    참조 이미지 포인트
 * @param pt_curr   현재 이미지 포인트
 * @param T_C_R     포즈
 * @param epipolar_direction 에피폴라 라인의 방향?
 * @param depth     깊이 평균
 * @param depth_cov2    깊이 방향
 * @return          성공 여부
 */
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * NCC 점수 계산
 * @param ref       참조 이미지
 * @param curr      현재 이미지
 * @param pt_ref    기준점
 * @param pt_curr   현재 지점
 * @return          NCC 점수
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// 쌍선형 그레이스케일 보간
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

// ------------------------------------------------------------------
// 일부 가젯
// 예상 깊이 맵 표시
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// 픽셀 대 카메라 좌표계
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 카메라 좌표계를 픽셀로
inline Vector2d cam2px(const Vector3d p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// 점이 이미지 블럭 안에 있는지 감지
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

// 에피폴라 매치를 보여주다
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// 에피폴라 라인 표시
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

/// 깊이 추정 평가
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------------------------


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }

    // 데이터세트에서 데이터 읽기
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // 첫 번째 사진
    Mat ref = imread(color_image_files[0], 0);                // gray-scale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;    // 깊이 초기값
    double init_cov2 = 3.0;     // 편차 초기값
    Mat depth(height, width, CV_64F, init_depth);             // 깊이 맵
    Mat depth_cov2(height, width, CV_64F, init_cov2);         // 깊이 맵 분산

    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // 좌표 변환 관계： T_C_W * T_W_R = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3d> &poses,
    cv::Mat &ref_depth) {
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // 데이터 형식: 이미지 파일 이름 tx, ty, tz, qx, qy, qz, qw , TCW가 아닌 TWC임을 유의하십시오.
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}

// 전체 깊이 맵 업데이트
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    for (int x = boarder; x < width - boarder; x++)
        for (int y = boarder; y < height - boarder; y++) {
            // 각 픽셀에 대해 반복
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov) // 깊이가 수렴하거나 발산했습니다.
                continue;
            // 에피폴라 라인에서 (x,y) 일치 검색
            Vector2d pt_curr;
            Vector2d epipolar_direction;
            bool ret = epipolarSearch(
                ref,
                curr,
                T_C_R,
                Vector2d(x, y),
                depth.ptr<double>(y)[x],
                sqrt(depth_cov2.ptr<double>(y)[x]),
                pt_curr,
                epipolar_direction
            );

            if (ret == false) // 경기 실패
                continue;

            // 일치 항목을 표시하려면 주석 처리를 제거하십시오.
            // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // 일치가 성공하면 깊이 맵을 업데이트합니다.
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
}

// 극지 탐색
// 이 방법은 책의 12.2절과 12.3절에 설명되어 있습니다.
bool epipolarSearch(
    const Mat &ref, const Mat &curr,
    const SE3d &T_C_R, const Vector2d &pt_ref,
    const double &depth_mu, const double &depth_cov,
    Vector2d &pt_curr, Vector2d &epipolar_direction) {
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;    // 참조 프레임의 P 벡터

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // 깊이 평균으로 투영된 픽셀
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));    // 최소 깊이로 투영된 픽셀
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));    // 최대 깊이로 투영된 픽셀

    Vector2d epipolar_line = px_max_curr - px_min_curr;    // 에피폴라 라인(선분 형태)
    epipolar_direction = epipolar_line;        // 에피폴라 라인 방향
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();    // 에피폴라 세그먼트의 절반 길이
    if (half_length > 100) half_length = 100;   // 우리는 너무 많이 검색하고 싶지 않습니다

    // 에피폴라 라인(선분)을 표시하려면 이 문장의 주석을 제거하십시오.
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 에피폴라 라인을 찾아 깊이 평균을 중심으로 좌우 길이의 반을 취한다.
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;  // 일치하는 포인트
        if (!inside(px_curr))
            continue;
        // 매칭하고자 하는 점과 기준 좌표계의 NCC 계산
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f)      // 높은 NCC 매치만 신뢰
        return false;
    pt_curr = best_px_curr;
    return true;
}

double NCC(
    const Mat &ref, const Mat &curr,
    const Vector2d &pt_ref, const Vector2d &pt_curr) {
    // 제로 평균 정규화 상호 상관
    // 먼저 의미
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // 기준 좌표계와 현재 좌표계의 평균
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // 제로 평균 NCC 계산
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   // 분모에서 0을 방지
}

bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2) {
    // 이걸 보고 있는 사람이 있을지 모르겠다
    // 삼각 측량으로 깊이 계산  
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    // 방정식
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 다음 행렬 방정식 시스템으로 변환
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Vector2d ans = A.inverse() * b;
    Vector3d xm = ans[0] * f_ref;           // ref 측의 결과
    Vector3d xn = t + ans[1] * f2;          // cur 결과
    Vector3d p_esti = (xm + xn) / 2.0;      // P의 위치는 두 값의 평균을 취합니다.
    double depth_estimation = p_esti.norm();   // 깊이 값

    // 불확실성 계산(1픽셀의 오류)
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // 가우스 융합
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

// 후자는 너무 간단해서 언급하지 않겠습니다(사실 게으름 때문에)
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    double ave_depth_error = 0;     // 평균 오차
    double ave_depth_error_sq = 0;      // 제곱 오차
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr) {

    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}
