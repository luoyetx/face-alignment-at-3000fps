#include <cmath>
#include <ctime>
#include <cstdio>
#include <cassert>
#include <cstdarg>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lbf/common.hpp"

using namespace cv;
using namespace std;

namespace lbf {

BBox::BBox() {}
BBox::~BBox() {}
BBox::BBox(const BBox &other) {}
BBox &BBox::operator=(const BBox &other) {
    if (this == &other) return *this;
    return *this;
}

BBox::BBox(double x, double y, double w, double h) {
    this->x = x; this->y = y;
    this->width = w; this->height = h;
    this->x_center = x + w / 2.;
    this->y_center = y + h / 2.;
    this->x_scale = w / 2.;
    this->y_scale = h / 2.;
}

// Project absolute shape to relative shape binding to this bbox
Mat BBox::Project(const Mat &shape) const {
    Mat_<double> res(shape.rows, shape.cols);
    const Mat_<double> &shape_ = (Mat_<double>)shape;
    for (int i = 0; i < shape.rows; i++) {
        res(i, 0) = (shape_(i, 0) - x_center) / x_scale;
        res(i, 1) = (shape_(i, 1) - y_center) / y_scale;
    }
    return res;
}

// Project relative shape to absolute shape binding to this bbox
Mat BBox::ReProject(const Mat &shape) const {
    Mat_<double> res(shape.rows, shape.cols);
    const Mat_<double> &shape_ = (Mat_<double>)shape;
    for (int i = 0; i < shape.rows; i++) {
        res(i, 0) = shape_(i, 0)*x_scale + x_center;
        res(i, 1) = shape_(i, 1)*y_scale + y_center;
    }
    return res;
}


// Similarity Transform, project shape2 to shape1
// p1 ~= scale * rotate * p2, p1 and p2 are vector in math
void calcSimilarityTransform(const Mat &shape1, const Mat &shape2, double &scale, Mat &rotate) {
    Mat_<double> rotate_(2, 2);
    double x1_center, y1_center, x2_center, y2_center;
    x1_center = cv::mean(shape1.col(0))[0];
    y1_center = cv::mean(shape1.col(1))[0];
    x2_center = cv::mean(shape2.col(0))[0];
    y2_center = cv::mean(shape2.col(1))[0];

    Mat temp1(shape1.rows, shape1.cols, CV_64FC1);
    Mat temp2(shape2.rows, shape2.cols, CV_64FC1);
    temp1.col(0) = shape1.col(0) - x1_center;
    temp1.col(1) = shape1.col(1) - y1_center;
    temp2.col(0) = shape2.col(0) - x2_center;
    temp2.col(1) = shape2.col(1) - y2_center;

    Mat_<double> covar1, covar2;
    Mat_<double> mean1, mean2;
    calcCovarMatrix(temp1, covar1, mean1, CV_COVAR_COLS);
    calcCovarMatrix(temp2, covar2, mean2, CV_COVAR_COLS);

    double s1 = sqrt(cv::norm(covar1));
    double s2 = sqrt(cv::norm(covar2));
    scale = s1 / s2;
    temp1 /= s1;
    temp2 /= s2;

    double num = temp1.col(1).dot(temp2.col(0)) - temp1.col(0).dot(temp2.col(1));
    double den = temp1.col(0).dot(temp2.col(0)) + temp1.col(1).dot(temp2.col(1));
    double normed = sqrt(num*num + den*den);
    double sin_theta = num / normed;
    double cos_theta = den / normed;
    rotate_(0, 0) = cos_theta; rotate_(0, 1) = -sin_theta;
    rotate_(1, 0) = sin_theta; rotate_(1, 1) = cos_theta;
    rotate = rotate_;
}


double calcVariance(const Mat &vec) {
    double m1 = cv::mean(vec)[0];
    double m2 = cv::mean(vec.mul(vec))[0];
    double variance = m2 - m1*m1;
    return variance;
}

double calcVariance(const vector<double> &vec) {
    if (vec.size() == 0) return 0.;
    Mat_<double> vec_(vec);
    double m1 = cv::mean(vec_)[0];
    double m2 = cv::mean(vec_.mul(vec_))[0];
    double variance = m2 - m1*m1;
    return variance;
}

double calcMeanError(vector<Mat> &gt_shapes, vector<Mat> &current_shapes) {
    int N = gt_shapes.size();
    Config &config = Config::GetInstance();
    int landmark_n = config.landmark_n;
    vector<int> &left = config.pupils[0];
    vector<int> &right = config.pupils[1];

    double e = 0;
    // every train data
    for (int i = 0; i < N; i++) {
        const Mat_<double> &gt_shape = (Mat_<double>)gt_shapes[i];
        const Mat_<double> &current_shape = (Mat_<double>)current_shapes[i];
        double x1, y1, x2, y2;
        x1 = x2 = y1 = y2 = 0;
        for (int j = 0; j < left.size(); j++) {
            x1 += gt_shape(left[j], 0);
            y1 += gt_shape(left[j], 1);
        }
        for (int j = 0; j < right.size(); j++) {
            x2 += gt_shape(right[j], 0);
            y2 += gt_shape(right[j], 1);
        }
        x1 /= left.size(); y1 /= left.size();
        x2 /= right.size(); y2 /= right.size();
        double pupils_distance = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
        // every landmark
        double e_ = 0;
        for (int i = 0; i < landmark_n; i++) {
            e_ += norm(gt_shape.row(i) - current_shape.row(i));
        }
        e += e_ / pupils_distance;
    }
    e /= N*landmark_n;
    return e;
}

// Get mean_shape over all dataset
Mat getMeanShape(vector<Mat> &gt_shapes, vector<BBox> &bboxes) {
    int N = gt_shapes.size();
    Mat mean_shape = Mat::zeros(gt_shapes[0].rows, 2, CV_64FC1);
    for (int i = 0; i < N; i++) {
        mean_shape += bboxes[i].Project(gt_shapes[i]);
    }
    mean_shape /= N;
    return mean_shape;
}

// Get relative delta_shapes for predicting target
vector<Mat> getDeltaShapes(vector<Mat> &gt_shapes, vector<Mat> &current_shapes, \
    vector<BBox> &bboxes, Mat &mean_shape) {
    vector<Mat> delta_shapes;
    int N = gt_shapes.size();
    delta_shapes.resize(N);
    double scale;
    Mat_<double> rotate;
    for (int i = 0; i < N; i++) {
        delta_shapes[i] = bboxes[i].Project(gt_shapes[i]) - bboxes[i].Project(current_shapes[i]);
        calcSimilarityTransform(mean_shape, bboxes[i].Project(current_shapes[i]), scale, rotate);
        delta_shapes[i] = scale * delta_shapes[i] * rotate.t();
    }
    return delta_shapes;
}

// Draw landmarks with bbox on image
Mat drawShapeInImage(const Mat &img, const Mat &shape, const BBox &bbox) {
    Mat img_ = img.clone();
    rectangle(img_, Rect(bbox.x, bbox.y, bbox.width, bbox.height), Scalar(0, 0, 255), 2);
    for (int i = 0; i < shape.rows; i++) {
        circle(img_, Point(shape.at<double>(i, 0), shape.at<double>(i, 1)), 2, Scalar(0, 255, 0), -1);
    }
    return img_;
}

// Logging with timestamp, message sholdn't be too long
void LOG(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char msg[256];
    vsprintf(msg, fmt, args);
    va_end(args);

    char buff[256];
    time_t t = time(NULL);
    strftime(buff, sizeof(buff), "[%x - %X]", localtime(&t));
    printf("%s %s\n", buff, msg);
}


Config::Config() {
    dataset = "../data/68";
    saved_file_name = "../model/68.model";
    stages_n = 5;
    tree_n = 6;
    tree_depth = 5;
    landmark_n = 68;
    initShape_n = 10;
    bagging_overlap = 0.4;
    int pupils[][6] = { { 36, 37, 38, 39, 40, 41 }, { 42, 43, 44, 45, 46, 47 } };
    int feats_m[] = { 500, 500, 500, 300, 300, 300, 200, 200, 200, 100 };
    double radius_m[] = { 0.3, 0.2, 0.15, 0.12, 0.10, 0.10, 0.08, 0.06, 0.06, 0.05 };
    for (int i = 0; i < 6; i++) {
        this->pupils[0].push_back(pupils[0][i]);
        this->pupils[1].push_back(pupils[1][i]);
    }
    for (int i = 0; i < 10; i++) {
        this->feats_m.push_back(feats_m[i]);
        this->radius_m.push_back(radius_m[i]);
    }
}

} // namespace lbf
