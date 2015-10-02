#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

#define TIMER_BEGIN { double __time__ = cv::getTickCount();
#define TIMER_NOW   ((cv::getTickCount() - __time__) / cv::getTickFrequency())
#define TIMER_END   }

namespace lbf {

class Config {
public:
    static inline Config& GetInstance() {
        static Config c;
        return c;
    }

public:
    int stages_n;
    int tree_n;
    int tree_depth;

    std::string dataset;
    std::string saved_file_name;
    int landmark_n;
    int initShape_n;
    std::vector<int> feats_m;
    std::vector<double> radius_m;
    double bagging_overlap;
    std::vector<int> pupils[2];

private:
    Config();
    ~Config() {}
    Config(const Config &other);
    Config &operator=(const Config &other);
};

class BBox {
public:
    BBox();
    ~BBox();
    //BBox(const BBox &other);
    //BBox &operator=(const BBox &other);
    BBox(double x, double y, double w, double h);

public:
    cv::Mat Project(const cv::Mat &shape) const;
    cv::Mat ReProject(const cv::Mat &shape) const;

public:
    double x, y;
    double x_center, y_center;
    double x_scale, y_scale;
    double width, height;
};

void calcSimilarityTransform(const cv::Mat &shape1, const cv::Mat &shape2, double &scale, cv::Mat &rotate);

double calcVariance(const cv::Mat &vec);
double calcVariance(const std::vector<double> &vec);
double calcMeanError(std::vector<cv::Mat> &gt_shapes, std::vector<cv::Mat> &current_shapes);

cv::Mat getMeanShape(std::vector<cv::Mat> &gt_shapes, std::vector<BBox> &bboxes);
std::vector<cv::Mat> getDeltaShapes(std::vector<cv::Mat> &gt_shapes, std::vector<cv::Mat> &current_shapes, \
                                    std::vector<BBox> &bboxes, cv::Mat &mean_shape);

cv::Mat drawShapeInImage(const cv::Mat &img, const cv::Mat &shape, const BBox &bbox);

void LOG(const char *fmt, ...);

} // namespace lbf

#endif // COMMON_HPP_
