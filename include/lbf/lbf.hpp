#ifndef LBF_HPP_
#define LBF_HPP_

#include <vector>
#include "lbf/rf.hpp"

namespace lbf {

class LbfCascador {
public:
    LbfCascador();
    ~LbfCascador();
    LbfCascador(const LbfCascador &other);
    LbfCascador &operator=(const LbfCascador &other);

public:
    void Init(int stages_n);
    void Train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, \
               std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes, \
               cv::Mat &mean_shape, int start_from = 0);
    void Test(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, std::vector<BBox> &bboxes);
    void GlobalRegressionTrain(std::vector<cv::Mat> &lbfs, std::vector<cv::Mat> &deltashapes, int stage);
    cv::Mat GlobalRegressionPredict(const cv::Mat &lbf, int stage);
    cv::Mat Predict(cv::Mat &img, BBox &bbox);
    void DumpTrainModel(int stage);
    void ResumeTrainModel(int start_from = 0);

    void Read(FILE *fd);
    void Write(FILE *fd);

public:
    int stages_n;
    int landmark_n;
    cv::Mat mean_shape;
    std::vector<RandomForest> random_forests;
    std::vector<cv::Mat> gl_regression_weights;
};

} // namespace lbf

#endif // LBF_HPP_
